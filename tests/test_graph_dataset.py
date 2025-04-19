"""Tests for GraphDataset class"""

import pytest
import os
import numpy as np
import torch
import zarr
from datetime import datetime
from pathlib import Path
import shutil
from unittest.mock import Mock, patch

from neural_lam.graph_dataset import GraphDataset
from neural_lam.create_mesh_graph import create_grid_to_mesh, create_mesh_to_grid

class TestGraphDataset:
    @pytest.fixture
    def mock_mesh_graph(self):
        """Create a mock mesh graph for testing"""
        import networkx as nx
        G = nx.grid_2d_graph(5, 5)
        pos = {(i, j): np.array([i, j]) for i, j in G.nodes()}
        nx.set_node_attributes(G, pos, 'pos')
        return nx.convert_node_labels_to_integers(G)
    
    @pytest.fixture
    def sample_weather_data(self):
        """Create sample weather data for testing"""
        return {
            'temperature': (
                torch.randn(10, 10),
                torch.ones(10, 10)
            ),
            'wind': (
                torch.randn(10, 10, 2),
                torch.ones(10, 10)
            ),
            'pressure': (
                torch.randn(10, 10),
                torch.ones(10, 10)
            )
        }
    
    @pytest.fixture
    def sample_data_path(self, tmp_path):
        """Create a sample Zarr dataset"""
        data_path = tmp_path / "test_data.zarr"
        
        # Create sample data
        root = zarr.open(str(data_path), mode='w')
        
        # Add time dimension
        times = np.array(['2025-01-01T00:00:00', '2025-01-01T01:00:00'], dtype='datetime64[s]')
        root.create_dataset('time', data=times)
        
        # Add sample weather data
        shape = (2, 10, 10)  # time, lat, lon
        root.create_dataset('temperature', data=np.random.rand(*shape))
        root.create_dataset('pressure', data=np.random.rand(*shape))
        
        return data_path
    
    @pytest.fixture
    def dataset_config(self):
        """Sample dataset configuration"""
        return {
            'mesh_resolution': 0.1,
            'cutoff_factor': 0.67,
            'num_neighbors': 4,
            'start_date': '2025-01-01',
            'end_date': '2025-01-02',
            'satellite_id': 'test_sat'
        }
    
    def test_dataset_initialization(self, sample_data_path, dataset_config):
        """Test basic dataset initialization"""
        dataset = GraphDataset(
            data_path=str(sample_data_path),
            save_path=str(Path(sample_data_path).parent),
            **dataset_config
        )
        
        assert dataset.data_path == str(sample_data_path)
        assert dataset.mesh_resolution == dataset_config['mesh_resolution']
        assert dataset.cutoff_factor == dataset_config['cutoff_factor']
    
    def test_prepare_data(self, sample_data_path, dataset_config):
        """Test data preparation"""
        dataset = GraphDataset(
            data_path=str(sample_data_path),
            save_path=str(Path(sample_data_path).parent),
            **dataset_config
        )
        
        # Should not raise error
        dataset.prepare_data()
        
        # Test with invalid path
        with pytest.raises(RuntimeError):
            invalid_dataset = GraphDataset(
                data_path="/invalid/path.zarr",
                save_path="/tmp",
                **dataset_config
            )
            invalid_dataset.prepare_data()
    
    def test_mesh_structure_creation(self, sample_data_path, dataset_config):
        """Test mesh structure creation and caching"""
        dataset = GraphDataset(
            data_path=str(sample_data_path),
            save_path=str(Path(sample_data_path).parent),
            **dataset_config
        )
        
        # Create mesh structure
        grid_coords = np.meshgrid(
            np.linspace(0, 1, 10),
            np.linspace(0, 1, 10)
        )
        
        args = {
            'cutoff': dataset.cutoff_factor,
            'num_neighbors': dataset.num_neighbors
        }
        
        mesh_structure = dataset.create_mesh_structure(
            xy=grid_coords,
            args=args,
            graph_dir_path=dataset.save_path
        )
        
        # Verify structure
        assert 'g2m_graph' in mesh_structure
        assert 'm2g_graph' in mesh_structure
        assert 'edge_features' in mesh_structure
    
    def test_cache_management(self, sample_data_path, dataset_config):
        """Test mesh structure caching"""
        dataset = GraphDataset(
            data_path=str(sample_data_path),
            save_path=str(Path(sample_data_path).parent),
            **dataset_config
        )
        
        # Get cache paths
        cache_paths = dataset._get_cache_paths()
        
        # Verify cache directory creation
        cache_dir = Path(dataset.save_path) / "mesh_cache"
        assert cache_dir.exists()
        
        # Verify cache key format
        cache_key = f"mesh_{dataset.mesh_resolution}_{dataset.cutoff_factor}_{dataset.num_neighbors}"
        assert all(str(cache_key) in str(path) for path in cache_paths.values())
    
    @pytest.mark.parametrize("cache_exists", [True, False])
    def test_mesh_structure_caching(self, sample_data_path, dataset_config, cache_exists):
        """Test mesh structure caching behavior"""
        dataset = GraphDataset(
            data_path=str(sample_data_path),
            save_path=str(Path(sample_data_path).parent),
            **dataset_config
        )
        
        cache_paths = dataset._get_cache_paths()
        
        if cache_exists:
            # Create dummy cache files
            for path in cache_paths.values():
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(torch.tensor([1.0]), path)
        else:
            # Remove cache directory if it exists
            cache_dir = Path(dataset.save_path) / "mesh_cache"
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
        
        # Create mesh structure
        grid_coords = np.meshgrid(
            np.linspace(0, 1, 10),
            np.linspace(0, 1, 10)
        )
        
        args = {
            'cutoff': dataset.cutoff_factor,
            'num_neighbors': dataset.num_neighbors
        }
        
        mesh_structure = dataset.create_mesh_structure(
            xy=grid_coords,
            args=args,
            graph_dir_path=dataset.save_path
        )
        
        # Verify cache files exist
        assert all(Path(path).exists() for path in cache_paths.values())
    
    def test_data_loading(self, sample_data_path, dataset_config):
        """Test data loading from Zarr dataset"""
        dataset = GraphDataset(
            data_path=str(sample_data_path),
            save_path=str(Path(sample_data_path).parent),
            **dataset_config
        )
        
        dataset.setup()
        
        # Verify data loading
        assert hasattr(dataset, 'z')
        assert hasattr(dataset, 'data_summary')
        assert hasattr(dataset, 'mesh_structure')
    
    def test_dynamic_graph_creation(self, sample_data_path, dataset_config):
        """Test dynamic graph creation"""
        dataset = GraphDataset(
            data_path=str(sample_data_path),
            save_path=str(Path(sample_data_path).parent),
            **dataset_config
        )
        
        # Setup dataset
        dataset.setup()
        
        # Create sample weather data
        weather_data = {
            'temperature': (
                torch.randn(10, 10),  # Values
                torch.ones(10, 10)    # Mask
            ),
            'pressure': (
                torch.randn(10, 10),  # Values
                torch.ones(10, 10)    # Mask
            )
        }
        
        # Create dynamic graph
        graph = dataset.create_dynamic_graph(weather_data)
        
        # Verify graph properties
        assert hasattr(graph, 'x')
        assert hasattr(graph, 'edge_index')
        assert hasattr(graph, 'mesh_structure')
    
    def test_graph_persistence(self, sample_data_path, dataset_config, mock_mesh_graph):
        """Test graph structure persistence across dataset instances"""
        # Create first dataset instance
        dataset1 = GraphDataset(
            data_path=str(sample_data_path),
            save_path=str(Path(sample_data_path).parent),
            **dataset_config
        )
        
        grid_coords = np.meshgrid(
            np.linspace(0, 1, 10),
            np.linspace(0, 1, 10)
        )
        
        # Create and cache mesh structure
        mesh_structure1 = dataset1.create_mesh_structure(
            xy=grid_coords,
            args={'cutoff': dataset1.cutoff_factor, 'num_neighbors': dataset1.num_neighbors},
            graph_dir_path=dataset1.save_path
        )
        
        # Create second dataset instance with same parameters
        dataset2 = GraphDataset(
            data_path=str(sample_data_path),
            save_path=str(Path(sample_data_path).parent),
            **dataset_config
        )
        
        # Get mesh structure from cache
        mesh_structure2 = dataset2.create_mesh_structure(
            xy=grid_coords,
            args={'cutoff': dataset2.cutoff_factor, 'num_neighbors': dataset2.num_neighbors},
            graph_dir_path=dataset2.save_path
        )
        
        # Verify structures are identical
        assert torch.equal(mesh_structure1['g2m_graph'].edge_index,
                         mesh_structure2['g2m_graph'].edge_index)
        assert torch.equal(mesh_structure1['m2g_graph'].edge_index,
                         mesh_structure2['m2g_graph'].edge_index)
    
    def test_graph_feature_computation(self, sample_data_path, dataset_config, sample_weather_data):
        """Test computation of graph features"""
        dataset = GraphDataset(
            data_path=str(sample_data_path),
            save_path=str(Path(sample_data_path).parent),
            **dataset_config
        )
        dataset.setup()
        
        graph = dataset.create_dynamic_graph(sample_weather_data)
        
        # Test feature dimensions
        for obs_type, (values, mask) in sample_weather_data.items():
            assert obs_type in graph.obs_features
            if values.dim() == 3:  # Vector observation
                assert graph.obs_features[obs_type].shape[-1] == values.shape[-1]
            else:  # Scalar observation
                assert graph.obs_features[obs_type].shape[-1] == 1
        
        # Test edge features
        assert hasattr(graph, 'edge_attr')
        assert graph.edge_attr.shape[1] > 0  # Should have at least some features
    
    def test_error_handling(self, sample_data_path, dataset_config):
        """Test error handling"""
        # Test with invalid dates
        with pytest.raises(ValueError):
            invalid_config = dataset_config.copy()
            invalid_config['start_date'] = '2025-01-02'
            invalid_config['end_date'] = '2025-01-01'
            
            dataset = GraphDataset(
                data_path=str(sample_data_path),
                save_path=str(Path(sample_data_path).parent),
                **invalid_config
            )
            dataset.setup()
        
        # Test with missing data
        with pytest.raises(KeyError):
            dataset = GraphDataset(
                data_path=str(sample_data_path),
                save_path=str(Path(sample_data_path).parent),
                **dataset_config
            )
            dataset.create_dynamic_graph({'invalid_var': (torch.randn(10, 10), torch.ones(10, 10))})
    
    def test_heterogeneous_observations(self, sample_data_path, dataset_config):
        """Test handling of heterogeneous observation types"""
        dataset = GraphDataset(
            data_path=str(sample_data_path),
            save_path=str(Path(sample_data_path).parent),
            **dataset_config
        )
        dataset.setup()
        
        # Create sample heterogeneous weather data
        weather_data = {
            'temperature': (
                torch.randn(10, 10),  # Values
                torch.ones(10, 10)    # Mask
            ),
            'wind': (
                torch.randn(10, 10, 2),  # U,V components
                torch.ones(10, 10)      # Mask
            ),
            'pressure': (
                torch.randn(10, 10),  # Values
                torch.ones(10, 10)    # Mask
            )
        }
        
        # Create dynamic graph
        graph = dataset.create_dynamic_graph(weather_data)
        
        # Verify heterogeneous features
        assert 'temperature' in graph.obs_features
        assert 'wind' in graph.obs_features
        assert 'pressure' in graph.obs_features
        
        # Check vector observation handling
        assert graph.obs_features['wind'].shape[-1] == 2
    
    def test_mesh_boundary_integration(self, sample_data_path, dataset_config):
        """Test integration of boundary detection with dataset"""
        dataset = GraphDataset(
            data_path=str(sample_data_path),
            save_path=str(Path(sample_data_path).parent),
            **dataset_config
        )
        
        grid_coords = np.meshgrid(
            np.linspace(0, 1, 10),
            np.linspace(0, 1, 10)
        )
        
        # Create mesh structure with boundary detection
        args = {
            'cutoff': dataset.cutoff_factor,
            'num_neighbors': dataset.num_neighbors,
            'include_boundary_mask': True
        }
        
        mesh_structure = dataset.create_mesh_structure(
            xy=grid_coords,
            args=args,
            graph_dir_path=dataset.save_path
        )
        
        # Verify boundary masks
        assert 'boundary_mask' in mesh_structure
        assert hasattr(mesh_structure['g2m_graph'], 'boundary_mask')
        assert hasattr(mesh_structure['m2g_graph'], 'boundary_mask')
    
    def test_adaptive_mesh_structure(self, sample_data_path, dataset_config):
        """Test adaptive mesh structure creation"""
        dataset = GraphDataset(
            data_path=str(sample_data_path),
            save_path=str(Path(sample_data_path).parent),
            **dataset_config
        )
        
        # Create non-uniform grid
        x = np.concatenate([
            np.linspace(0, 0.5, 5),   # Dense region
            np.linspace(0.6, 1.0, 3)   # Sparse region
        ])
        y = x.copy()
        grid_coords = np.meshgrid(x, y)
        
        args = {
            'cutoff': dataset.cutoff_factor,
            'num_neighbors': dataset.num_neighbors,
            'adaptive_neighbors': True
        }
        
        mesh_structure = dataset.create_mesh_structure(
            xy=grid_coords,
            args=args,
            graph_dir_path=dataset.save_path
        )
        
        # Verify adaptive structure
        g2m_graph = mesh_structure['g2m_graph']
        edge_index = g2m_graph.edge_index
        
        # Count neighbors for each node
        neighbor_counts = torch.bincount(edge_index[0])
        
        # Dense regions should have more neighbors
        assert neighbor_counts[:25].mean() > neighbor_counts[25:].mean()
    
    def test_mesh_resolution_effects(self, sample_data_path, dataset_config):
        """Test effects of different mesh resolutions"""
        resolutions = [0.1, 0.2, 0.5]
        grid_sizes = []
        
        for res in resolutions:
            config = dataset_config.copy()
            config['mesh_resolution'] = res
            
            dataset = GraphDataset(
                data_path=str(sample_data_path),
                save_path=str(Path(sample_data_path).parent),
                **config
            )
            
            grid_coords = np.meshgrid(
                np.linspace(0, 1, int(1/res)),
                np.linspace(0, 1, int(1/res))
            )
            
            mesh_structure = dataset.create_mesh_structure(
                xy=grid_coords,
                args={'cutoff': dataset.cutoff_factor, 'num_neighbors': dataset.num_neighbors},
                graph_dir_path=dataset.save_path
            )
            
            grid_sizes.append(mesh_structure['g2m_graph'].num_nodes)
        
        # Verify that higher resolution (lower value) creates more nodes
        assert grid_sizes[0] > grid_sizes[1] > grid_sizes[2]
