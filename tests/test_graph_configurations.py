"""Tests for different graph configurations in GraphDataset"""

import pytest
import numpy as np
import torch
import zarr
from pathlib import Path
import networkx as nx
from torch_geometric.data import Data

from neural_lam.graph_dataset import GraphDataset
from neural_lam.create_mesh_graph import create_grid_to_mesh, create_mesh_to_grid

class TestGraphConfigurations:
    @pytest.fixture
    def sample_data_path(self, tmp_path):
        """Create a sample Zarr dataset"""
        data_path = tmp_path / "test_data.zarr"
        root = zarr.open(str(data_path), mode='w')
        times = np.array(['2025-01-01T00:00:00'], dtype='datetime64[s]')
        root.create_dataset('time', data=times)
        shape = (1, 20, 20)
        root.create_dataset('temperature', data=np.random.rand(*shape))
        return data_path
    
    @pytest.fixture
    def base_config(self):
        """Base configuration for dataset"""
        return {
            'mesh_resolution': 0.1,
            'cutoff_factor': 0.67,
            'num_neighbors': 4,
            'start_date': '2025-01-01',
            'end_date': '2025-01-02',
            'satellite_id': 'test_sat'
        }
    
    def test_different_mesh_topologies(self, sample_data_path, base_config):
        """Test different mesh topologies"""
        dataset = GraphDataset(
            data_path=str(sample_data_path),
            save_path=str(Path(sample_data_path).parent),
            **base_config
        )
        
        # Test regular grid
        regular_coords = np.meshgrid(
            np.linspace(0, 1, 10),
            np.linspace(0, 1, 10)
        )
        
        regular_mesh = dataset.create_mesh_structure(
            xy=regular_coords,
            args={'cutoff': dataset.cutoff_factor, 'num_neighbors': 4},
            graph_dir_path=dataset.save_path
        )
        
        # Test irregular grid
        theta = np.linspace(0, 2*np.pi, 20)
        r = np.linspace(0.2, 1, 10)
        theta, r = np.meshgrid(theta, r)
        irregular_coords = [
            r * np.cos(theta),
            r * np.sin(theta)
        ]
        
        irregular_mesh = dataset.create_mesh_structure(
            xy=irregular_coords,
            args={'cutoff': dataset.cutoff_factor, 'num_neighbors': 6},
            graph_dir_path=dataset.save_path
        )
        
        # Verify different properties
        assert regular_mesh['g2m_graph'].num_nodes != irregular_mesh['g2m_graph'].num_nodes
        
        # Regular grid should have more uniform edge lengths
        regular_edges = regular_mesh['edge_features']['length']
        irregular_edges = irregular_mesh['edge_features']['length']
        
        regular_std = torch.std(regular_edges)
        irregular_std = torch.std(irregular_edges)
        assert regular_std < irregular_std
    
    def test_neighbor_configurations(self, sample_data_path, base_config):
        """Test different neighbor configurations"""
        dataset = GraphDataset(
            data_path=str(sample_data_path),
            save_path=str(Path(sample_data_path).parent),
            **base_config
        )
        
        grid_coords = np.meshgrid(
            np.linspace(0, 1, 10),
            np.linspace(0, 1, 10)
        )
        
        neighbor_counts = [4, 8, 12]
        edge_counts = []
        
        for n_neighbors in neighbor_counts:
            mesh = dataset.create_mesh_structure(
                xy=grid_coords,
                args={'cutoff': dataset.cutoff_factor, 'num_neighbors': n_neighbors},
                graph_dir_path=dataset.save_path
            )
            edge_counts.append(mesh['g2m_graph'].num_edges)
        
        # Verify edge count increases with neighbors
        assert edge_counts[0] < edge_counts[1] < edge_counts[2]
    
    def test_cutoff_configurations(self, sample_data_path, base_config):
        """Test different cutoff configurations"""
        dataset = GraphDataset(
            data_path=str(sample_data_path),
            save_path=str(Path(sample_data_path).parent),
            **base_config
        )
        
        grid_coords = np.meshgrid(
            np.linspace(0, 1, 10),
            np.linspace(0, 1, 10)
        )
        
        cutoffs = [0.5, 1.0, 1.5]
        edge_lengths = []
        
        for cutoff in cutoffs:
            mesh = dataset.create_mesh_structure(
                xy=grid_coords,
                args={'cutoff': cutoff, 'num_neighbors': 4},
                graph_dir_path=dataset.save_path
            )
            edge_lengths.append(mesh['edge_features']['length'].mean())
        
        # Verify average edge length increases with cutoff
        assert edge_lengths[0] < edge_lengths[1] < edge_lengths[2]
    
    def test_adaptive_configurations(self, sample_data_path, base_config):
        """Test adaptive graph configurations"""
        dataset = GraphDataset(
            data_path=str(sample_data_path),
            save_path=str(Path(sample_data_path).parent),
            **base_config
        )
        
        # Create non-uniform grid with dense and sparse regions
        x = np.concatenate([
            np.linspace(0, 0.3, 15),    # Dense region
            np.linspace(0.4, 1.0, 5)     # Sparse region
        ])
        y = x.copy()
        grid_coords = np.meshgrid(x, y)
        
        # Test with and without adaptive neighbors
        standard_mesh = dataset.create_mesh_structure(
            xy=grid_coords,
            args={
                'cutoff': dataset.cutoff_factor,
                'num_neighbors': 4,
                'adaptive_neighbors': False
            },
            graph_dir_path=dataset.save_path
        )
        
        adaptive_mesh = dataset.create_mesh_structure(
            xy=grid_coords,
            args={
                'cutoff': dataset.cutoff_factor,
                'num_neighbors': 4,
                'adaptive_neighbors': True
            },
            graph_dir_path=dataset.save_path
        )
        
        # Analyze edge distribution
        def get_region_edges(mesh, region='dense'):
            edge_index = mesh['g2m_graph'].edge_index
            edge_lengths = mesh['edge_features']['length']
            
            # Define dense region indices (first 15*15 nodes)
            dense_nodes = set(range(15*15))
            
            dense_edges = []
            sparse_edges = []
            
            for i, (src, dst) in enumerate(edge_index.t()):
                src, dst = src.item(), dst.item()
                if src in dense_nodes and dst in dense_nodes:
                    dense_edges.append(edge_lengths[i])
                elif src not in dense_nodes and dst not in dense_nodes:
                    sparse_edges.append(edge_lengths[i])
            
            return torch.tensor(dense_edges if region == 'dense' else sparse_edges)
        
        # Compare edge length statistics
        standard_dense = get_region_edges(standard_mesh, 'dense')
        standard_sparse = get_region_edges(standard_mesh, 'sparse')
        adaptive_dense = get_region_edges(adaptive_mesh, 'dense')
        adaptive_sparse = get_region_edges(adaptive_mesh, 'sparse')
        
        # Adaptive mesh should have more uniform edge length distribution
        standard_ratio = standard_sparse.mean() / standard_dense.mean()
        adaptive_ratio = adaptive_sparse.mean() / adaptive_dense.mean()
        assert adaptive_ratio < standard_ratio
    
    def test_boundary_configurations(self, sample_data_path, base_config):
        """Test different boundary configurations"""
        dataset = GraphDataset(
            data_path=str(sample_data_path),
            save_path=str(Path(sample_data_path).parent),
            **base_config
        )
        
        # Create L-shaped domain
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        X, Y = np.meshgrid(x, y)
        
        # Remove points to create L-shape
        mask = (X < 0.6) | (Y < 0.6)
        X = X[mask]
        Y = Y[mask]
        grid_coords = [X.reshape(-1), Y.reshape(-1)]
        
        mesh = dataset.create_mesh_structure(
            xy=grid_coords,
            args={
                'cutoff': dataset.cutoff_factor,
                'num_neighbors': 4,
                'include_boundary_mask': True
            },
            graph_dir_path=dataset.save_path
        )
        
        # Verify boundary detection
        boundary_mask = mesh['boundary_mask']
        
        # Count corner nodes (should have fewer neighbors)
        edge_index = mesh['g2m_graph'].edge_index
        neighbor_counts = torch.bincount(edge_index[0])
        
        # Nodes with fewer neighbors should be marked as boundary
        low_neighbor_nodes = (neighbor_counts < neighbor_counts.mean()).nonzero()
        assert all(boundary_mask[node] == 0 for node in low_neighbor_nodes)
