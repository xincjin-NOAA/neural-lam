"""Tests for HeteroObservationModel"""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data
from neural_lam.models.hetero_observation_model import HeteroObservationGraphModel
from neural_lam.config import Config
from unittest.mock import patch

class TestHeteroObservationModel:
    @pytest.fixture
    def setup_lambert_projection(self):
        config = Config({
            'projection': {
                'class': 'LambertConformal',
                'kwargs': {
                    'central_longitude': -97.0,
                    'central_latitude': 38.0,
                    'standard_parallels': (38.0, 38.0)
                }
            }
        })
        return config

    @pytest.fixture
    def model_args(self):
        """Basic model arguments"""
        class Args:
            def __init__(self):
                self.hidden_dim = 32
                self.hidden_layers = 2
                self.data_config = Config({
                    'input_dim': 2,  # 2D coordinates
                    'output_dim': 1,  # Single output value
                    'static_dim': 0,   # No static features
                    'dataset': {
                        'name': 'test_dataset',
                        'num_forcing_features': 0,
                        'var_names': ['temperature']
                    }
                })
                self.graph = {
                    'mesh_structure': {
                        'nodes': 10,  # 10 mesh nodes
                        'edges': 15,   # 15 mesh edges
                        'features': 2   # 2D positions
                    },
                    'config': Config({
                        'projection': {
                            'class': 'LambertConformal',
                            'kwargs': {
                                'central_longitude': -97.0,
                                'central_latitude': 38.0,
                                'standard_parallels': (38.0, 38.0)
                            }
                        }
                    })
                }
                self.observation_types = {
                    'temperature': {'dim': 1},
                    'wind': {'dim': 2},
                    'pressure': {'dim': 1}
                }
                # ARModel parameters
                self.output_std = False
                self.step_length = 1
                self.restore_opt = False
                self.n_example_pred = 1
                self.processor_layers = 3
                self.mesh_aggr = 'mean'
                self.loss = 'mse'
                self.lr = 0.001
                self.val_steps_to_log = [1, 2, 3]
                self.param_weights = torch.ones(1)
        return Args()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample graph data"""
        # Create mesh structure
        mesh_pos = torch.rand((10, 2))  # 10 mesh nodes with 2D positions
        mesh_edges = torch.randint(0, 10, (2, 15))  # 15 random mesh edges
        
        # Create observation data
        data = Data()
        data.mesh_structure = Data(pos=mesh_pos, edge_index=mesh_edges)
        
        # Add observation-specific data
        batch_size = 2
        for obs_type in ['temperature', 'wind', 'pressure']:
            # Random number of observations for each type
            n_obs = np.random.randint(5, 15)
            
            # Create observation locations and values
            locations = torch.rand((batch_size, n_obs, 2))
            if obs_type == 'wind':
                values = torch.rand((batch_size, n_obs, 2))  # 2D wind vectors
            else:
                values = torch.rand((batch_size, n_obs, 1))  # Scalar values
            
            # Create graph mappings
            g2m_edges = torch.randint(0, n_obs, (2, n_obs * 3))  # 3 connections per obs
            m2g_edges = torch.randint(0, 10, (2, n_obs * 3))
            
            # Add to data object
            data[f"{obs_type}_locations"] = locations
            data[f"{obs_type}_values"] = values
            data[f"{obs_type}_g2m_graph"] = Data(edge_index=g2m_edges)
            data[f"{obs_type}_m2g_graph"] = Data(edge_index=m2g_edges)
            data[f"{obs_type}_edge_features"] = {
                'weights': torch.rand(g2m_edges.size(1)),
                'attention': torch.rand(g2m_edges.size(1))
            }
        
        return data

    def test_model_initialization(self, model_args):
        """Test model initialization"""
        with patch('neural_lam.models.base_graph_model.utils.load_graph') as mock_load_graph, \
             patch('neural_lam.models.ar_model.utils.load_static_data') as mock_load_static:
            mock_load_graph.return_value = (False, {
                'mesh_static_features': torch.randn(10, 2),  # 10 nodes, 2 features each
                'm2m_features': torch.randn(15, 2),  # 15 edges, 2 features each
                'm2m_edge_index': torch.randint(0, 10, (2, 15)),  # 15 edges between mesh nodes
                'g2m_features': torch.randn(20, 2),  # 20 grid-to-mesh edges
                'g2m_edge_index': torch.randint(0, 10, (2, 20)),
                'm2g_features': torch.randn(20, 2),  # 20 mesh-to-grid edges
                'm2g_edge_index': torch.randint(0, 10, (2, 20))
            })
            
            mock_load_static.return_value = {
                'grid_static_features': torch.randn(10, 2),  # Static features for each grid point
                'border_mask': torch.zeros(10, 1),  # Border mask
                'step_diff_std': torch.ones(1),  # Standard deviation for step differences
            }
            
            model = HeteroObservationGraphModel(model_args)
        
        # Check observation networks are created
        for obs_type in model_args.observation_types:
            assert obs_type in model.observation_embedders
            assert obs_type in model.observation_to_mesh
            assert obs_type in model.mesh_to_observation
    
    def test_forward_pass(self, model_args, sample_data):
        """Test forward pass with sample data"""
        model = HeteroObservationGraphModel(model_args)
        output = model(sample_data)
        
        # Check output structure
        assert isinstance(output, dict)
        for obs_type in model_args.observation_types:
            assert obs_type in output
            
            # Check output dimensions
            batch_size = sample_data[f"{obs_type}_values"].size(0)
            n_obs = sample_data[f"{obs_type}_values"].size(1)
            expected_dim = 2 if obs_type == 'wind' else 1
            assert output[obs_type].size() == (batch_size, n_obs, expected_dim)
    
    def test_observation_embedding(self, model_args, sample_data):
        """Test observation embedding layer"""
        model = HeteroObservationGraphModel(model_args)
        
        for obs_type in model_args.observation_types:
            values = sample_data[f"{obs_type}_values"]
            embedded = model.observation_embedders[obs_type](values)
            
            # Check embedding dimension
            assert embedded.size(-1) == model_args.hidden_dim
    
    def test_mesh_processing(self, model_args, sample_data):
        """Test processing on mesh"""
        model = HeteroObservationGraphModel(model_args)
        
        # Get mesh features (this would normally happen in forward pass)
        mesh_features = torch.rand(10, model_args.hidden_dim)  # 10 mesh nodes
        processed = model.mesh_gnn(mesh_features, sample_data.mesh_structure.edge_index)
        
        # Check output dimensions
        assert processed.size() == mesh_features.size()
    
    def test_observation_mapping(self, model_args, sample_data):
        """Test observation-to-mesh and mesh-to-observation mapping"""
        model = HeteroObservationGraphModel(model_args)
        
        for obs_type in model_args.observation_types:
            # Get sample features
            obs_features = torch.rand_like(sample_data[f"{obs_type}_values"])
            mesh_features = torch.rand(10, model_args.hidden_dim)
            
            # Test observation to mesh mapping
            to_mesh = model.observation_to_mesh[obs_type](
                obs_features,
                sample_data[f"{obs_type}_g2m_graph"].edge_index,
                sample_data[f"{obs_type}_edge_features"]
            )
            assert to_mesh.size(0) == 10  # Number of mesh nodes
            
            # Test mesh to observation mapping
            to_obs = model.mesh_to_observation[obs_type](
                mesh_features,
                sample_data[f"{obs_type}_m2g_graph"].edge_index,
                sample_data[f"{obs_type}_edge_features"]
            )
            assert to_obs.size(1) == sample_data[f"{obs_type}_values"].size(1)
    
    def test_batch_processing(self, model_args):
        """Test processing of batched data"""
        model = HeteroObservationGraphModel(model_args)
        batch_size = 3
        
        # Create a batch of data
        batch_list = []
        for _ in range(batch_size):
            batch_list.append(self.sample_data())
        
        batch = Batch.from_data_list(batch_list)
        output = model(batch)
        
        # Check batch processing
        assert all(output[obs_type].size(0) == batch_size 
                  for obs_type in model_args.observation_types)
    
    @pytest.mark.parametrize("obs_type,obs_dim", [
        ("temperature", 1),
        ("wind", 2),
        ("pressure", 1)
    ])
    def test_observation_dimensions(self, model_args, obs_type, obs_dim):
        """Test handling of different observation dimensions"""
        # Modify model args for single observation type
        model_args.observation_types = {obs_type: {'dim': obs_dim}}
        model = HeteroObservationGraphModel(model_args)
        
        # Create sample data with specific dimension
        n_obs = 5
        values = torch.rand(2, n_obs, obs_dim)  # batch_size=2
        locations = torch.rand(2, n_obs, 2)
        
        data = Data()
        data.mesh_structure = Data(
            pos=torch.rand(10, 2),
            edge_index=torch.randint(0, 10, (2, 15))
        )
        data[f"{obs_type}_values"] = values
        data[f"{obs_type}_locations"] = locations
        data[f"{obs_type}_g2m_graph"] = Data(edge_index=torch.randint(0, n_obs, (2, n_obs * 3)))
        data[f"{obs_type}_m2g_graph"] = Data(edge_index=torch.randint(0, 10, (2, n_obs * 3)))
        data[f"{obs_type}_edge_features"] = {
            'weights': torch.rand(n_obs * 3),
            'attention': torch.rand(n_obs * 3)
        }
        
        output = model(data)
        assert output[obs_type].size() == (2, n_obs, obs_dim)
