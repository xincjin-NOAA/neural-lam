"""Tests for boundary mask functionality"""

import pytest
import numpy as np
import networkx as nx
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data

from neural_lam.create_mesh_graph import (
    create_boundary_mask,
    create_grid_to_mesh,
    plot_boundary_nodes
)

class TestBoundaryMask:
    @pytest.fixture
    def sample_mesh(self):
        """Create a sample 2D mesh"""
        # Create a 5x5 grid
        nx_size, ny_size = 5, 5
        G = nx.grid_2d_graph(nx_size, ny_size)
        
        # Convert to position-based nodes
        pos = {(i, j): np.array([i, j]) for i, j in G.nodes()}
        nx.set_node_attributes(G, pos, 'pos')
        
        # Convert to 0-based indexing
        G = nx.convert_node_labels_to_integers(G)
        
        return G, nx_size, ny_size
    
    def test_boundary_mask_creation(self, sample_mesh):
        """Test basic boundary mask creation"""
        G, nx_size, ny_size = sample_mesh
        coords = np.array([[i, j] for i in range(nx_size) for j in range(ny_size)])
        
        mask = create_boundary_mask(G, coords)
        
        # Check mask properties
        assert isinstance(mask, torch.Tensor)
        assert mask.shape[0] == len(G.nodes)
        assert mask.dtype == torch.float
        
        # Check boundary identification
        n_boundary_nodes = sum(mask == 0).item()
        expected_boundary = 2 * nx_size + 2 * (ny_size - 2)  # Perimeter nodes
        assert n_boundary_nodes == expected_boundary
    
    def test_boundary_mask_with_dict_coords(self, sample_mesh):
        """Test boundary mask creation with dictionary coordinates"""
        G, nx_size, ny_size = sample_mesh
        coords = {
            'x': np.array([i for i in range(nx_size) for _ in range(ny_size)]),
            'y': np.array([j for _ in range(nx_size) for j in range(ny_size)])
        }
        
        mask = create_boundary_mask(G, coords)
        
        # Verify mask values
        assert torch.all(mask >= 0) and torch.all(mask <= 1)
        assert torch.any(mask == 0)  # Should have boundary nodes
        assert torch.any(mask == 1)  # Should have interior nodes
    
    def test_boundary_mask_integration(self, sample_mesh):
        """Test boundary mask integration with grid-to-mesh creation"""
        G, nx_size, ny_size = sample_mesh
        coords = np.array([[i, j] for i in range(nx_size) for j in range(ny_size)])
        
        # Create grid-to-mesh with boundary mask
        g2m_data = create_grid_to_mesh(
            coords=coords,
            G_bottom_mesh=G,
            all_mesh_nodes=list(G.nodes()),
            args={
                'cutoff': 0.67,
                'num_neighbors': 3,
                'include_boundary_mask': True
            }
        )
        
        # Check mask in output
        assert 'boundary_mask' in g2m_data
        assert hasattr(g2m_data['g2m_graph'], 'boundary_mask')
        
        # Verify mask properties
        mask = g2m_data['boundary_mask']
        assert mask.shape[0] == len(G.nodes)
        assert torch.all(mask >= 0) and torch.all(mask <= 1)
    
    def test_boundary_node_connections(self, sample_mesh):
        """Test connections of boundary nodes"""
        G, nx_size, ny_size = sample_mesh
        coords = np.array([[i, j] for i in range(nx_size) for j in range(ny_size)])
        
        g2m_data = create_grid_to_mesh(
            coords=coords,
            G_bottom_mesh=G,
            all_mesh_nodes=list(G.nodes()),
            args={
                'cutoff': 0.67,
                'num_neighbors': 3,
                'include_boundary_mask': True
            }
        )
        
        # Get edge index and mask
        edge_index = g2m_data['g2m_graph'].edge_index
        mask = g2m_data['boundary_mask']
        
        # Check boundary node connections
        boundary_nodes = torch.where(mask == 0)[0]
        for node in boundary_nodes:
            # Get neighbors
            neighbors = edge_index[1][edge_index[0] == node]
            # Boundary nodes should have fewer connections on average
            assert len(neighbors) <= 4  # Max connections for grid
    
    def test_boundary_visualization(self, sample_mesh):
        """Test boundary visualization functionality"""
        G, nx_size, ny_size = sample_mesh
        coords = np.array([[i, j] for i in range(nx_size) for j in range(ny_size)])
        
        g2m_data = create_grid_to_mesh(
            coords=coords,
            G_bottom_mesh=G,
            all_mesh_nodes=list(G.nodes()),
            args={
                'cutoff': 0.67,
                'num_neighbors': 3,
                'include_boundary_mask': True
            }
        )
        
        # Test basic plotting
        fig = plot_boundary_nodes(g2m_data['g2m_graph'])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
        # Test with show_boundary option
        fig = plot_graph(g2m_data['g2m_graph'], show_boundary=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    @pytest.mark.parametrize("shape", [
        (3, 3),  # Small grid
        (10, 10),  # Medium grid
        (5, 10)   # Rectangular grid
    ])
    def test_different_mesh_shapes(self, shape):
        """Test boundary mask with different mesh shapes"""
        nx_size, ny_size = shape
        G = nx.grid_2d_graph(nx_size, ny_size)
        pos = {(i, j): np.array([i, j]) for i, j in G.nodes()}
        nx.set_node_attributes(G, pos, 'pos')
        G = nx.convert_node_labels_to_integers(G)
        
        coords = np.array([[i, j] for i in range(nx_size) for j in range(ny_size)])
        mask = create_boundary_mask(G, coords)
        
        # Check expected number of boundary nodes
        expected_boundary = 2 * nx_size + 2 * (ny_size - 2)
        assert sum(mask == 0).item() == expected_boundary
    
    def test_floating_point_precision(self, sample_mesh):
        """Test boundary mask with floating point coordinates"""
        G, nx_size, ny_size = sample_mesh
        
        # Create coordinates with floating point values
        coords = np.array([[float(i) + 0.1, float(j) + 0.1] 
                          for i in range(nx_size) for j in range(ny_size)])
        
        mask = create_boundary_mask(G, coords)
        
        # Check if boundaries are correctly identified despite floating point values
        n_boundary_nodes = sum(mask == 0).item()
        expected_boundary = 2 * nx_size + 2 * (ny_size - 2)
        assert n_boundary_nodes == expected_boundary
