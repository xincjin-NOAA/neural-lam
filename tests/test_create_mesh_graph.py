"""Tests for mesh and grid creation functionality"""

import pytest
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

from neural_lam.create_mesh_graph import (
    mk_2d_graph,
    create_grid_to_mesh,
    create_mesh_to_grid,
    plot_graph
)

class TestCreateMeshGraph:
    @pytest.fixture
    def sample_grid(self):
        """Create a sample 2D grid"""
        x = np.linspace(0, 10, 5)
        y = np.linspace(0, 10, 5)
        return np.meshgrid(x, y)
    
    @pytest.fixture
    def sample_coords(self):
        """Create sample coordinates"""
        return np.array([
            [0, 0], [0, 5], [5, 0], [5, 5],
            [2.5, 2.5], [7.5, 2.5], [2.5, 7.5], [7.5, 7.5]
        ])
    
    def test_mk_2d_graph(self, sample_grid):
        """Test 2D graph creation"""
        # Create graph
        nx, ny = 5, 5
        G = mk_2d_graph(sample_grid, nx, ny)
        
        # Test graph properties
        assert isinstance(G, nx.Graph)
        assert len(G.nodes) == nx * ny
        
        # Test node positions
        for node in G.nodes:
            assert 'pos' in G.nodes[node]
            assert len(G.nodes[node]['pos']) == 2
        
        # Test connectivity (each non-boundary node should have 4 neighbors)
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                node = i * ny + j
                assert len(list(G.neighbors(node))) == 4
    
    def test_grid_to_mesh_creation(self, sample_coords):
        """Test grid-to-mesh graph creation"""
        # Create base mesh graph
        G_mesh = nx.grid_2d_graph(3, 3)
        pos = {(i, j): np.array([i, j]) for i, j in G_mesh.nodes()}
        nx.set_node_attributes(G_mesh, pos, 'pos')
        
        # Test creation
        g2m_data = create_grid_to_mesh(
            coords=sample_coords,
            G_bottom_mesh=G_mesh,
            all_mesh_nodes=list(G_mesh.nodes()),
            args={'cutoff': 0.67, 'num_neighbors': 3, 'plot': False}
        )
        
        # Verify output
        assert 'g2m_graph' in g2m_data
        assert isinstance(g2m_data['g2m_graph'], Data)
        assert 'edge_index' in g2m_data['g2m_graph']
        
        # Test edge properties
        edge_index = g2m_data['g2m_graph'].edge_index
        assert edge_index.shape[0] == 2  # Source and target nodes
        assert edge_index.max() < len(sample_coords) + len(G_mesh.nodes())
    
    def test_mesh_to_grid_creation(self, sample_coords):
        """Test mesh-to-grid graph creation"""
        # Create mesh nodes
        vm = {i: {'pos': pos} for i, pos in enumerate(sample_coords)}
        
        # Test creation
        m2g_data = create_mesh_to_grid(
            coords=sample_coords,
            vm=vm,
            args={'cutoff': 0.67, 'num_neighbors': 3, 'plot': False},
            graph_dir_path='.'
        )
        
        # Verify output
        assert 'm2g_graph' in m2g_data
        assert 'edge_features' in m2g_data
        assert isinstance(m2g_data['m2g_graph'], Data)
        
        # Test edge features
        edge_features = m2g_data['edge_features']
        assert 'length' in edge_features
        assert 'vector_diff' in edge_features
    
    def test_heterogeneous_observations(self, sample_coords):
        """Test handling of heterogeneous observation types"""
        # Create mesh nodes
        vm = {i: {'pos': pos} for i, pos in enumerate(sample_coords)}
        
        # Test with different observation types
        obs_types = ['temperature', 'wind', 'pressure']
        for obs_type in obs_types:
            m2g_data = create_mesh_to_grid(
                coords=sample_coords,
                vm=vm,
                args={
                    'cutoff': 0.67,
                    'num_neighbors': 3,
                    'plot': False,
                    'obs_type': obs_type,
                    'adaptive_neighbors': True
                },
                graph_dir_path='.'
            )
            
            # Verify observation-specific features
            edge_features = m2g_data['edge_features']
            assert 'weights' in edge_features
            assert 'attention' in edge_features
    
    def test_adaptive_neighbor_selection(self, sample_coords):
        """Test adaptive neighbor selection"""
        # Create mesh with varying density
        sparse_coords = np.array([[0, 0], [10, 10]])  # Sparse region
        dense_coords = np.random.rand(20, 2)  # Dense region
        mixed_coords = np.vstack([sparse_coords, dense_coords])
        
        vm = {i: {'pos': pos} for i, pos in enumerate(mixed_coords)}
        
        # Test with adaptive neighbors
        m2g_data = create_mesh_to_grid(
            coords=sample_coords,
            vm=vm,
            args={
                'cutoff': 0.67,
                'num_neighbors': 3,
                'plot': False,
                'adaptive_neighbors': True
            },
            graph_dir_path='.'
        )
        
        # Verify edge distribution
        edge_index = m2g_data['m2g_graph'].edge_index
        edge_lengths = m2g_data['edge_features']['length']
        
        # Check if sparse regions have longer edges
        sparse_edges = edge_lengths > edge_lengths.mean()
        assert sparse_edges.any()  # Some edges should be longer
    
    def test_edge_feature_computation(self, sample_coords):
        """Test computation of edge features"""
        vm = {i: {'pos': pos} for i, pos in enumerate(sample_coords)}
        
        m2g_data = create_mesh_to_grid(
            coords=sample_coords,
            vm=vm,
            args={'cutoff': 0.67, 'num_neighbors': 3, 'plot': False},
            graph_dir_path='.'
        )
        
        edge_features = m2g_data['edge_features']
        
        # Test length features
        lengths = edge_features['length']
        assert torch.all(lengths >= 0)  # All lengths should be non-negative
        
        # Test vector differences
        vector_diffs = edge_features['vector_diff']
        assert vector_diffs.shape[1] == 2  # 2D vector differences
    
    def test_graph_visualization(self, sample_coords):
        """Test graph visualization functionality"""
        # Create a simple graph
        G = nx.grid_2d_graph(3, 3)
        pos = {(i, j): np.array([i, j]) for i, j in G.nodes()}
        nx.set_node_attributes(G, pos, 'pos')
        
        # Convert to PyG data
        edge_index = torch.tensor([[i, j] for i, j in G.edges()]).t()
        data = Data(edge_index=edge_index, pos=torch.tensor(list(pos.values())))
        
        # Test plotting
        fig = plot_graph(data, title="Test Graph")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_boundary_conditions(self, sample_coords):
        """Test handling of boundary conditions"""
        # Create mesh with boundary nodes
        boundary_coords = np.array([
            [0, 0], [0, 5], [0, 10],  # Left boundary
            [10, 0], [10, 5], [10, 10],  # Right boundary
            [5, 0], [5, 10]  # Top/bottom boundary
        ])
        
        vm = {i: {'pos': pos} for i, pos in enumerate(boundary_coords)}
        
        # Test creation
        m2g_data = create_mesh_to_grid(
            coords=sample_coords,
            vm=vm,
            args={'cutoff': 0.67, 'num_neighbors': 3, 'plot': False},
            graph_dir_path='.'
        )
        
        # Verify boundary node connections
        edge_index = m2g_data['m2g_graph'].edge_index
        assert edge_index.max() < len(sample_coords) + len(boundary_coords)
    
    def test_large_scale_mesh(self):
        """Test creation of large-scale mesh"""
        # Create large grid
        n = 50
        x = np.linspace(0, 100, n)
        y = np.linspace(0, 100, n)
        grid = np.meshgrid(x, y)
        
        # Test creation
        G = mk_2d_graph(grid, n, n)
        
        # Verify properties
        assert len(G.nodes) == n * n
        assert all('pos' in G.nodes[node] for node in G.nodes)
        
        # Test memory efficiency
        import psutil
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        assert memory_usage < 1000  # Should use less than 1GB
    
    @pytest.mark.parametrize("n_neighbors", [2, 4, 8])
    def test_neighbor_count_impact(self, sample_coords, n_neighbors):
        """Test impact of neighbor count on graph structure"""
        vm = {i: {'pos': pos} for i, pos in enumerate(sample_coords)}
        
        m2g_data = create_mesh_to_grid(
            coords=sample_coords,
            vm=vm,
            args={'cutoff': 0.67, 'num_neighbors': n_neighbors, 'plot': False},
            graph_dir_path='.'
        )
        
        edge_index = m2g_data['m2g_graph'].edge_index
        unique_targets = torch.unique(edge_index[1])
        
        # Each node should have approximately n_neighbors connections
        connections_per_node = edge_index.shape[1] / len(unique_targets)
        assert abs(connections_per_node - n_neighbors) <= 1
