# tests/test_mesh_graphs.py
# pytest tests/test_mesh_graphs.py -v -k "mesh_structure"

import pytest
import numpy as np
import networkx as nx
import torch
from neural_lam.create_mesh_vs1 import CreateMesh  # Adjust import based on your class name

@pytest.fixture
def sample_coords():
    """Create sample grid coordinates."""
    # Create a 3x3 grid of points
    x = np.array([0, 1, 2])
    y = np.array([0, 1, 2])
    xx, yy = np.meshgrid(x, y)
    return {
        'x': xx,
        'y': yy
    }

@pytest.fixture
def sample_mesh_nodes():
    """Create sample mesh nodes."""
    # Create a simple mesh with 5 nodes
    mesh = nx.Graph()
    positions = [
        [0.5, 0.5],  # Center
        [0.0, 0.0],  # Bottom-left
        [0.0, 2.0],  # Top-left
        [2.0, 0.0],  # Bottom-right
        [2.0, 2.0]   # Top-right
    ]
    for i, pos in enumerate(positions):
        mesh.add_node(i, pos=np.array(pos))
    return mesh.nodes

@pytest.fixture
def mock_args():
    """Create mock arguments."""
    class Args:
        plot = False
    return Args()

def test_create_mesh_to_grid_basic(sample_coords, sample_mesh_nodes, mock_args, tmp_path):
    """Test basic functionality of create_mesh_to_grid."""
    # Setup
    mesh_creator = CreateMesh()  # Adjust based on your class initialization
    graph_dir = str(tmp_path / "graphs")
    
    # Execute
    result = mesh_creator.create_mesh_to_grid(
        coords=sample_coords,
        vm=sample_mesh_nodes,
        args=mock_args,
        graph_dir_path=graph_dir
    )
    
    # Assert
    assert 'm2g_graph' in result
    assert 'edge_indices' in result
    assert 'edge_features' in result
    
    # Check graph structure
    m2g_graph = result['m2g_graph']
    assert isinstance(m2g_graph.edge_index, torch.Tensor)
    assert m2g_graph.num_nodes > 0
    assert m2g_graph.num_edges > 0

def test_edge_features(sample_coords, sample_mesh_nodes, mock_args, tmp_path):
    """Test edge features of the created graph."""
    mesh_creator = CreateMesh()
    graph_dir = str(tmp_path / "graphs")
    
    result = mesh_creator.create_mesh_to_grid(
        coords=sample_coords,
        vm=sample_mesh_nodes,
        args=mock_args,
        graph_dir_path=graph_dir
    )
    
    # Check edge features
    edge_features = result['edge_features']
    assert 'length' in edge_features
    assert 'vector_diff' in edge_features
    
    # Verify feature dimensions
    assert len(edge_features['length'].shape) == 1
    assert len(edge_features['vector_diff'].shape) == 2
    assert edge_features['vector_diff'].shape[1] == 2  # 2D coordinates

def test_coordinate_formats(sample_mesh_nodes, mock_args, tmp_path):
    """Test different coordinate input formats."""
    mesh_creator = CreateMesh()
    graph_dir = str(tmp_path / "graphs")
    
    # Test array input
    coords_array = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ])
    
    result_array = mesh_creator.create_mesh_to_grid(
        coords=coords_array,
        vm=sample_mesh_nodes,
        args=mock_args,
        graph_dir_path=graph_dir
    )
    assert result_array['m2g_graph'].num_nodes == len(coords_array) + len(sample_mesh_nodes)

    # Test dictionary input
    coords_dict = {
        'x': np.array([[0, 1], [0, 1]]),
        'y': np.array([[0, 0], [1, 1]])
    }
    
    result_dict = mesh_creator.create_mesh_to_grid(
        coords=coords_dict,
        vm=sample_mesh_nodes,
        args=mock_args,
        graph_dir_path=graph_dir
    )
    assert result_dict['m2g_graph'].num_nodes == 4 + len(sample_mesh_nodes)

def test_edge_properties(sample_coords, sample_mesh_nodes, mock_args, tmp_path):
    """Test properties of created edges."""
    mesh_creator = CreateMesh()
    graph_dir = str(tmp_path / "graphs")
    
    result = mesh_creator.create_mesh_to_grid(
        coords=sample_coords,
        vm=sample_mesh_nodes,
        args=mock_args,
        graph_dir_path=graph_dir
    )
    
    edge_features = result['edge_features']
    
    # Check that lengths are positive
    assert torch.all(edge_features['length'] > 0)
    
    # Check that vector differences are reasonable
    vector_diffs = edge_features['vector_diff']
    assert torch.all(torch.abs(vector_diffs) <= 2.0)  # Based on our coordinate range

@pytest.mark.parametrize("num_points", [4, 9, 16])
def test_different_grid_sizes(num_points, sample_mesh_nodes, mock_args, tmp_path):
    """Test with different grid sizes."""
    mesh_creator = CreateMesh()
    graph_dir = str(tmp_path / "graphs")
    
    # Create grid of different sizes
    size = int(np.sqrt(num_points))
    x = np.linspace(0, 2, size)
    y = np.linspace(0, 2, size)
    xx, yy = np.meshgrid(x, y)
    coords = {'x': xx, 'y': yy}
    
    result = mesh_creator.create_mesh_to_grid(
        coords=coords,
        vm=sample_mesh_nodes,
        args=mock_args,
        graph_dir_path=graph_dir
    )
    
    assert result['m2g_graph'].num_nodes == num_points + len(sample_mesh_nodes)



    # Add to tests/test_mesh_graphs.py
    
    @pytest.fixture
    def sample_bottom_mesh():
        """Create a sample bottom mesh graph."""
        G = nx.Graph()
        # Create a 2x2 mesh with 4 nodes
        positions = [
            [0.5, 0.5],  # Center-left
            [1.5, 0.5],  # Center-right
            [0.5, 1.5],  # Top-left
            [1.5, 1.5]   # Top-right
        ]
        for i, pos in enumerate(positions):
            G.add_node(i, pos=np.array(pos))
        
        # Add edges between nodes
        G.add_edge(0, 1)  # Horizontal bottom
        G.add_edge(2, 3)  # Horizontal top
        G.add_edge(0, 2)  # Vertical left
        G.add_edge(1, 3)  # Vertical right
        
        return G
    
    @pytest.fixture
    def sample_all_mesh_nodes(sample_bottom_mesh):
        """Create sample all mesh nodes with data."""
        return list(sample_bottom_mesh.nodes(data=True))
    
    def test_grid_to_mesh_basic(sample_coords, sample_bottom_mesh, sample_all_mesh_nodes, mock_args):
        """Test basic functionality of create_grid_to_mesh."""
        mesh_creator = CreateMesh()
        
        result = mesh_creator.create_grid_to_mesh(
            coords=sample_coords,
            G_bottom_mesh=sample_bottom_mesh,
            all_mesh_nodes=sample_all_mesh_nodes,
            args=mock_args
        )
        
        # Check required keys
        assert 'g2m_graph' in result
        assert 'grid_graph' in result
        assert 'mesh_distance' in result
        
        # Check graph properties
        g2m_graph = result['g2m_graph']
        assert isinstance(g2m_graph, torch_geometric.data.Data)
        assert g2m_graph.num_nodes > 0
        assert g2m_graph.num_edges > 0
    
    def test_grid_to_mesh_edge_properties(sample_coords, sample_bottom_mesh, sample_all_mesh_nodes, mock_args):
        """Test edge properties in grid-to-mesh graph."""
        mesh_creator = CreateMesh()
        
        result = mesh_creator.create_grid_to_mesh(
            coords=sample_coords,
            G_bottom_mesh=sample_bottom_mesh,
            all_mesh_nodes=sample_all_mesh_nodes,
            args=mock_args
        )
        
        grid_graph = result['grid_graph']
        
        # Check edge attributes
        for _, _, data in grid_graph.edges(data=True):
            assert 'len' in data
            assert 'vdiff' in data
            assert isinstance(data['len'], (float, np.float64))
            assert isinstance(data['vdiff'], np.ndarray)
            assert data['vdiff'].shape == (2,)  # 2D vector difference
    
    def test_grid_to_mesh_node_indices(sample_coords, sample_bottom_mesh, sample_all_mesh_nodes, mock_args):
        """Test node indexing in grid-to-mesh graph."""
        mesh_creator = CreateMesh()
        
        result = mesh_creator.create_grid_to_mesh(
            coords=sample_coords,
            G_bottom_mesh=sample_bottom_mesh,
            all_mesh_nodes=sample_all_mesh_nodes,
            args=mock_args
        )
        
        grid_graph = result['grid_graph']
        
        # Check grid node indices (should be prepended with 1000)
        grid_nodes = [n for n in grid_graph.nodes() if isinstance(n, str) and n.startswith('1000')]
        assert len(grid_nodes) == len(sample_coords['x'].flatten())
    
    @pytest.mark.parametrize("grid_size", [2, 3, 4])
    def test_grid_to_mesh_different_sizes(grid_size, sample_bottom_mesh, sample_all_mesh_nodes, mock_args):
        """Test grid-to-mesh creation with different grid sizes."""
        mesh_creator = CreateMesh()
        
        # Create grids of different sizes
        x = np.linspace(0, 2, grid_size)
        y = np.linspace(0, 2, grid_size)
        xx, yy = np.meshgrid(x, y)
        coords = {'x': xx, 'y': yy}
        
        result = mesh_creator.create_grid_to_mesh(
            coords=coords,
            G_bottom_mesh=sample_bottom_mesh,
            all_mesh_nodes=sample_all_mesh_nodes,
            args=mock_args
        )
        
        expected_grid_nodes = grid_size * grid_size
        actual_grid_nodes = len([n for n in result['grid_graph'].nodes() 
                               if isinstance(n, str) and n.startswith('1000')])
        assert actual_grid_nodes == expected_grid_nodes
    
    def test_grid_to_mesh_distance_calculation(sample_coords, sample_bottom_mesh, sample_all_mesh_nodes, mock_args):
        """Test mesh distance calculation in grid-to-mesh."""
        mesh_creator = CreateMesh()
        
        result = mesh_creator.create_grid_to_mesh(
            coords=sample_coords,
            G_bottom_mesh=sample_bottom_mesh,
            all_mesh_nodes=sample_all_mesh_nodes,
            args=mock_args
        )
        
        # Check mesh distance
        assert 'mesh_distance' in result
        assert result['mesh_distance'] > 0
        
        # Verify distance is reasonable (should be around 1.0 given our mesh setup)
        assert 0.5 < result['mesh_distance'] < 1.5
    
    def test_grid_to_mesh_coordinate_formats(sample_bottom_mesh, sample_all_mesh_nodes, mock_args):
        """Test different coordinate input formats for grid-to-mesh."""
        mesh_creator = CreateMesh()
        
        # Test array input
        coords_array = np.array([
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1]
        ])
        
        result_array = mesh_creator.create_grid_to_mesh(
            coords=coords_array,
            G_bottom_mesh=sample_bottom_mesh,
            all_mesh_nodes=sample_all_mesh_nodes,
            args=mock_args
        )
        
        assert result_array['g2m_graph'].num_nodes == len(coords_array) + len(sample_all_mesh_nodes)
        
        # Test dictionary input
        coords_dict = {
            'x': np.array([[0, 1], [0, 1]]),
            'y': np.array([[0, 0], [1, 1]])
        }
        
        result_dict = mesh_creator.create_grid_to_mesh(
            coords=coords_dict,
            G_bottom_mesh=sample_bottom_mesh,
            all_mesh_nodes=sample_all_mesh_nodes,
            args=mock_args
        )
        
        assert result_dict['g2m_graph'].num_nodes == 4 + len(sample_all_mesh_nodes)

# Add to tests/test_mesh_graphs.py

@pytest.fixture
def sample_xy():
    """Create sample xy coordinates for mesh structure."""
    # Create a 4x4 grid
    x = np.linspace(0, 3, 4)
    y = np.linspace(0, 3, 4)
    xx, yy = np.meshgrid(x, y)
    return {
        'x': xx,
        'y': yy
    }

def test_create_mesh_structure_basic(sample_xy, mock_args, tmp_path):
    """Test basic functionality of create_mesh_structure."""
    mesh_creator = CreateMesh()
    graph_dir = str(tmp_path / "graphs")
    
    result = mesh_creator.create_mesh_structure(
        xy=sample_xy,
        args=mock_args,
        graph_dir_path=graph_dir
    )
    
    # Check required keys in result
    assert 'm2m_graphs' in result
    assert 'mesh_pos' in result
    assert 'mesh_nodes' in result
    assert 'G_bottom_mesh' in result
    
    # Check mesh-to-mesh graphs
    assert isinstance(result['m2m_graphs'], list)
    assert len(result['m2m_graphs']) > 0
    for graph in result['m2m_graphs']:
        assert isinstance(graph, torch_geometric.data.Data)

def test_mesh_structure_positions(sample_xy, mock_args, tmp_path):
    """Test mesh positions in created structure."""
    mesh_creator = CreateMesh()
    graph_dir = str(tmp_path / "graphs")
    
    result = mesh_creator.create_mesh_structure(
        xy=sample_xy,
        args=mock_args,
        graph_dir_path=graph_dir
    )
    
    mesh_pos = result['mesh_pos']
    
    # Check position structure
    assert isinstance(mesh_pos, dict)
    assert len(mesh_pos) > 0
    
    # Check position values
    for pos in mesh_pos.values():
        assert isinstance(pos, np.ndarray)
        assert pos.shape == (2,)  # 2D coordinates
        # Check positions are within the input grid bounds
        assert 0 <= pos[0] <= 3
        assert 0 <= pos[1] <= 3

def test_mesh_structure_bottom_mesh(sample_xy, mock_args, tmp_path):
    """Test bottom mesh properties in created structure."""
    mesh_creator = CreateMesh()
    graph_dir = str(tmp_path / "graphs")
    
    result = mesh_creator.create_mesh_structure(
        xy=sample_xy,
        args=mock_args,
        graph_dir_path=graph_dir
    )
    
    G_bottom_mesh = result['G_bottom_mesh']
    
    # Check bottom mesh structure
    assert isinstance(G_bottom_mesh, nx.Graph)
    assert len(G_bottom_mesh.nodes) > 0
    assert len(G_bottom_mesh.edges) > 0
    
    # Check node attributes
    for node in G_bottom_mesh.nodes:
        assert 'pos' in G_bottom_mesh.nodes[node]
        pos = G_bottom_mesh.nodes[node]['pos']
        assert isinstance(pos, np.ndarray)
        assert pos.shape == (2,)

@pytest.mark.parametrize("grid_size", [(2, 2), (3, 3), (4, 4)])
def test_mesh_structure_different_sizes(grid_size, mock_args, tmp_path):
    """Test mesh structure creation with different grid sizes."""
    mesh_creator = CreateMesh()
    graph_dir = str(tmp_path / "graphs")
    
    # Create grids of different sizes
    x = np.linspace(0, grid_size[0]-1, grid_size[0])
    y = np.linspace(0, grid_size[1]-1, grid_size[1])
    xx, yy = np.meshgrid(x, y)
    xy = {'x': xx, 'y': yy}
    
    result = mesh_creator.create_mesh_structure(
        xy=xy,
        args=mock_args,
        graph_dir_path=graph_dir
    )
    
    # Check mesh structure scales appropriately
    assert len(result['mesh_pos']) > 0
    assert len(result['m2m_graphs']) > 0

def test_mesh_structure_connectivity(sample_xy, mock_args, tmp_path):
    """Test connectivity of created mesh structure."""
    mesh_creator = CreateMesh()
    graph_dir = str(tmp_path / "graphs")
    
    result = mesh_creator.create_mesh_structure(
        xy=sample_xy,
        args=mock_args,
        graph_dir_path=graph_dir
    )
    
    G_bottom_mesh = result['G_bottom_mesh']
    
    # Check graph connectivity
    assert nx.is_connected(G_bottom_mesh)
    
    # Check edge properties
    for u, v in G_bottom_mesh.edges:
        pos_u = G_bottom_mesh.nodes[u]['pos']
        pos_v = G_bottom_mesh.nodes[v]['pos']
        # Check that connected nodes are reasonably close
        distance = np.sqrt(np.sum((pos_u - pos_v) ** 2))
        assert distance < 2.0  # Assuming unit grid spacing

def test_mesh_structure_hierarchical(sample_xy, mock_args, tmp_path):
    """Test hierarchical structure of mesh-to-mesh graphs."""
    mesh_creator = CreateMesh()
    graph_dir = str(tmp_path / "graphs")
    
    result = mesh_creator.create_mesh_structure(
        xy=sample_xy,
        args=mock_args,
        graph_dir_path=graph_dir
    )
    
    m2m_graphs = result['m2m_graphs']
    
    # Check hierarchical structure
    for i in range(len(m2m_graphs)-1):
        curr_graph = m2m_graphs[i]
        next_graph = m2m_graphs[i+1]
        # Each level should have fewer or equal nodes than the previous
        assert curr_graph.num_nodes >= next_graph.num_nodes

def test_mesh_structure_invalid_input(mock_args, tmp_path):
    """Test mesh structure creation with invalid input."""
    mesh_creator = CreateMesh()
    graph_dir = str(tmp_path / "graphs")
    
    # Test with empty coordinates
    empty_xy = {'x': np.array([]), 'y': np.array([])}
    with pytest.raises(Exception):
        mesh_creator.create_mesh_structure(
            xy=empty_xy,
            args=mock_args,
            graph_dir_path=graph_dir
        )
    
    # Test with mismatched x, y dimensions
    invalid_xy = {
        'x': np.array([[0, 1], [0, 1]]),
        'y': np.array([0, 1])
    }
    with pytest.raises(Exception):
        mesh_creator.create_mesh_structure(
            xy=invalid_xy,
            args=mock_args,
            graph_dir_path=graph_dir
        )

# Add to tests/test_mesh_graphs.py

def test_lambert_projection():
    """Test Lambert projection setup and coordinate projection."""
    mesh_creator = CreateMesh()
    
    # Test default projection setup
    proj = mesh_creator.setup_lambert_projection()
    assert proj.crs.name == 'Lambert Conformal Conic'
    
    # Test custom projection parameters
    custom_params = {
        'lat_1': 25.0,
        'lat_2': 45.0,
        'lat_0': 35.0,
        'lon_0': -100.0,
        'earth_radius': 6378137
    }
    custom_proj = mesh_creator.setup_lambert_projection(custom_params)
    assert custom_proj.crs.name == 'Lambert Conformal Conic'
    
    # Test coordinate projection
    test_coords = {
        'lat': np.array([[30.0, 35.0], [40.0, 45.0]]),
        'lon': np.array([[-95.0, -90.0], [-85.0, -80.0]])
    }
    projected = mesh_creator.project_coordinates(test_coords, proj)
    
    # Check output shape and type
    assert isinstance(projected, np.ndarray)
    assert projected.shape == (4, 2)  # 4 points, 2 coordinates each
    
    # Test array input
    array_coords = np.array([
        [30.0, -95.0],
        [35.0, -90.0],
        [40.0, -85.0],
        [45.0, -80.0]
    ])
    projected_array = mesh_creator.project_coordinates(array_coords, proj)
    assert projected_array.shape == (4, 2)

@pytest.mark.parametrize("input_type", ["dict", "array"])
def test_projection_input_formats(input_type):
    """Test projection with different input formats."""
    mesh_creator = CreateMesh()
    proj = mesh_creator.setup_lambert_projection()
    
    if input_type == "dict":
        coords = {
            'lat': np.array([[35.0, 40.0]]),
            'lon': np.array([[-90.0, -85.0]])
        }
    else:  # array
        coords = np.array([
            [35.0, -90.0],
            [40.0, -85.0]
        ])
    
    projected = mesh_creator.project_coordinates(coords, proj)
    assert isinstance(projected, np.ndarray)
    assert projected.shape[1] == 2  # Always 2D coordinates


    