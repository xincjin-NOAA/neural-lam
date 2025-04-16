from torch.utils.data import Dataset
from torch_geometric.data import Data
import zarr
import pandas as pd
from timing_utils import organize_bins_times
from process_timeseries import extract_features

class GraphDataset(Dataset):
    def __init__(self, ...):
        self.mesh_structure = self.create_mesh_structure(
            xy=grid_coordinates,
            args=args,
            graph_dir_path=save_path
        )
        g2m_data = self.create_grid_to_mesh(
            xy=grid_coordinates,
            G_bottom_mesh=mesh_data['G_bottom_mesh'],
            all_mesh_nodes=mesh_data['all_mesh_nodes'],
            args=args
        )
        
        self.g2m_graph = g2m_data['g2m_graph']
        self.grid_graph = g2m_data['grid_graph']

                # After creating g2m graph
        m2g_data = self.create_mesh_to_grid(
            G_g2m=g2m_data['g2m_graph'],
            vm=mesh_data['G_bottom_mesh'].nodes,
            vm_xy=mesh_positions,
            vg_list=grid_nodes,
            args=args,
            graph_dir_path=save_path
        )
        
        self.m2g_graph = m2g_data['m2g_graph']
        self.edge_features = m2g_data['edge_features']

        # Using array of projected coordinates
coords = np.array([[x1, y1], [x2, y2], ...])  # Lambert projected coordinates

# Optional projection parameters
proj_params = {
    'lat_1': 33.0,
    'lat_2': 45.0,
    'lat_0': 40.0,
    'lon_0': -97.0
}

g2m_data = create_grid_to_mesh(
    coords, 
    G_bottom_mesh, 
    all_mesh_nodes, 
    args,
    proj_params=proj_params
)


# Using array of lat/lon pairs
coords = np.array([[lat1, lon1], [lat2, lon2], ...])

# Optional projection parameters
proj_params = {
    'lat_1': 33.0,
    'lat_2': 45.0,
    'lat_0': 40.0,
    'lon_0': -97.0
}

m2g_data = create_mesh_to_grid(
    coords,
    G_g2m,
    vm,
    args,
    graph_dir_path,
    proj_params=proj_params
)



class GraphDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        start_date,
        end_date,
        satellite_id,
        batch_size=1,
        mesh_resolution=2,
        cutoff_factor=0.6,
        num_neighbors=3,
    )
        super().__init__()
        self.data_path = data_path
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.satellite_id = satellite_id
        self.batch_size = batch_size

        # Graph parameters
        self.mesh_resolution = mesh_resolution
        self.cutoff_factor = cutoff_factor
        self.num_neighbors = num_neighbors

        # # You might want to store graph generation parameters
        # self.graph_params = {
        #     'threshold': threshold,  # Example parameter for dynamic connectivity
        #     'max_neighbors': max_neighbors,
        #     'distance_metric': distance_metric
        # }

    def prepare_data(self):
        """
        Check if Zarr dataset exists.
        """
        try:
            zarr.open(self.data_path, mode="r")
        except Exception as e:
            raise RuntimeError(f"Failed to open Zarr dataset at {self.data_path}: {e}")

    def setup(self, stage=None):
        """
        Prepare data for training/validation.
        """
        # Open Zarr dataset
        self.z = zarr.open(self.data_path, mode="r")

        # Process time bins and features
        self.data_summary = organize_bins_times(
            self.z, self.start_date, self.end_date, self.satellite_id
        )
        self.data_summary = extract_features(self.z, self.data_summary)

        # Create graph structure for first bin (can be extended to handle multiple bins)
        data_dict = self._create_graph_structure(self.data_summary["bin1"])

        # Split data into train/val
        total_samples = len(data_dict["x"])
        int(0.8 * total_samples)

        # Create train/val splits
        self.train_data = self._create_data_object(data_dict, slice(0, len(data_dict["x"])))
        # self.val_data = self._create_data_object(hetero_data, slice(train_size, total_samples))
        # TODO make validation work


    def create_dynamic_graph(self, weather_data):
        """Create graph structure based on weather conditions"""
        # Example: Create edges based on weather features
        # weather_data shape: [time, features, height, width] or similar
        
        # 1. Create node features
        node_features = weather_data.reshape(-1, weather_data.shape[1])  # [nodes, features]
        
        # 2. Dynamic edge creation based on weather conditions
        edge_list = []
        edge_attrs = []
        
        # Example: Create edges based on weather similarity
        for i in range(len(node_features)):
            # Find connected nodes based on weather patterns
            # e.g., connect nodes with similar temperature/pressure
            similar_nodes = self.find_connected_nodes(
                node_features[i],
                node_features,
                threshold=self.graph_params['threshold']
            )
            
            # Add edges
            for j in similar_nodes:
                edge_list.append([i, j])
                # Edge attributes could be weather-based
                edge_attr = self.compute_edge_attributes(
                    node_features[i],
                    node_features[j]
                )
                edge_attrs.append(edge_attr)
        
        # 3. Create PyG Data object
        return Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(edge_list, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_attrs, dtype=torch.float)
        )
    
def create_dynamic_graph(self, weather_data):
    """
    Create graph structure based on weather conditions following create_mesh.py approach.
    Creates both grid-to-mesh (g2m) and mesh-to-grid (m2g) graphs.
    
    Args:
        weather_data: Dictionary containing:
            - data: Weather features [time, features, height, width]
            - coordinates: Grid coordinates
    """
    import networkx as nx
    import numpy as np
    from scipy.spatial import KDTree
    
    # Constants from create_mesh.py
    DM_SCALE = 0.67  # radius scale for grid-mesh association
    
    # 1. Extract grid coordinates
    grid_coords = weather_data['coordinates']  # [2, height, width]
    Ny, Nx = grid_coords.shape[1:]
    
    # 2. Create mesh grid with specified resolution
    xm, xM = np.amin(grid_coords[0][0, :]), np.amax(grid_coords[0][0, :])
    ym, yM = np.amin(grid_coords[1][:, 0]), np.amax(grid_coords[1][:, 0])
    
    # Create mesh points avoiding borders
    dx = (xM - xm) / Nx * self.mesh_resolution
    dy = (yM - ym) / Ny * self.mesh_resolution
    lx = np.linspace(xm + dx/2, xM - dx/2, Nx)
    ly = np.linspace(ym + dy/2, yM - dy/2, Ny)
    mesh_grid = np.meshgrid(lx, ly)
    
    # 3. Create base grid graph
    G_grid = nx.grid_2d_graph(Ny, Nx)
    G_grid.clear_edges()  # We'll add edges based on weather data
    
    # Add grid node positions
    for node in G_grid.nodes:
        G_grid.nodes[node]['pos'] = np.array([
            grid_coords[0][node],
            grid_coords[1][node]
        ])
    
    # 4. Create mesh graph
    G_mesh = nx.Graph()
    mesh_points = np.vstack([mesh_grid[0].ravel(), mesh_grid[1].ravel()]).T
    for i, pos in enumerate(mesh_points):
        G_mesh.add_node(i, pos=pos)
    
    # 5. Grid to Mesh (g2m) conversion
    G_g2m = nx.DiGraph()
    G_g2m.add_nodes_from(G_grid.nodes(data=True))
    G_g2m.add_nodes_from(G_mesh.nodes(data=True))
    
    # Build KD-tree for mesh points
    mesh_tree = KDTree(mesh_points)
    
    # Add g2m edges based on proximity and weather similarity
    for v in G_grid.nodes():
        grid_pos = G_grid.nodes[v]['pos']
        grid_weather = weather_data['data'][:, v[0], v[1]]  # Get weather features at this point
        
        # Find nearest mesh points
        distances, indices = mesh_tree.query(grid_pos, k=self.num_neighbors)
        
        for idx, dist in zip(indices, distances):
            if dist < self.cutoff_factor * dx:  # Use cutoff_factor for radius control
                # Add edge from grid to mesh
                G_g2m.add_edge(v, idx)
                # Add edge attributes
                G_g2m.edges[v, idx]['len'] = dist
                G_g2m.edges[v, idx]['vdiff'] = G_mesh.nodes[idx]['pos'] - grid_pos
    
    # 6. Mesh to Grid (m2g) conversion
    G_m2g = G_g2m.copy()
    G_m2g.clear_edges()
    
    # Build KD-tree for grid points
    grid_points = np.array([G_grid.nodes[n]['pos'] for n in G_grid.nodes()])
    grid_tree = KDTree(grid_points)
    
    # Add m2g edges
    for v in G_mesh.nodes():
        mesh_pos = G_mesh.nodes[v]['pos']
        
        # Find nearest grid points
        distances, indices = grid_tree.query(mesh_pos, k=self.num_neighbors)
        
        for idx, dist in zip(indices, distances):
            grid_node = list(G_grid.nodes())[idx]
            if dist < self.cutoff_factor * dx:
                # Add edge from mesh to grid
                G_m2g.add_edge(v, grid_node)
                G_m2g.edges[v, grid_node]['len'] = dist
                G_m2g.edges[v, grid_node]['vdiff'] = (
                    G_grid.nodes[grid_node]['pos'] - mesh_pos
                )
    
    # 7. Convert to PyG Data objects
    pyg_g2m = from_networkx(G_g2m)
    pyg_m2g = from_networkx(G_m2g)
    
    # 8. Add weather features to nodes
    weather_features = weather_data['data'].reshape(-1, weather_data['data'].shape[0])
    pyg_g2m.x = torch.tensor(weather_features, dtype=torch.float)
    pyg_m2g.x = torch.tensor(weather_features, dtype=torch.float)
    
    return {
        'g2m': pyg_g2m,
        'm2g': pyg_m2g,
        'grid_graph': G_grid,
        'mesh_graph': G_mesh
    }

    def find_connected_nodes(self, node_feature, all_features, threshold):
        """Find nodes that should be connected based on weather similarity"""
        # Example implementation:
        distances = torch.norm(all_features - node_feature, dim=1)
        connected = torch.where(distances < threshold)[0]
        return connected
    
    def compute_edge_attributes(self, feature1, feature2):
        """Compute edge attributes based on connected nodes' features"""
        # Example: Edge weight could be feature similarity
        return torch.exp(-torch.norm(feature1 - feature2))
    
    def __getitem__(self, idx):
        # 1. Load weather data
        weather_data = self.load_weather_sample(idx)
        
        # 2. Create dynamic graph based on this specific weather sample
        graph = self.create_dynamic_graph(weather_data)
        
        # 3. You might want to add additional features
        graph.original_shape = weather_data.shape  # Keep original spatial info
        graph.timestamp = self.sample_names[idx]   # Keep temporal info
        
        return graph

    def __len__(self):
        return len(self.sample_names)