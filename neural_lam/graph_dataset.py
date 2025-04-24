from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch
import zarr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from typing import Dict, List, Tuple, Optional
from functools import lru_cache

from .create_mesh_graph import mk_2d_graph, create_grid_to_mesh, create_mesh_to_grid, plot_graph
from .process_timeseries import extract_features, organize_bins_times

class GraphDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        start_date: str,
        end_date: str,
        satellite_id: str,
        observation_types: Dict[str, Dict],
        batch_size: int = 1,
        mesh_resolution: float = 2.0,
        cutoff_factor: float = 0.6,
        num_neighbors: int = 3,
        save_path: str = "./graph_data"
    ):
        super().__init__()

        # Data parameters
        self.data_path = data_path
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.satellite_id = satellite_id
        self.batch_size = batch_size
        self.observation_types = observation_types
        self.save_path = save_path

        # Graph parameters
        self.mesh_resolution = mesh_resolution
        self.cutoff_factor = cutoff_factor
        self.num_neighbors = num_neighbors

        # Initialize graph structures
        self.mesh_structure = None
        self.g2m_graph = None
        self.m2g_graph = None
        self.edge_features = {}
        
        # Call setup to initialize everything
        self.prepare_data()
        


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

        self.mesh_structure = self.create_mesh_structure(
            xy=grid_coordinates,
            args=args,
            graph_dir_path=save_path
        )

    @lru_cache(maxsize=None)
    def _create_mesh_structure_cached(self, mesh_resolution: float, cutoff_factor: float, num_neighbors: int):
        """Create the static mesh structure for the domain with caching"""
        print("Creating new mesh structures...")
        
        # Get domain bounds from data
        self.z = zarr.open(self.data_path, mode="r")
        grid_data = self.z["grid"]
        
        # Get grid coordinates
        xy = np.meshgrid(grid_data["lon"][:], grid_data["lat"][:])
        
        # Calculate number of mesh points based on resolution
        nx = int((xy[0].max() - xy[0].min()) / mesh_resolution)
        ny = int((xy[1].max() - xy[1].min()) / mesh_resolution)
        
        # Create 2D mesh graph
        G = mk_2d_graph(xy, nx, ny)
        mesh_structure = from_networkx(G)
        mesh_points = np.array([G.nodes[n]["pos"] for n in sorted(G.nodes())])
        
        # Create observation-specific grid-to-mesh and mesh-to-grid mappings
        g2m_graphs = {}
        m2g_graphs = {}
        edge_features = {}
        
        for obs_type in self.observation_types:
            # Get observation locations for this type
            obs_data = self.z[obs_type]
            grid_coords = np.stack([obs_data["lat"][:], obs_data["lon"][:]], axis=1)
            
            # Create grid-to-mesh mapping for this observation type
            g2m_data = create_grid_to_mesh(
                coords=grid_coords,
                G_bottom_mesh=G,
                all_mesh_nodes=list(G.nodes()),
                args={
                    "cutoff": cutoff_factor,
                    "num_neighbors": num_neighbors,
                    "plot": False,
                    "obs_type": obs_type  # Pass observation type for type-specific handling
                }
            )
            g2m_graphs[obs_type] = {
                "graph": g2m_data["g2m_graph"],
                "weights": g2m_data.get("edge_weights", None)
            }
            
            # Create mesh-to-grid mapping for this observation type
            m2g_data = create_mesh_to_grid(
                coords=grid_coords,
                vm=G.nodes(data=True),
                args={
                    "cutoff": cutoff_factor,
                    "num_neighbors": num_neighbors,
                    "plot": False,
                    "obs_type": obs_type,  # Pass observation type
                    "adaptive_neighbors": True  # Enable adaptive neighbor selection
                },
                graph_dir_path=self.save_path
            )
            m2g_graphs[obs_type] = {
                "graph": m2g_data["m2g_graph"],
                "edge_indices": m2g_data["edge_indices"],
                "edge_features": m2g_data["edge_features"]
            }
        
        return {
            "mesh_structure": mesh_structure,
            "mesh_points": mesh_points,
            "g2m_graphs": g2m_graphs,
            "m2g_graphs": m2g_graphs,
            "edge_features": edge_features
        }
    
    def create_mesh_structure(self):
        """Create or load the static mesh structure for the domain"""
        # Get mesh structures from cached function
        mesh_data = self._create_mesh_structure_cached(
            mesh_resolution=self.mesh_resolution,
            cutoff_factor=self.cutoff_factor,
            num_neighbors=self.num_neighbors
        )
        
        # Set instance attributes
        self.mesh_structure = mesh_data["mesh_structure"]
        self.mesh_points = mesh_data["mesh_points"]
        self.g2m_graphs = mesh_data["g2m_graphs"]

def create_dynamic_graph(self, weather_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> Data:
    """Create a PyTorch Geometric data object for the current weather sample"""
    if self.mesh_structure is None:
        raise RuntimeError("Mesh structure not created yet. Call create_mesh_structure() first.")
    
    # Create graph with mesh structure
    graph = Data()
    graph.mesh_structure = self.mesh_structure
    
    # Add observation-specific graphs and data
    for obs_type, (locations, values) in weather_data.items():
        # Add graph structures for this observation type
        graph[f"{obs_type}_g2m_graph"] = self.g2m_graphs[obs_type]
        graph[f"{obs_type}_m2g_graph"] = self.m2g_graphs[obs_type]
        graph[f"{obs_type}_edge_features"] = self.edge_features[obs_type]
        sample = self.data_summary.iloc[idx]
        weather_data = {}
        
        for obs_type, config in self.observation_types.items():
            # Get locations and values for this observation type
            locations = torch.tensor(
                self.z[f"{obs_type}/locations"][sample.name],
                dtype=torch.float32
            )
            values = torch.tensor(
                self.z[f"{obs_type}/values"][sample.name],
                dtype=torch.float32
            )
            weather_data[obs_type] = (locations, values)
        
        return weather_data

    def __getitem__(self, idx: int) -> Data:
        """Get a single sample from the dataset"""
        # 1. Load weather data for all observation types
        weather_data = self.load_weather_sample(idx)
        
        # 2. Create dynamic graph with the weather data
        graph = self.create_dynamic_graph(weather_data)
        
        # 3. Add metadata
        graph.timestamp = self.data_summary.index[idx]
        
        return graph

    def plot_mesh(self, title: str = "Mesh Structure", figsize: Tuple[int, int] = (10, 10)):
        """Visualize the mesh structure"""
        if self.mesh_structure is None:
            raise RuntimeError("Mesh structure not created yet. Call create_mesh_structure() first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot mesh nodes
        ax.scatter(
            self.mesh_points[:, 0],
            self.mesh_points[:, 1],
            c='blue',
            s=20,
            alpha=0.6,
            label='Mesh Nodes'
        )
        
        # Plot mesh edges
        edge_index = self.mesh_structure.edge_index.numpy()
        edges = self.mesh_points[edge_index.T]
        lc = LineCollection(edges, colors='gray', alpha=0.3, linewidth=0.5)
        ax.add_collection(lc)
        
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        return fig, ax
    
    def plot_grid_connections(self, sample_idx: int = 0, title: str = "Grid-Mesh Connections", figsize: Tuple[int, int] = (15, 5)):
        """Visualize grid-to-mesh and mesh-to-grid connections"""
        if self.g2m_graph is None or self.m2g_graph is None:
            raise RuntimeError("Grid-mesh connections not created yet.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Get weather data for visualization
        weather_data = self.load_weather_sample(sample_idx)
        grid_points = next(iter(weather_data.values()))[0].numpy()
        
        # Plot Grid-to-Mesh
        ax1.scatter(grid_points[:, 0], grid_points[:, 1], c='red', s=10, alpha=0.6, label='Grid Points')
        ax1.scatter(self.mesh_points[:, 0], self.mesh_points[:, 1], c='blue', s=20, alpha=0.6, label='Mesh Nodes')
        
        g2m_edges = self.g2m_graph.edge_index.numpy()
        edges = np.stack([
            grid_points[g2m_edges[0]],
            self.mesh_points[g2m_edges[1]]
        ], axis=1)
        lc = LineCollection(edges, colors='gray', alpha=0.2, linewidth=0.5)
        ax1.add_collection(lc)
        ax1.set_title('Grid-to-Mesh Connections')
        ax1.legend()
        
        # Plot Mesh-to-Grid
        ax2.scatter(grid_points[:, 0], grid_points[:, 1], c='red', s=10, alpha=0.6, label='Grid Points')
        ax2.scatter(self.mesh_points[:, 0], self.mesh_points[:, 1], c='blue', s=20, alpha=0.6, label='Mesh Nodes')
        
        m2g_edges = self.m2g_graph.edge_index.numpy()
        edges = np.stack([
            self.mesh_points[m2g_edges[0]],
            grid_points[m2g_edges[1]]
        ], axis=1)
        lc = LineCollection(edges, colors='gray', alpha=0.2, linewidth=0.5)
        ax2.add_collection(lc)
        ax2.set_title('Mesh-to-Grid Connections')
        ax2.legend()
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig, (ax1, ax2)
    
    def plot_weather_sample(self, sample_idx: int = 0, obs_type: str = None, figsize: Tuple[int, int] = (10, 10)):
        """Visualize weather data on the mesh"""
        # Get weather data
        weather_data = self.load_weather_sample(sample_idx)
        
        if obs_type is None:
            obs_type = next(iter(weather_data.keys()))
        
        if obs_type not in weather_data:
            raise ValueError(f"Observation type '{obs_type}' not found in weather data")
        
        locations, values = weather_data[obs_type]
        locations = locations.numpy()
        values = values.numpy()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot mesh structure
        ax.scatter(
            self.mesh_points[:, 0],
            self.mesh_points[:, 1],
            c='gray',
            s=10,
            alpha=0.3,
            label='Mesh Nodes'
        )
        
        # Plot weather data
        scatter = ax.scatter(
            locations[:, 0],
            locations[:, 1],
            c=values,
            cmap='viridis',
            s=30,
            alpha=0.8,
            label=f'{obs_type} Values'
        )
        plt.colorbar(scatter, ax=ax)
        
        ax.set_title(f'{obs_type} Values at {self.data_summary.index[sample_idx]}')
        ax.legend()
        plt.tight_layout()
        return fig, ax
    
    def __len__(self):
        return len(self.sample_names)
    '''
    dataset = GraphDataset(...)

# Plot mesh structure
fig, ax = dataset.plot_mesh()
plt.show()

# Plot grid-mesh connections
fig, (ax1, ax2) = dataset.plot_grid_connections(sample_idx=0)
plt.show()

# Plot weather data
fig, ax = dataset.plot_weather_sample(sample_idx=0, obs_type="temperature")
plt.show()
'''