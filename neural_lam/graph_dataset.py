from torch.utils.data import Dataset
from torch_geometric.data import Data
import os
import torch
import zarr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from typing import Dict, List, Tuple, Optional
from functools import lru_cache
from zarr.storage import LRUStoreCache

from .process_timeseries import extract_features, organize_bins_times
from .create_mesh_graph import create_obs_conn_mesh, project_coords

class GraphDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        start_date: str,
        end_date: str,
        observation_config: Dict[str, Dict],
        mesh_structure: Dict = None,
        args = None
    ):
        super().__init__()
        
        self.z = None
        # Data parameters
        self.data_path = data_path
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.observation_config = observation_config
        self.mesh_structure = mesh_structure
        self.args = args

    def setup(self, stage=None):
        """
        Prepare data for training/validation using time-binned zarr structure.
        """
        if self.z is None:
            self.z = {}
            for obs_type in self.observation_config.keys():
                self.z[obs_type] = {}
                for key in self.observation_config[obs_type].keys():
                    data_path = os.path.join(self.data_path, key) + ".zarr"
                    self.z[obs_type][key] = zarr.open(LRUStoreCache(zarr.DirectoryStore(data_path), max_size=2_000_000_000), mode="r")
                   
        self.data_summary = organize_bins_times(self.z, self.start_date, self.end_date, self.observation_config)
        self.data_summary = extract_features(self.z, self.data_summary, self.observation_config)

        all_bin_names = list(self.data_summary.keys())
        self.sample_names = sorted(all_bin_names)
  
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample from the dataset with its observation-mesh graphs"""
        # Get the time bin name for this index
        bin_name = self.sample_names[idx]
        bin_data = self.data_summary[bin_name]
        print(bin_data.keys())
        
        # Process each observation type
        for obs_type in self.observation_config.keys():
            for inst_name in self.observation_config[obs_type].keys():
                inst_data = bin_data[obs_type][inst_name]
                # Create observation-mesh graph 
                # Project coordinates
                coords = project_coords(
                    inst_data["input_lat_deg"],  # latitude
                    inst_data["input_lon_deg"]  # longitude
                )
                
                # Create graphs for this observation
                inst_data['o2m'] = create_obs_conn_mesh(
                        coords,
                        self.mesh_structure['G_bottom_mesh'],
                        self.mesh_structure['all_mesh_nodes'],
                        self.args
                    )
                
                coords = project_coords(
                    inst_data["target_lat_deg"],  # latitude
                    inst_data["target_lon_deg"]  # longitude
                )
                inst_data['m2o'] = create_obs_conn_mesh(
                        coords,
                        self.mesh_structure['G_bottom_mesh'],
                        self.mesh_structure['all_mesh_nodes'],
                        self.args,
                        conn='m2g'
                    )
                     
        return bin_data

  
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
        
    def get_bin_time(self, idx: int) -> pd.Timestamp:
        """Get the start time of a bin"""
        bin_name = self.sample_names[idx]
        return pd.to_datetime(bin_name.split('_')[1], format='%Y%m%d_%H%M')
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