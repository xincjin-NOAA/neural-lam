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

from .process_timeseries import convert_to_time_binned_zarr
from .create_mesh import create_obs_conn_mesh, project_coords

class GraphDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        start_date: str,
        end_date: str,
        observation_config: Dict[str, Dict],
        mesh_structure: Dict = None,
        args = None,
        bin_size: str = '12h'
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
        self.bin_size = bin_size
        
        # Initialize time-binned zarr store
        self.time_binned_path = os.path.join(os.path.dirname(data_path), 'time_binned.zarr')
        if not os.path.exists(self.time_binned_path):
            convert_to_time_binned_zarr(
                input_zarr_path=data_path,
                output_zarr_path=self.time_binned_path,
                observation_config=observation_config,
                bin_size=bin_size
            )

    def setup(self, stage=None):
        """
        Prepare data for training/validation using time-binned zarr structure.
        """
        # Open the time-binned zarr store
        self.z = zarr.open(self.time_binned_path)
        
        # Get all time bin names that fall within our date range
        all_bin_names = []
        for bin_name in self.z.keys():
            # Parse bin start time from format: bin_YYYYMMDD_HHMM_to_YYYYMMDD_HHMM
            bin_start = pd.to_datetime(bin_name.split('_')[1], format='%Y%m%d_%H%M')
            if self.start_date <= bin_start < self.end_date:
                all_bin_names.append(bin_name)
        
        self.sample_names = sorted(all_bin_names)
  
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample from the dataset with its observation-mesh graphs"""
        # Get the time bin name for this index
        bin_name = self.sample_names[idx]
        bin_group = self.z[bin_name]
        
        # Create observation data dictionary
        observations = {}
        obs_graphs = {}
        
        # Process each observation type
        for obs_type in self.observation_config.keys():
            if obs_type not in bin_group:
                continue
                
            obs_group = bin_group[obs_type]
            # Combine data from all keys for this observation type
            all_features = []
            all_metadata = []
            all_lats = []
            all_lons = []
            
            for key in self.observation_config[obs_type].keys():
                if key not in obs_group:
                    continue
                    
                data_group = obs_group[key]
                # Get pre-computed features and metadata
                features = torch.tensor(data_group['features_normalized'][:], dtype=torch.float32)
                lat = torch.tensor(data_group['latitude'][:], dtype=torch.float32)
                lon = torch.tensor(data_group['longitude'][:], dtype=torch.float32)
                
                all_features.append(features)
                all_lats.append(lat)
                all_lons.append(lon)
                
                # Get metadata if available
                if 'metadata' in data_group:
                    metadata = torch.tensor(data_group['metadata'][:], dtype=torch.float32)
                    all_metadata.append(metadata)
            
            if not all_features:  # Skip if no data for this observation type
                continue
                
            # Combine all features and locations
            features = torch.cat(all_features, dim=0)
            lats = torch.cat(all_lats, dim=0)
            lons = torch.cat(all_lons, dim=0)
            locations = torch.stack([lats, lons], dim=1)
            
            # Create observation data dictionary
            obs_data = {
                'features': features,
                'locations': locations,
            }
            
            # Add metadata if available
            if all_metadata:
                metadata = torch.cat(all_metadata, dim=0)
                obs_data['metadata'] = metadata
            
            observations[obs_type] = obs_data
            
            # Create observation-mesh graph if mesh structure is available
            if self.mesh_structure is not None:
                # Project coordinates
                coords = project_coords(
                    lats.numpy(),  # latitude
                    lons.numpy()   # longitude
                )
                    
                # Create graphs for this observation type
                obs_graphs[obs_type] = {
                    'o2m': create_obs_conn_mesh(
                        coords,
                        self.mesh_structure['G_bottom_mesh'],
                        self.mesh_structure['all_mesh_nodes'],
                        self.args
                    ),
                    'm2o': create_obs_conn_mesh(
                        coords,
                        self.mesh_structure['all_mesh_nodes'],
                        self.mesh_structure['G_bottom_mesh'],
                        self.args
                    )
                }
        
        return {
            'observations': observations,
            'obs_graphs': obs_graphs if self.mesh_structure is not None else None,
            'bin_name': bin_name,
            'bin_time': self.get_bin_time(idx)
        }

  
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