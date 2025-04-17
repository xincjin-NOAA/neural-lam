from torch.utils.data import Dataset
from torch_geometric.data import Data
import zarr
import pandas as pd
from timing_utils import organize_bins_times
from process_timeseries import extract_features

class GraphDataset(Dataset):
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


    def create_dynamic_graph(self, weather_data):
        g2m_data = create_grid_to_mesh(
            coords, 
            G_bottom_mesh, 
            all_mesh_nodes, 
            args,
            proj_params=proj_params
        )

        m2g_data = create_mesh_to_grid(
            coords,
            G_g2m,
            vm,
            args,
            graph_dir_path,
            proj_params=proj_params
        )
    
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