"""
Implementation of a graph neural network model for processing heterogeneous observations
with different variable types and spatial locations.
"""

import numpy as np
import os
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from torch_geometric.data import Data, Dataset, Batch

from .ar_model import ARModel
from .. import utils
from ..interaction_net import InteractionNet
from ..create_mesh_graph import create_mesh_structure, create_obs_conn_mesh

class HeteroObservationGraphModel(ARModel):
    """
    Graph neural network model for processing heterogeneous observations.
    Handles observations of different types (temperature, wind, pressure, etc.)
    at different spatial locations.

    Workflow:
    1. Multiple observation types -> type-specific embeddings
    2. Map to mesh via spatial graph connections
    3. Process on mesh
    4. Map back to observation space
    """

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        
        # Dictionary to store observation type configurations
        self.observation_types = {}
        
        # # Process features on mesh using GNN
        # self.mesh_gnn = InteractionNet(
        #     edge_index=self.m2m_edge_index,  # Will be set from dataset
        #     input_dim=args.hidden_dim,
        #     hidden_layers=args.hidden_layers,
        #     update_edges=True
        # )
        
        # Networks for each observation type
        self.observation_embedders = nn.ModuleDict()
        
        # Use graph_dataset utilities for observation-to-mesh mapping
        self.observation_to_mesh = nn.ModuleDict()
        self.mesh_to_observation = nn.ModuleDict()
        
        # GraphModel parameters
        self.mesh_resolution = args.mesh_resolution
        self.cutoff_factor = args.cutoff_factor
        self.num_neighbors = args.num_neighbors
        
        # Feature combination layers
        self.mesh_feature_combiner = utils.make_mlp(
            [args.hidden_dim * len(self.observation_types)] + 
            [args.hidden_dim] * args.hidden_layers
        )
        
        # Attention for feature combination
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=args.hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Graph structures
        self.mesh_structure = None  # Will be set
        self.m2o_graph = None       # Will be set
        self.o2m_graph = None       # Will be set

    def setup_observation_networks(self, observation_config: Dict):
        """
        Initialize networks for each observation type.
        
        Args:
            observation_config: Dictionary mapping observation types to their configs
                {
                    'temperature': {'dim': 1, 'locations': [...], ...},
                    'wind': {'dim': 2, 'locations': [...], ...},
                    ...
                }
        """
        self.observation_types = observation_config
        
        for obs_type, config in observation_config.items():
            # Create embedder for this observation type
            self.observation_embedders[obs_type] = utils.make_mlp(
                [config['dim']] + self.mlp_blueprint_end
            )
            
            # Create graph networks for mesh mapping
            self.observation_to_mesh[obs_type] = InteractionNet(
                edge_index=None,  # Will be set in build_observation_graphs
                input_dim=self.args.hidden_dim,
                hidden_layers=self.args.hidden_layers,
                update_edges=False
            )
            
            self.mesh_to_observation[obs_type] = InteractionNet(
                edge_index=None,  # Will be set in build_observation_graphs
                input_dim=self.args.hidden_dim,
                hidden_layers=self.args.hidden_layers,
                update_edges=False
            )

    def get_num_mesh(self):
        """
        Compute number of mesh nodes from loaded features,
        and number of mesh nodes that should be ignored in encoding/decoding
        """
        # TODO
        return 100, 0  #  self.mesh_static_features.shape[0], 0
        
    def build_observation_graphs(self, observation_locations: Dict[str, torch.Tensor]):
        """
        Build edges between observation locations and mesh nodes.
        
        Args:
            observation_locations: Dict mapping observation types to their locations
                {'temperature': tensor of locations, ...}
        """
        for obs_type, locations in observation_locations.items():
            # Find K nearest mesh nodes for each observation location
            edges = utils.compute_knn_edges(
                locations,
                self.mesh_points,
                k=3  # Could be made configurable per observation type
            )
            self.obs_to_mesh_edges[obs_type] = edges
            # Reverse edges for mesh to observation mapping
            self.mesh_to_obs_edges[obs_type] = edges.flip(0)
            
            # Update edge indices in graph networks
            self.observation_to_mesh[obs_type].edge_index = edges
            self.mesh_to_observation[obs_type].edge_index = edges.flip(0)

    def combine_features(self, feature_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Combine features from different observation types using attention.
        
        Args:
            feature_list: List of tensors, each (B, N, hidden_dim)
        
        Returns:
            Combined features (B, N, hidden_dim)
        """
        # Stack features for attention
        features = torch.stack(feature_list, dim=0)  # (num_types, B, N, hidden_dim)
        
        # Apply attention
        combined, _ = self.attention_layer(
            features[0],  # query from first type
            features,     # keys from all types
            features,     # values from all types
        )
        
        return combined
    def create_mesh_structures(self):
        """
        Create or load the static mesh structure for the domain
        """
        grid_coordinates = np.load(os.path.join('/scratch1/NCEPDEV/da/Xin.C.Jin/my_projects/neural_lam/scripts/data/rrfs_15km_example/static', 
                                  '15km_rrfs-grib-grid_xy_coordinates.npy'))
        save_path = './graph_mesh'
        # Get mesh structures from cached function
        mesh_data = create_mesh_structure(
            xy=grid_coordinates,
            args=self.args,
            graph_dir_path=save_path
            )
        
        # Set instance attributes
        self.mesh_structure = mesh_data
        
    def create_obs_conn_mesh(self, bin_data):
        """
        Set graph structures from the provided GraphDataset.
        
        Args:
            dataset: GraphDataset instance containing mesh and grid structures
        """
 
        # Create observation to mesh and mesh-to-observation graphs
        
        self.o2m_graph = create_obs_conn_mesh()
        self.m2o_graph = create_obs_conn_mesh()

        # TODO Update GNN with correct edge indices
        # self.mesh_gnn.edge_index = self.mesh_structure.edge_index
        
        # Store edge features if available
        # if hasattr(dataset, 'edge_features'):
        #     self.edge_features = dataset.edge_features

    def predict_step(
        self, 
        observations: Dict[str, Tuple[torch.Tensor, torch.Tensor]], 
        prev_observations: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict next state for each observation type using the mesh structure.
        Handles batched inputs where each tensor has shape (batch_size, ...).
        
        Args:
            observations: Current observations for each type
                {'temperature': (locations [B,N,2], values [B,N,D]), ...}
            prev_observations: Previous observations if available
                Same structure as observations
        
        Returns:
            Dictionary mapping observation types to their predicted values
                {'temperature': predicted_values [B,N,D], ...}
        """
        batch_size = next(iter(observations.values()))[1].shape[0]
        device = next(iter(observations.values()))[1].device
        
        self.create_obs2mesh(observations)
        self.create_mesh2obs(observations)
        
        # Process each observation type and map to mesh
        mesh_features_list = []
        for obs_type, (locations, values) in observations.items():
            # Reshape batch dimension for locations
            flat_locations = locations.reshape(-1, 2)  # [B*N, 2]
            
            # Find nearest mesh nodes for these observations
            if obs_type not in self.obs_to_mesh_edges:
                edges = self.knn_search(flat_locations, self.mesh_graph.pos)
                
                # Adjust edge indices for batching
                num_mesh_nodes = self.mesh_graph.pos.shape[0]
                batch_offset = torch.arange(batch_size, device=device).view(-1, 1) * num_mesh_nodes
                batch_offset = batch_offset.repeat(1, edges.shape[1]).view(-1)
                
                # Apply batch offsets to edges
                edges = edges + batch_offset.unsqueeze(0)
                
                self.obs_to_mesh_edges[obs_type] = edges
                self.mesh_to_obs_edges[obs_type] = edges.flip(0)
            
            # Embed observations
            flat_values = values.reshape(-1, values.shape[-1])  # [B*N, D]
            obs_features = self.observation_embedders[obs_type](flat_values)
            
            # Map to mesh using KNN edges
            mesh_features = self.observation_to_mesh[obs_type](
                obs_features,
                None,  # No receiver features initially
                edge_index=self.obs_to_mesh_edges[obs_type]
            )
            
            # Reshape back to batched form
            mesh_features = mesh_features.view(batch_size, -1, mesh_features.shape[-1])  # [B, M, D]
            mesh_features_list.append(mesh_features)
        
        # Combine features on mesh
        mesh_features = self.combine_features(mesh_features_list)
        
        # Process on mesh using mesh graph structure (handle each batch independently)
        mesh_features_flat = mesh_features.reshape(-1, mesh_features.shape[-1])  # [B*M, D]
        
        # Adjust mesh edge indices for batching
        batch_mesh_edges = []
        num_mesh_nodes = self.mesh_graph.pos.shape[0]
        for b in range(batch_size):
            offset = b * num_mesh_nodes
            batch_edges = self.mesh_graph.edge_index + offset
            batch_mesh_edges.append(batch_edges)
        batch_mesh_edges = torch.cat(batch_mesh_edges, dim=1)
        
        # Process on mesh
        mesh_features_processed = self.mesh_gnn(
            mesh_features_flat,
            edge_index=batch_mesh_edges
        )
        
        # Reshape back to batched form
        mesh_features = mesh_features_processed.view(batch_size, -1, mesh_features_processed.shape[-1])  # [B, M, D]
        
        # Map back to observation locations
        predictions = {}
        for obs_type, (locations, values) in observations.items():
            # Flatten for message passing
            mesh_features_flat = mesh_features.reshape(-1, mesh_features.shape[-1])  # [B*M, D]
            
            # Map back using batched edges
            obs_pred = self.mesh_to_observation[obs_type](
                mesh_features_flat,
                None,  # No receiver features needed
                edge_index=self.mesh_to_obs_edges[obs_type]
            )
            
            # Reshape to match input batch shape
            obs_pred = obs_pred.view(batch_size, -1, obs_pred.shape[-1])  # [B, N, D]
            predictions[obs_type] = obs_pred
        
        return predictions

    def plot_mesh_structure(self, title: str = "Mesh Structure"):
        """
        Visualize the mesh graph structure.
        Uses create_mesh.plot_graph utility.
        """
        from .. import create_mesh
        import matplotlib.pyplot as plt
        
        fig, ax = create_mesh.plot_graph(
            self.mesh_graph,
            title=title
        )
        return fig, ax
    
    def to_graph_data(self, observations: Dict[str, Tuple[torch.Tensor, torch.Tensor]], batch_idx: int = 0) -> Data:
        """
        Convert observations to PyTorch Geometric Data object.
        
        Args:
            observations: Dictionary of observations
            batch_idx: Batch index for this data point
        
        Returns:
            PyTorch Geometric Data object
        """
        # Collect all observation features
        obs_x = []
        obs_pos = []
        obs_type_indices = []
        
        for idx, (obs_type, (locations, values)) in enumerate(observations.items()):
            obs_x.append(values)
            obs_pos.append(locations)
            obs_type_indices.extend([idx] * len(locations))
        
        # Concatenate all observations
        x = torch.cat(obs_x, dim=0)
        pos = torch.cat(obs_pos, dim=0)
        obs_type = torch.tensor(obs_type_indices, dtype=torch.long)
        
        # Create edge indices using KNN
        edge_index = self.knn_search(pos, self.mesh_graph.pos)
        
        # Create PyG Data object
        data = Data(
            x=x,
            pos=pos,
            edge_index=edge_index,
            obs_type=obs_type,
            mesh_pos=self.mesh_graph.pos,
            mesh_edge_index=self.mesh_graph.edge_index,
            batch=torch.full((len(x),), batch_idx, dtype=torch.long)
        )
        
        return data
    
    def process_batch(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """
        Process a batch of graph data.
        
        Args:
            batch: PyTorch Geometric Batch object
        
        Returns:
            Dictionary of predictions for each observation type
        """
        # Embed observations
        obs_features = []
        for obs_type, embedder in self.observation_embedders.items():
            mask = batch.obs_type == list(self.observation_types.keys()).index(obs_type)
            if mask.any():
                obs_feat = embedder(batch.x[mask])
                obs_features.append(obs_feat)
        
        # Map to mesh
        mesh_features = []
        for obs_type, mapper in self.observation_to_mesh.items():
            mask = batch.obs_type == list(self.observation_types.keys()).index(obs_type)
            if mask.any():
                mesh_feat = mapper(
                    obs_features[list(self.observation_types.keys()).index(obs_type)],
                    None,
                    edge_index=batch.edge_index
                )
                mesh_features.append(mesh_feat)
        
        # Combine and process on mesh
        mesh_features = self.combine_features(mesh_features)
        mesh_features = self.mesh_gnn(
            mesh_features,
            edge_index=batch.mesh_edge_index
        )
        
        # Map back to observations
        predictions = {}
        for obs_type in self.observation_types:
            mask = batch.obs_type == list(self.observation_types.keys()).index(obs_type)
            if mask.any():
                pred = self.mesh_to_observation[obs_type](
                    mesh_features,
                    None,
                    edge_index=batch.edge_index.flip(0)
                )
                predictions[obs_type] = pred[mask]
        
        return predictions

    def test_batched_processing(self, batch_size: int = 2, num_obs: int = 100):
        """
        Test batched processing to verify correctness.
        
        Args:
            batch_size: Number of batches to test
            num_obs: Number of observations per type
        
        Returns:
            Dict containing test results and any errors found

        Example usage:
        model = HeteroObservationGraphModel(args)
        test_results = model.test_batched_processing(batch_size=3, num_obs=100)
        """
        import torch
        import numpy as np
        
        # Create synthetic test data
        test_obs = {
            'temperature': (
                # Random 2D locations
                torch.randn(batch_size, num_obs, 2),
                # Random scalar values
                torch.randn(batch_size, num_obs, 1)
            ),
            'wind': (
                # Same locations as temperature
                torch.randn(batch_size, num_obs, 2),
                # Random 2D vector values
                torch.randn(batch_size, num_obs, 2)
            )
        }
        
        # Initialize networks if not done
        if not self.observation_embedders:
            self.setup_observation_networks({
                'temperature': {'dim': 1},
                'wind': {'dim': 2}
            })
        
        # Run prediction
        results = {}
        try:
            # Test shape preservation
            predictions = self.predict_step(test_obs)
            
            for obs_type, (locations, values) in test_obs.items():
                pred = predictions[obs_type]
                results[f"{obs_type}_shape_correct"] = (
                    pred.shape[0] == batch_size and
                    pred.shape[1] == num_obs and
                    pred.shape[2] == values.shape[2]
                )
            
            # Test edge connectivity
            for obs_type in test_obs:
                edges = self.obs_to_mesh_edges[obs_type]
                results[f"{obs_type}_edge_indices_valid"] = (
                    edges.max() < batch_size * self.mesh_graph.pos.shape[0] and
                    edges.min() >= 0
                )
            
            # Test feature propagation (no NaNs or infinities)
            for obs_type, pred in predictions.items():
                results[f"{obs_type}_values_valid"] = (
                    not torch.isnan(pred).any() and
                    not torch.isinf(pred).any()
                )
            
            # Test batch independence
            # Predictions for different batches should be different
            for obs_type, pred in predictions.items():
                batch_diffs = []
                for i in range(batch_size):
                    for j in range(i+1, batch_size):
                        diff = (pred[i] - pred[j]).abs().mean().item()
                        batch_diffs.append(diff)
                results[f"{obs_type}_batch_independent"] = np.mean(batch_diffs) > 0
            
            results["overall_success"] = all(results.values())
            
        except Exception as e:
            results["error"] = str(e)
            results["overall_success"] = False
        
        return results

    def plot_observations_on_mesh(
        self,
        observations: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        values_key: str = None,
        title: str = None,
        ax = None,
        **kwargs
    ):
        """
        Plot observations on top of the mesh structure.
        
        Args:
            observations: Dictionary of observation locations and values
            values_key: Which observation type's values to plot
            title: Plot title
            ax: Matplotlib axis to plot on
            **kwargs: Additional arguments for scatter plot
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        if ax is None:
            fig, ax = self.plot_mesh_structure(title="Mesh")
        
        # Plot observations
        for obs_type, (locations, values) in observations.items():
            if values_key is not None and obs_type != values_key:
                continue
                
            # Get colors from values if provided
            if values is not None:
                colors = values.detach().cpu().numpy()
                if len(colors.shape) > 1:
                    colors = np.mean(colors, axis=-1)  # Average if multi-dimensional
            else:
                colors = None
            
            # Plot observation points
            locations_np = locations.detach().cpu().numpy()
            scatter = ax.scatter(
                locations_np[:, 0],
                locations_np[:, 1],
                c=colors,
                label=obs_type,
                **kwargs
            )
            
            # Add colorbar if values provided
            if values is not None:
                plt.colorbar(scatter, ax=ax)
        
        if title:
            ax.set_title(title)
        ax.legend()
        
        return ax
    
    def visualize_prediction(
        self,
        observations: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        predictions: Dict[str, torch.Tensor],
        obs_type: str = None
    ):
        """
        Visualize observations and their predictions.
        
        Args:
            observations: Original observations
            predictions: Predicted values
            obs_type: Which observation type to visualize
        """
        import matplotlib.pyplot as plt
        
        if obs_type is None:
            obs_type = next(iter(observations.keys()))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot original observations
        self.plot_observations_on_mesh(
            {obs_type: observations[obs_type]},
            title=f"Original {obs_type}",
            ax=ax1
        )
        
        # Plot predictions
        pred_obs = {
            obs_type: (observations[obs_type][0], predictions[obs_type])
        }
        self.plot_observations_on_mesh(
            pred_obs,
            title=f"Predicted {obs_type}",
            ax=ax2
        )
        
        plt.tight_layout()
        return fig
