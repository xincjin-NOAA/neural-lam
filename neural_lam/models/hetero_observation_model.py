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
from ..create_mesh_graph import create_mesh_structure, create_obs_conn_mesh, project_coords

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
        self.hidden_dim = args.hidden_dim
        self.num_heads = args.num_heads
        
        # Blueprint for observation MLPs
        self.mlp_blueprint = [self.hidden_dim] * args.num_layers
        
        # Dictionary to store observation type configurations
        self.observation_types = {}
        
        # Create dictionaries to store networks
        self.observation_embedders = nn.ModuleDict()
        self.observation_to_mesh = nn.ModuleDict()
        self.mesh_to_observation = nn.ModuleDict()
        
        # Create mesh processing GNN
        self.mesh_gnn = InteractionNet(
            edge_index=None,  # Will be set during forward pass
            input_dim=self.hidden_dim,
            hidden_layers=args.num_layers,
            update_edges=False
        )
        
        # Initialize mesh graph (will be set later)
        self.mesh_graph = None
        
    def create_batch_mesh_edges(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Create batched mesh edges by repeating the mesh graph for each batch item.
        
        Args:
            batch_size: Number of items in the batch
            device: Device to create tensor on
            
        Returns:
            Tensor of shape [2, E*B] containing batched edge indices
        """
        num_mesh_nodes = self.mesh_graph.pos.shape[0]
        batch_mesh_edges = []
        
        for b in range(batch_size):
            offset = b * num_mesh_nodes
            batch_edges = self.mesh_graph.edge_index + offset
            batch_mesh_edges.append(batch_edges)
            
        return torch.cat(batch_mesh_edges, dim=1)
        
    def initialize_mesh_features(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Initialize mesh features for the batch.
        
        Args:
            batch_size: Number of items in the batch
            device: Device to create tensor on
            
        Returns:
            Tensor of shape [B, M, D] containing initialized mesh features
        """
        return torch.zeros(
            batch_size,
            self.mesh_graph.pos.shape[0],
            self.hidden_dim,
            device=device
        )
            update_edges=False
        )
        
        if observation_config is not None:
            self.setup_observation_networks(observation_config)
        
        # Initialize mesh graph attributes
        self.mesh_graph = None
        self.mesh_features = None
        self.obs_to_mesh_edges = {}
        self.mesh_to_obs_edges = {}
        
        # These will be set when processing observations
        self.m2o_graph = None       # Will be set
        self.o2m_graph = None       # Will be set
        
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

    def setup_observation_networks(self, observation_config: Dict[str, Dict]):
        """
        Initialize networks for processing each observation type.
        
        Args:
            observation_config: Dictionary mapping observation types to their configurations
        """
        # Create encoder networks for each observation type
        self.observation_encoders = nn.ModuleDict()
        
        # Create embedder networks for each observation type
        self.observation_embedders = nn.ModuleDict()
        
        # Create observation-to-mesh networks
        self.observation_to_mesh = nn.ModuleDict()
        
        # Create mesh-to-observation networks
        self.mesh_to_observation = nn.ModuleDict()
        
        # Create decoder networks
        self.observation_decoders = nn.ModuleDict()
        
        for obs_type, config in observation_config.items():
            # Create encoder (input features -> hidden)
            self.observation_encoders[obs_type] = nn.Sequential(
                utils.make_mlp([config['dim'], self.hidden_dim * 2]),
                nn.LayerNorm(self.hidden_dim * 2),
                nn.ReLU(),
                utils.make_mlp([self.hidden_dim * 2, self.hidden_dim])
            )
            
            # Create embedder (processes encoded features)
            self.observation_embedders[obs_type] = nn.Sequential(
                utils.make_mlp([self.hidden_dim] + self.mlp_blueprint),
                nn.LayerNorm(self.hidden_dim)
            )
            
            # Create observation-to-mesh network
            self.observation_to_mesh[obs_type] = GATConv(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                heads=self.num_heads,
                concat=False
            )
            
            # Create mesh-to-observation network
            self.mesh_to_observation[obs_type] = GATConv(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,  # Keep in hidden dim for decoder
                heads=self.num_heads,
                concat=False
            )
            
            # Create decoder (hidden -> output features)
            self.observation_decoders[obs_type] = nn.Sequential(
                utils.make_mlp([self.hidden_dim, self.hidden_dim * 2]),
                nn.LayerNorm(self.hidden_dim * 2),
                nn.ReLU(),
                utils.make_mlp([self.hidden_dim * 2, config['dim']])
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
            self.create_obs_conn_mesh(bin_data)
            
            self.obs_to_mesh_edges[obs_type] = self.o2m_graph["edge_index"]
            # Reverse edges for mesh to observation mapping
            self.mesh_to_obs_edges[obs_type] = self.m2o_graph["edge_index"]
            
            # Update edge indices in graph networks
            self.observation_to_mesh[obs_type].edge_index = self.o2m_graph["edge_index"]
            self.mesh_to_observation[obs_type].edge_index = self.m2o_graph["edge_index"]

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
        

    def predict_step(
        self,
        batch_data: Dict
    ) -> Dict[str, torch.Tensor]:
        """
        Predict next state for each observation type using the mesh structure.
        
        Args:
            batch_data: Dictionary containing:
                - observations: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
                    {'obs_type': (locations [B,N,2], values [B,N,D]), ...}
                - obs_graphs: Dict[str, Dict[str, Data]]
                    {'obs_type': {'o2m': graph, 'm2o': graph}, ...}
        
        Returns:
            Dictionary mapping observation types to their predicted values
                {'obs_type': predicted_values [B,N,D], ...}
        """
        observations = batch_data['observations']
        obs_graphs = batch_data['obs_graphs']
        
        # Get batch info
        batch_size = next(iter(observations.values()))[0].shape[0]
        device = next(iter(observations.values()))[0].device
        
        # Initialize mesh features for the batch
        mesh_features = self.initialize_mesh_features(batch_size, device)
        
        # Process each observation type and map to mesh
        mesh_features_list = []
        for obs_type, (locations, values) in observations.items():
            # Get the graph for this observation type
            o2m_graph = obs_graphs[obs_type]['o2m']
            
            # Encode observation values
            encoded_obs = self.observation_encoders[obs_type](values)
            
            # Embed encoded features
            embedded_obs = self.observation_embedders[obs_type](encoded_obs)
            
            # Use observation graph to propagate features to mesh
            mesh_features = self.observation_to_mesh[obs_type](
                embedded_obs,
                mesh_features,  # Now properly initialized for batch
                edge_index=o2m_graph.edge_index
            )
            mesh_features_list.append(mesh_features)
        
        # Combine features from all observation types on mesh
        mesh_features = self.combine_features(mesh_features_list)
        
        # Process on mesh using InteractionNet
        mesh_features_flat = mesh_features.reshape(-1, mesh_features.shape[-1])  # [B*M, D]
        
        # Create batched mesh edges
        batch_mesh_edges = self.create_batch_mesh_edges(batch_size, device)
        
        # Process features on mesh using InteractionNet
        mesh_features_processed = self.mesh_gnn(
            x=mesh_features_flat,
            edge_index=batch_mesh_edges,
            edge_attr=None  # No edge features for now
        )
        
        # Reshape back to batched form
        mesh_features = mesh_features_processed.reshape(batch_size, -1, self.hidden_dim)
        
        # Map processed features back to observations
        predictions = {}
        for obs_type, (locations, values) in observations.items():
            # Get the graph for this observation type
            m2o_graph = obs_graphs[obs_type]['m2o']
            
            # Get encoded features for skip connection
            encoded_obs = self.observation_encoders[obs_type](values)
            
            # Propagate features back to observations
            obs_features = self.mesh_to_observation[obs_type](
                mesh_features,
                self.observation_embedders[obs_type](encoded_obs),  # Re-embed encoded for skip
                edge_index=m2o_graph.edge_index
            )
            
            # Decode final predictions
            predictions[obs_type] = self.observation_decoders[obs_type](obs_features)
        
        return predictions

  