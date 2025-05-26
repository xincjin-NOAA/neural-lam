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
from torch_geometric.nn import GATConv
import torch
import torch.nn as nn

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
        self.mlp_blueprint_end = [args.hidden_dim] * (args.hidden_layers + 1)
        
        # Initialize network dictionaries
        self.observation_encoders = nn.ModuleDict()
        self.observation_embedders = nn.ModuleDict()
        self.observation_to_mesh = nn.ModuleDict()
        self.mesh_to_observation = nn.ModuleDict()
        self.observation_decoders = nn.ModuleDict()
        
        # Initialize graph structures
        self.mesh_structure = None  # Will be set in setup_mesh
        self.mesh_graph = None      # Will be set in setup_mesh
        self.m2o_graph = None       # Will be set during forward pass
        self.o2m_graph = None       # Will be set during forward pass

        self.create_mesh_structures()
        
        # Create mesh processing GNN
        self.mesh_gnn = InteractionNet(
            edge_index=self.mesh_graph.m2m_graphs[0].edge_index,  
            input_dim=self.hidden_dim,
            hidden_layers=args.hidden_layers,
            update_edges=False
        )
        
        # Feature combination layers
        self.mesh_feature_combiner = utils.make_mlp(
            [self.hidden_dim * len(args.observation_config)] + 
            [self.hidden_dim] * args.hidden_layers
        )
        
        # Attention for feature combination
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            batch_first=True
        )
        
        # Set up observation networks if config provided
        if hasattr(args, 'observation_config') and args.observation_config:
            self.setup_observation_networks(args.observation_config)
        
    def create_batch_mesh_edges(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Create batched mesh edges by repeating the mesh graph for each batch item.
        
        Args:
            batch_size: Number of items in the batch
            device: Device to create tensor on
            
        Returns:
            Tensor of shape [2, E*B] containing batched edge indices
        """
        num_mesh_nodes = self.get_num_mesh()
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
        # Initialize mesh features with zeros
        # Shape: [batch_size, num_mesh_nodes, hidden_dim]
        mesh_features = torch.zeros(
            batch_size,
            self.mesh_graph.m2m_graphs[0].pos.shape[0],  # Using num_mesh_nodes directly
            self.hidden_dim,
            device=device
        )
        
        # Optionally, we could initialize with mesh node positions
        # mesh_pos = self.mesh_graph.mesh_pos  # [M, 2]
        # mesh_pos_expanded = mesh_pos.unsqueeze(0).expand(batch_size, -1, -1)  # [B, M, 2]
        # mesh_features[..., :2] = mesh_pos_expanded  # Initialize first 2 dims with positions
        
        return mesh_features

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
        obs_dim = 3 # TODO make sure haw to set up this
        for obs_t in observation_config.keys():
            for inst_name, inst_config in observation_config[obs_t].items():
                obs_type = f'{obs_t}_{inst_name}'
                print(inst_config)
                input_dim = inst_config['input_dim']
                target_dim = inst_config['target_dim']
                # Create encoder (input features -> hidden)
                self.observation_encoders[obs_type] = nn.Sequential(
                    utils.make_mlp([input_dim, self.hidden_dim * 2]),
                    nn.LayerNorm(self.hidden_dim * 2),
                    nn.ReLU(),
                    utils.make_mlp([self.hidden_dim * 2, self.hidden_dim])
                )
                
                # Create embedder (processes encoded features)
                self.observation_embedders[obs_type] = nn.Sequential(
                    utils.make_mlp([self.hidden_dim] + self.mlp_blueprint_end),
                    nn.LayerNorm(self.hidden_dim)
                )
                
                # Create observation-to-mesh network
                self.observation_to_mesh[obs_type] = GATConv(
                    in_channels=(self.hidden_dim, self.hidden_dim),  # (grid_dim, mesh_dim)
                    out_channels=self.hidden_dim,
                    heads=self.num_heads,
                    concat=False
                )
                
                # Create mesh-to-observation network
                self.mesh_to_observation[obs_type] = GATConv(
                    in_channels=(self.hidden_dim, self.hidden_dim),  # (mesh_dim, grid_dim)
                    out_channels=self.hidden_dim,
                    heads=self.num_heads,
                    concat=False
                )
                
                # Create decoder (hidden -> output features)
                self.observation_decoders[obs_type] = nn.Sequential(
                    utils.make_mlp([self.hidden_dim, self.hidden_dim * 2]),
                    nn.LayerNorm(self.hidden_dim * 2),
                    nn.ReLU(),
                    utils.make_mlp([self.hidden_dim * 2, target_dim])
                )

    def get_num_mesh(self) -> Tuple[int, int]:
        """
        Get the number of mesh nodes and number of nodes to ignore.
        
        Returns:
            Tuple containing:
            - Number of mesh nodes in the graph
            - Number of mesh nodes to ignore (usually 0)
        
        Raises:
            RuntimeError: If mesh_graph is not initialized
        """
        if self.mesh_graph is None:
            raise RuntimeError("Mesh graph must be initialized before calling get_num_mesh")
            
        return self.mesh_graph.m2m_graphs[0].pos.shape[0]  # No nodes are ignored in our implementation
        
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


        return torch.mean(features, dim=0) # (num_mesh, hidden_dim]
        # # Apply attention
        # combined, _ = self.attention_layer(
        #     features[0],  # query from first type
        #     features,     # keys from all types
        #     features,     # values from all types
        # )
        
        # return combined

    def create_mesh_structures(self) -> None:
        """
        Create or load the static mesh structure for the domain.
        This initializes both self.mesh_structure and self.mesh_graph.
        
        The mesh structure contains the basic grid topology and connectivity,
        while mesh_graph is the PyG Data object used for graph operations.
        
        Raises:
            FileNotFoundError: If grid coordinates file cannot be found
            RuntimeError: If mesh creation fails
        """
        try:
            # Load grid coordinates
            grid_path = os.path.join(
                '/scratch1/NCEPDEV/da/Xin.C.Jin/my_projects/neural_lam/scripts/data/rrfs_15km_example/static',
                '15km_rrfs-grib-grid_xy_coordinates.npy'
            )
            grid_coordinates = np.load(grid_path)
            
            # Create or load mesh structure
            save_path = './graph_mesh'
            mesh_data = create_mesh_structure(
                xy=grid_coordinates,
                args=self.args,
                graph_dir_path=save_path
            )
            
            # Store both the mesh structure and graph
            self.mesh_structure = mesh_data
            self.mesh_graph = Data(
                pos=torch.from_numpy(grid_coordinates).float(),
                m2m_graphs = mesh_data['m2m_graphs'],
                edge_index=mesh_data['m2m_graphs'][0].edge_index,
                edge_attr=mesh_data['m2m_graphs']['edge_attr'] if 'edge_attr' in mesh_data else None
            )
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not find grid coordinates file: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to create mesh structure: {e}")

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
        for obs_type in observations.keys():
            for inst_name in observations[obs_type].keys():
                obs_type_str = f'{obs_type}_{inst_name}'
                bin_data = observations[obs_type][inst_name]
                # Get the graph for this observation type
                o2m_graph = bin_data['o2m']
                
                # Encode observation values
                values = bin_data["input_features_final"]
                encoded_obs = self.observation_encoders[obs_type_str](values)
                
                # Embed encoded features
                embedded_obs = self.observation_embedders[obs_type_str](encoded_obs)
                
                # Use observation graph to propagate features to mesh
                mesh_features = self.observation_to_mesh[obs_type_str](
                    (embedded_obs, mesh_features),  # (grid_features, mesh_features)
                    edge_index=o2m_graph['g2m_graph'].edge_index
                )
                mesh_features_list.append(mesh_features)
        
        # Combine features from all observation types on mesh
        mesh_features = self.combine_features(mesh_features_list)
        
        # Process on mesh using InteractionNet
        mesh_features_flat = mesh_features.reshape(-1, mesh_features.shape[-1])  # [B*M, D]
        
        # Create batched mesh edges
        batch_mesh_edges = self.create_batch_mesh_edges(batch_size, device)
        
        # Process features on mesh using InteractionNet
        # Process features on mesh using InteractionNet
        # Note: mesh_features are both senders and receivers in mesh-to-mesh communication
        mesh_features_processed = self.mesh_gnn(
            mesh_features_flat,     # send_rep: Mesh nodes as senders
            mesh_features_flat,     # rec_rep: Same nodes as receivers
            None                   # edge_rep: No edge features for now
        )
        
        # Reshape back to batched form
        mesh_features = mesh_features_processed.reshape(batch_size, -1, self.hidden_dim)
        
        # Map processed features back to observations
        predictions = {}
        for obs_type in observations:
            predictions[obs_type] = {}
            for inst_name in observations[obs_type]:
                obs_type_str = f'{obs_type}_{inst_name}'
                bin_data = observations[obs_type][inst_name]
                
                # Initialize observation features with zeros
                num_obs = bin_data['o2m']['g2m_graph'].grid_pos.shape[0]
                obs_features = torch.zeros(
                    batch_size, num_obs, self.hidden_dim,
                    device=mesh_features.device
                )
                
                # Propagate features from mesh to observations using reversed edges
                # Flip edge indices since we're going mesh->grid instead of grid->mesh
                g2m_edges = bin_data['o2m']['g2m_graph'].edge_index
                m2g_edges = torch.stack([g2m_edges[1], g2m_edges[0]], dim=0)
                
                obs_features = self.mesh_to_observation[obs_type_str](
                    (mesh_features, obs_features),  # (mesh_features, grid_features)
                    edge_index=m2g_edges
                )
                
                # Decode predictions from mesh-derived features
                predictions[obs_type][inst_name] = self.observation_decoders[obs_type_str](obs_features)
        
        return predictions

  