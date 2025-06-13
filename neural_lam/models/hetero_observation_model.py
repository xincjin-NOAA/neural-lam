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

from .ar_dop_model import ARDOPModel
from .. import utils
from ..interaction_net import InteractionNet
from ..create_mesh_graph import create_mesh_structure, create_obs_conn_mesh, project_coords

from collections import defaultdict # Add this import
from .. import metrics_dop # Assuming metrics_dop is accessible like this

class HeteroObservationGraphModel(ARDOPModel): # Or pl.LightningModule if not inheriting ARDOPModel's PL logic

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
        # ... (your existing __init__ content) ...
        
        # Ensure loss function is initialized
        if not hasattr(self, 'loss_function'): # If not already inherited and set up
            self.loss_function = metrics_dop.get_metric(args.loss) # Ensure args.loss is defined
        
        # For storing outputs from validation/test steps
        self.validation_step_outputs = []
        self.test_step_outputs = []

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
        # Note: edge_index will be updated during forward pass for each batch
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
    # ... (your existing methods like create_batch_mesh_edges, predict_step, etc.) ...

    def _get_std_for_loss(self, pred_value_tensor, std_value_representation=None):
        """
        Helper to get a valid standard deviation tensor for loss functions.
        If std_value_representation is None, returns ones (for unweighted loss).
        If std_value_representation is log_var, it converts to std.
        If std_value_representation is already std, it returns it.
        Adjust this based on how your decoders output uncertainty.
        """
        if std_value_representation is None:
            return torch.ones_like(pred_value_tensor)
        
        # Example: if your decoder outputs log_variance for numerical stability
        # actual_std = torch.exp(0.5 * std_value_representation)
        # return actual_std
        
        # For now, assume std_value_representation is already the actual std if not None
        return std_value_representation


    def _compute_batch_loss_and_outputs(self, predictions_dict, target_features_dict, pred_std_dict=None, compute_metrics_for_epoch_end=False):
        """
        Computes aggregated batch loss and collects outputs for epoch-end metrics.

        Args:
            predictions_dict (dict): Model predictions.
                {'obs_type': {'inst_name': pred_tensor, ...}, ...}
            target_features_dict (dict): Ground truth targets.
                {'obs_type': {'inst_name': target_tensor, ...}, ...}
            pred_std_dict (dict, optional): Predicted standard deviations (or their representations).
                Structure similar to predictions_dict.
            compute_metrics_for_epoch_end (bool): If True, also collects detached tensors
                                                 for epoch-end calculations.

        Returns:
            torch.Tensor: Scalar batch loss.
            dict (optional): If compute_metrics_for_epoch_end is True, returns a dictionary
                             of detached predictions, targets, and std_devs for metric calculation.
                             Otherwise, None.
        """
        accumulated_loss_terms = []
        outputs_for_metrics_collection = defaultdict(lambda: defaultdict(lambda: {'preds': None, 'targets': None, 'pred_stds': None}))

        for obs_type in predictions_dict:
            if obs_type not in target_features_dict:
                # print(f"Warning: Targets missing for obs_type {obs_type}")
                continue
            for inst_name in predictions_dict[obs_type]:
                if inst_name not in target_features_dict[obs_type]:
                    # print(f"Warning: Targets missing for {obs_type}/{inst_name}")
                    continue

                pred_values = predictions_dict[obs_type][inst_name]
                true_values = target_features_dict[obs_type][inst_name]
                
                current_pred_std_repr = None
                if pred_std_dict and \
                   obs_type in pred_std_dict and \
                   inst_name in pred_std_dict[obs_type] and \
                   pred_std_dict[obs_type][inst_name] is not None:
                    current_pred_std_repr = pred_std_dict[obs_type][inst_name]

                # Get a valid std tensor for the loss function
                # This helper encapsulates logic like converting log_var to std if needed
                actual_pred_std_for_loss = self._get_std_for_loss(pred_values, current_pred_std_repr)

                # Placeholder for actual mask if available per observation type/instance
                obs_point_mask = None 

                loss_term = self.loss_function(
                    pred_values,
                    true_values,
                    actual_pred_std_for_loss,
                    mask=obs_point_mask
                ) # Expected to return (B,) or similar if averaging over N, summing over D
                accumulated_loss_terms.append(loss_term)

                if compute_metrics_for_epoch_end:
                    obs_key = f"{obs_type}_{inst_name}"
                    outputs_for_metrics_collection[obs_key]['preds'] = pred_values.detach()
                    outputs_for_metrics_collection[obs_key]['targets'] = true_values.detach()
                    outputs_for_metrics_collection[obs_key]['pred_stds'] = actual_pred_std_for_loss.detach() # Store the std used for loss

        if not accumulated_loss_terms:
            final_batch_loss = torch.tensor(0.0, device=self.device, requires_grad=True if self.training else False)
        else:
            # Stack losses for different observation types/instances.
            # If each loss_term is (B,), stacking makes it (num_loss_terms, B)
            stacked_losses = torch.stack(accumulated_loss_terms) # Shape: (num_obs_types_processed, B)
            
            # Mean across different observation types/instances, then mean across batch
            mean_loss_across_obs_types = torch.mean(stacked_losses, dim=0) # Shape (B,)
            final_batch_loss = torch.mean(mean_loss_across_obs_types)     # Scalar
        
        returned_metrics_outputs = dict(outputs_for_metrics_collection) if compute_metrics_for_epoch_end else None
        return final_batch_loss, returned_metrics_outputs

    def training_step(self, batch, batch_idx):
        # Assuming batch structure: {'observations': {...}, 'targets': {...}, ...}
        # 'observations' is the input to predict_step
        # 'targets' is the ground truth for loss calculation
        
        # The input to predict_step might be the whole batch or a part of it
        # Adjust based on your DataModule and predict_step signature
        # For this example, assume batch_data for predict_step is batch itself or batch['observations']
        batch_data_for_model = batch # Or batch.get('observations_input_format_for_predict_step')

        predictions_dict, pred_std_dict = self.predict_step(batch_data_for_model)
        
        # target_features_dict = batch['targets'] # Assuming targets are in batch['targets']
        observations = batch_data_for_model[0]
        target_features_dict = {}
        for obs_type in observations:
            target_features_dict[obs_type] = {}
            pred_std_dict[obs_type] = {} # Initialize inner dict for std devs
            for inst_name in observations[obs_type]:
                bin_data = observations[obs_type][inst_name] 
                target_features_dict[obs_type][inst_name] = bin_data["target_features_final"]

        batch_loss, _ = self._compute_batch_loss_and_outputs(
            predictions_dict,
            target_features_dict,
            pred_std_dict,
            compute_metrics_for_epoch_end=False # Not needed for training_step outputs
        )

        #self.log("train_loss", batch_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return batch_loss

    def validation_step(self, batch, batch_idx):
        batch_data_for_model = batch
        predictions_dict, pred_std_dict = self.predict_step(batch_data_for_model)
        target_features_dict = batch['targets']

        batch_loss, outputs_for_metrics = self._compute_batch_loss_and_outputs(
            predictions_dict,
            target_features_dict,
            pred_std_dict,
            compute_metrics_for_epoch_end=True
        )
        
        self.log("val_loss_step", batch_loss, on_step=True, on_epoch=False, sync_dist=True)
        # Store all necessary parts for on_validation_epoch_end
        # The 'outputs_for_metrics' already contains detached preds, targets, stds
        self.validation_step_outputs.append({
            'loss': batch_loss.detach(), 
            'metrics_data': outputs_for_metrics
        })
        # No explicit return needed if self.validation_step_outputs is used in on_validation_epoch_end

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        # Aggregate losses
        avg_epoch_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        self.log("val_loss_epoch", avg_epoch_loss, sync_dist=True)

        # Aggregate metrics_data
        # This will be a dict: {'obs_type_inst_name': {'preds': [tensors], 'targets': [tensors], 'pred_stds': [tensors]}}
        aggregated_metrics_data = defaultdict(lambda: {'preds': [], 'targets': [], 'pred_stds': []})

        for step_output in self.validation_step_outputs:
            for obs_key, data_dict in step_output['metrics_data'].items():
                aggregated_metrics_data[obs_key]['preds'].append(data_dict['preds'])
                aggregated_metrics_data[obs_key]['targets'].append(data_dict['targets'])
                if data_dict['pred_stds'] is not None: # Should always be a tensor now due to _get_std_for_loss
                    aggregated_metrics_data[obs_key]['pred_stds'].append(data_dict['pred_stds'])
        
        log_dict_epoch_metrics = {}
        for obs_key, data_lists in aggregated_metrics_data.items():
            all_preds = torch.cat(data_lists['preds'], dim=0)
            all_targets = torch.cat(data_lists['targets'], dim=0)
            all_pred_stds_for_loss = torch.cat(data_lists['pred_stds'], dim=0) # Stds used in loss

            # Re-calculate primary loss metric (e.g., WMSE) on all aggregated data for this obs_key
            # This ensures the epoch metric is based on the full validation set for that obs_type
            metric_values = self.loss_function(all_preds, all_targets, all_pred_stds_for_loss, mask=None)
            avg_metric = torch.mean(metric_values)
            log_dict_epoch_metrics[f"val_metric_{obs_key}"] = avg_metric # e.g. val_wmse_obsType_instName

            # You can add other metrics here, e.g., unweighted RMSE
            # rmse_fn = metrics_dop.get_metric("rmse") # You'd need to implement rmse or derive from mse
            # For RMSE, you'd typically use unweighted MSE first, then sqrt
            unweighted_mse_values = metrics_dop.mse(all_preds, all_targets, torch.ones_like(all_preds), mask=None)
            rmse_value = torch.sqrt(torch.mean(unweighted_mse_values))
            log_dict_epoch_metrics[f"val_rmse_{obs_key}"] = rmse_value

        self.log_dict(log_dict_epoch_metrics, sync_dist=True)
        self.validation_step_outputs.clear() # Important!

    def test_step(self, batch, batch_idx):
        # Analogous to validation_step
        batch_data_for_model = batch
        predictions_dict, pred_std_dict = self.predict_step(batch_data_for_model)
        target_features_dict = batch['targets']

        batch_loss, outputs_for_metrics = self._compute_batch_loss_and_outputs(
            predictions_dict,
            target_features_dict,
            pred_std_dict,
            compute_metrics_for_epoch_end=True
        )
        
        self.log("test_loss_step", batch_loss, on_step=True, on_epoch=False, sync_dist=True)
        self.test_step_outputs.append({
            'loss': batch_loss.detach(),
            'metrics_data': outputs_for_metrics
        })

    def on_test_epoch_end(self):
        # Analogous to on_validation_epoch_end, using self.test_step_outputs
        # and logging with "test_" prefix.
        if not self.test_step_outputs:
            return

        avg_epoch_loss = torch.stack([x['loss'] for x in self.test_step_outputs]).mean()
        self.log("test_loss_epoch", avg_epoch_loss, sync_dist=True)

        aggregated_metrics_data = defaultdict(lambda: {'preds': [], 'targets': [], 'pred_stds': []})
        for step_output in self.test_step_outputs:
            for obs_key, data_dict in step_output['metrics_data'].items():
                aggregated_metrics_data[obs_key]['preds'].append(data_dict['preds'])
                aggregated_metrics_data[obs_key]['targets'].append(data_dict['targets'])
                if data_dict['pred_stds'] is not None:
                     aggregated_metrics_data[obs_key]['pred_stds'].append(data_dict['pred_stds'])
        
        log_dict_epoch_metrics = {}
        for obs_key, data_lists in aggregated_metrics_data.items():
            all_preds = torch.cat(data_lists['preds'], dim=0)
            all_targets = torch.cat(data_lists['targets'], dim=0)
            all_pred_stds_for_loss = torch.cat(data_lists['pred_stds'], dim=0)

            metric_values = self.loss_function(all_preds, all_targets, all_pred_stds_for_loss, mask=None)
            avg_metric = torch.mean(metric_values)
            log_dict_epoch_metrics[f"test_metric_{obs_key}"] = avg_metric
            
            unweighted_mse_values = metrics_dop.mse(all_preds, all_targets, torch.ones_like(all_preds), mask=None)
            rmse_value = torch.sqrt(torch.mean(unweighted_mse_values))
            log_dict_epoch_metrics[f"test_rmse_{obs_key}"] = rmse_value

        self.log_dict(log_dict_epoch_metrics, sync_dist=True)
        self.test_step_outputs.clear() # Important!

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
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
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

        # assume batch size as 1
        observations = batch_data[0]

        # Get batch info
        batch_size = 1  # next(iter(observations.values()))[0].shape[0]
        device = observations['satellite']['atms']['input_features_final'][0].device
        # next(iter(observations.values()))[0].device
        
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
                print(o2m_graph.keys())

                # Encode observation values
                values = bin_data["input_features_final"]
                encoded_obs = self.observation_encoders[obs_type_str](values)

                # Embed encoded features
                embedded_obs = self.observation_embedders[obs_type_str](encoded_obs)
                print(o2m_graph["graph"].edge_index.shape)
                print(embedded_obs.shape)
                print("embedded_obs shape:", embedded_obs.shape)
                print("mesh_features shape:", mesh_features.shape)
                print("edge_index max values:",
                      o2m_graph["graph"].edge_index[0].max().item(),
                      o2m_graph["graph"].edge_index[1].max().item())

                # Use observation graph to propagate features to mesh
                mesh_features = self.observation_to_mesh[obs_type_str](
                    (embedded_obs, mesh_features),
                    edge_index=o2m_graph["graph"].edge_index
                )
                mesh_features_list.append(mesh_features)

        # Combine features from all observation types on mesh
        mesh_features = self.combine_features(mesh_features_list)

        # Process on mesh using InteractionNet
        mesh_features_flat = mesh_features.reshape(-1, mesh_features.shape[-1])  # [B*M, D]

        # Create batched mesh edges
        batch_mesh_edges = self.create_batch_mesh_edges(batch_size, device)

        self.mesh_gnn.edge_index = batch_mesh_edges
        # Process features on mesh using InteractionNet
        mesh_features_processed = self.mesh_gnn(
            mesh_features_flat,
            mesh_features_flat,
            None  # No edge features for now
        )

        # Reshape back to batched form
        mesh_features = mesh_features_flat.reshape(batch_size, -1, self.hidden_dim)

        # Map processed features back to observations
        predictions = {}
        pred_std_dict = {} # Initialize dictionary for standard deviations
        for obs_type in observations:
            predictions[obs_type] = {}
            pred_std_dict[obs_type] = {} # Initialize inner dict for std devs
            for inst_name in observations[obs_type]:
                obs_type_str = f'{obs_type}_{inst_name}'
                bin_data = observations[obs_type][inst_name]

                # Initialize observation features with zeros
                m2o_graph = bin_data['m2o']
                num_obs = m2o_graph['graph'].grid_pos.shape[0]
                obs_features = torch.zeros(
                    batch_size, num_obs, self.hidden_dim,
                    device=mesh_features.device
                )

                # Propagate features from mesh to observations using reversed edges
                # Flip edge indices since we're going mesh->grid instead of grid->mesh
                m2g_edges = m2o_graph['graph'].edge_index
                print(m2o_graph["graph"].edge_index.shape)
                print(embedded_obs.shape)
                print("embedded_obs shape:", obs_features.shape)
                print("mesh_features shape:", mesh_features.shape)
                print("edge_index max values:",
                      m2o_graph["graph"].edge_index[0].max().item(),
                      m2o_graph["graph"].edge_index[1].max().item())
                obs_features = self.mesh_to_observation[obs_type_str](
                    (mesh_features[0], obs_features[0]),  # (mesh_features, grid_features)
                    edge_index=m2g_edges
                )

                # Decode predictions from mesh-derived features
                decoded_output = self.observation_decoders[obs_type_str](obs_features)

                if isinstance(decoded_output, tuple) and len(decoded_output) == 2:
                    predictions[obs_type][inst_name] = decoded_output[0]  # Mean
                    # Store the std representation (e.g., actual std, or log_var to be processed later)
                    pred_std_dict[obs_type][inst_name] = decoded_output[1] 
                else:  # Decoder only outputs mean
                    predictions[obs_type][inst_name] = decoded_output
                    pred_std_dict[obs_type][inst_name] = None  # Or a tensor of ones if needed by loss
        
        return predictions, pred_std_dict
