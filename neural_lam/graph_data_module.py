"""
PyTorch Lightning DataModule for weather data using GraphDataset.
"""

import pytorch_lightning as pl
import torch
from typing import Dict, List, Optional, Union
from torch_geometric.data import Batch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from .graph_dataset import GraphDataset


def collate_weather_batch(batch: List[Dict]) -> Dict:
    """
    Collate function for batching weather data samples.
    
    Args:
        batch: List of dictionaries from GraphDataset.__getitem__
        
    Returns:
        Batched dictionary with:
        - For each observation type and instrument:
            - input_values: Input observation values
            - target_values: Target observation values
            - o2m: Dictionary with graph data for encoding
            - m2o: Dictionary with graph data for decoding
    """
    return batch

    
def collate_weather_batch_together(batch: List[Dict]) -> Dict:
    """
    Collate function for batching weather data samples.
    
    Args:
        batch: List of dictionaries from GraphDataset.__getitem__
        
    Returns:
        Batched dictionary with:
        - For each observation type and instrument:
            - input_values: Input observation values
            - target_values: Target observation values
            - o2m: Dictionary with graph data for encoding
            - m2o: Dictionary with graph data for decoding
    """
    batched = {}

    included_features = ['input_features_final', 'target_features_final','input_metadata', 'input_metadata', 'o2m', 'm2o']
    # Get first item to determine structure
    first_item = batch[0]
    
    # Process each observation type
    for obs_type in first_item.keys():
        batched[obs_type] = {}
        
        # Process each instrument
        for inst_name in first_item[obs_type].keys():
            # Get data for this observation type/instrument
            inst_data = {}
            
            for key in first_item[obs_type][inst_name].keys():
                if key in included_features:
                    # For graph data, just collect the dictionaries
                    inst_data[key] = [
                        item[obs_type][inst_name][key]
                        for item in batch
                    ]
                else:
                    print(f'{key} is excluded')
                
            
            batched[obs_type][inst_name] = inst_data
            
    return batched

class WeatherDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        start_date: str,
        end_date: str,
        observation_config: Dict[str, Dict],
        mesh_structure: Dict,
        train_val_test_split: Optional[List[float]] = None,
        batch_size: int = 8,
        num_workers: int = 4,
        args = None
    ):
        """
        DataModule for weather prediction using GraphDataset.
        
        Args:
            data_path: Path to data directory
            start_date: Start date for data range
            end_date: End date for data range
            observation_config: Configuration for observation types
            mesh_structure: Mesh graph structure
            train_val_test_split: Optional list of [train, val, test] fractions
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            args: Additional arguments passed to GraphDataset
        """
        super().__init__()
        self.data_path = data_path
        self.start_date = start_date
        self.end_date = end_date
        self.observation_config = observation_config
        self.mesh_structure = mesh_structure
        self.train_val_test_split = train_val_test_split or [0.8, 0.1, 0.1]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.args = args

        self.collate_batch = collate_weather_batch
        
        # Will be set up in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup(self, stage: Optional[str] = None):
        """Create and split dataset if not already created"""
        if self.train_dataset is None:
            # Create full dataset
            dataset = GraphDataset(
                data_path=self.data_path,
                start_date=self.start_date,
                end_date=self.end_date,
                observation_config=self.observation_config,
                mesh_structure=self.mesh_structure,
                args=self.args
            )
            dataset.setup()
            
            # Split dataset
            total_size = len(dataset)
            train_size = int(self.train_val_test_split[0] * total_size)
            val_size = int(self.train_val_test_split[1] * total_size)
            test_size = total_size - train_size - val_size
            
            self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size, test_size]
            )
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_batch
        )
    
    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        """Create validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_batch
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        """Create test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_batch
        )
  