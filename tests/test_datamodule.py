"""Test script for WeatherDataModule"""

import os
import torch
from neural_lam.graph_data_module import WeatherDataModule

def main():
    # Example configuration
    data_path = "/path/to/data"  # Update this
    observation_config = {
        "temperature": {
            "station": {
                "input_vars": ["temperature"],
                "target_vars": ["temperature"],
            }
        }
    }
    
    # Create data module
    datamodule = WeatherDataModule(
        data_path=data_path,
        start_date="2020-01-01",
        end_date="2020-01-31",
        observation_config=observation_config,
        mesh_structure=None,  # Will need to provide actual mesh structure
        batch_size=8,
        num_workers=0  # Use 0 for easier debugging
    )
    
    # Setup creates and splits the dataset
    print("Setting up data module...")
    datamodule.setup()
    
    # Get a dataloader
    print("\nGetting train dataloader...")
    train_loader = datamodule.train_dataloader()
    
    # Get one batch
    print("\nGetting one batch...")
    batch = next(iter(train_loader))
    
    # Print batch structure
    print("\nBatch structure:")
    for obs_type in batch:
        print(f"\nObservation type: {obs_type}")
        for inst_name in batch[obs_type]:
            print(f"\n  Instrument: {inst_name}")
            for key, value in batch[obs_type][inst_name].items():
                if isinstance(value, torch.Tensor):
                    print(f"    {key}: Tensor of shape {value.shape}")
                elif isinstance(value, list):
                    print(f"    {key}: List of {len(value)} items")
                    if value:  # If list is not empty
                        print(f"      First item keys: {list(value[0].keys())}")

if __name__ == "__main__":
    main()
