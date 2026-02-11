"""
Data preparation module for warehouse inventory and order fulfillment.

This module processes inventory, distance, and ETA data to create batched inputs
for machine learning models.
"""

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict, Tuple, Any


class DataPreparator:
    """Handles data preparation for warehouse order fulfillment prediction."""
    
    def __init__(self, 
                 path):
        """
        Initialize the data preparator with file paths.
        
        Args:
            inventory_path: Path to inventory data CSV
            distance_path: Path to pincode-warehouse distance CSV
            eta_path: Path to ETA (Estimated Time of Arrival) CSV
        """
        self.inventory_path = f'{path}/inventory_train.csv'
        self.distance_path = f'{path}/pincode_warehouse_distance.csv'
        self.eta_path = f'{path}/eta_hour.csv'
        
        # Data containers
        self.inv_dict = None
        self.dist_pivot = None
        self.eta_dicts = None
    
    def load_and_process_inventory(self) -> Dict[Tuple[str, str], List]:
        """
        Load and process inventory data.
        
        Returns:
            Dictionary mapping (DATE, SKU_ID) to inventory quantities
        """
        inv = pd.read_csv(self.inventory_path)
        
        # Remove duplicates and pivot data
        inventory_processed = (
            inv.drop_duplicates(subset=['DATE', 'SKU_ID', 'WAREHOUSE_CODE'])
            .set_index(['DATE', 'SKU_ID', 'WAREHOUSE_CODE'])
            .unstack(fill_value=0)
            .stack()
            .reset_index()
            .sort_values(by=['DATE', 'SKU_ID', 'WAREHOUSE_CODE'])
        )
        
        # Create dictionary of inventory quantities
        inv_dict = (
            inventory_processed
            .groupby(['DATE', 'SKU_ID'])['QUANTITY']
            .apply(list)
            .to_dict()
        )
        
        return inv_dict
    
    def load_and_process_distances(self) -> pd.DataFrame:
        """
        Load and process warehouse distance data.
        
        Returns:
            Pivot table of distances by pincode and warehouse
        """
        distance = pd.read_csv(self.distance_path)
        
        dist_pivot = (
            distance
            .drop_duplicates(['WAREHOUSE_CODE', 'CENTRAL_PINCODE'])
            .pivot(index='CENTRAL_PINCODE', 
                   columns='WAREHOUSE_CODE', 
                   values='distance')
            .replace(np.nan, 1000)  # Fill missing distances with large value
        )
        
        return dist_pivot
    
    def load_and_process_eta(self) -> Dict[Tuple[int, int], pd.DataFrame]:
        """
        Load and process ETA (Estimated Time of Arrival) data.
        
        Returns:
            Dictionary mapping (hour, is_fast_delivery) to ETA pivot tables
        """
        eta = pd.read_csv(self.eta_path)
        
        # Remove duplicates and pivot
        eta_pivoted = (
            eta
            .drop_duplicates(['WAREHOUSE_CODE', 'CENTRAL_PINCODE', 'HOUR', 'DELIVERY_TYPE'])
            .pivot_table(index=['WAREHOUSE_CODE', 'CENTRAL_PINCODE', 'HOUR'],
                        columns='DELIVERY_TYPE',
                        values='ETA_TO_MEAN')
            .reset_index()
        )
        
        eta_pivoted.columns = ['WAREHOUSE_CODE', 'CENTRAL_PINCODE', 'HOUR', 
                               'slow_delivery_eta_to', 'fast_delivery_eta_to']
        
        # Create ETA dictionaries for each hour and delivery type
        eta_dicts = {}
        
        for hour in range(24):
            hour_data = eta_pivoted[eta_pivoted.HOUR == hour]
            
            # fast_delivery delivery ETA
            fast_delivery_eta = (
                hour_data[['WAREHOUSE_CODE', 'CENTRAL_PINCODE', 'fast_delivery_eta_to']]
                .drop_duplicates(subset=['WAREHOUSE_CODE', 'CENTRAL_PINCODE'])
                .pivot(index='CENTRAL_PINCODE', 
                       columns='WAREHOUSE_CODE', 
                       values='fast_delivery_eta_to')
                .replace(np.nan, 150)
            )
            eta_dicts[(hour, 1)] = fast_delivery_eta
            
            # Non-fast_delivery delivery ETA
            slow_delivery_eta = (
                hour_data[['WAREHOUSE_CODE', 'CENTRAL_PINCODE', 'slow_delivery_eta_to']]
                .drop_duplicates(subset=['WAREHOUSE_CODE', 'CENTRAL_PINCODE'])
                .pivot(index='CENTRAL_PINCODE', 
                       columns='WAREHOUSE_CODE', 
                       values='slow_delivery_eta_to')
                .replace(np.nan, 150)
            )
            eta_dicts[(hour, 0)] = slow_delivery_eta
        
        return eta_dicts
    
    def add_inventory_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add inventory-related features to the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with inventory features added
        """
        if self.inv_dict is None:
            self.inv_dict = self.load_and_process_inventory()
        
        def get_inventory(row):
            date_key = str(row.ORDER_DATE.date())
            sku_key = row['SKU_ID']
            return self.inv_dict.get((date_key, sku_key))
        
        df['inv'] = df.apply(get_inventory, axis=1)
        df = df[df.inv.notnull()].copy()
        
        return df
    
    def add_distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add distance-related features to the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with distance features added
        """
        if self.dist_pivot is None:
            self.dist_pivot = self.load_and_process_distances()
        
        df['dist'] = df.CENTRAL_PINCODE.map(lambda x: self.dist_pivot.loc[x].to_list())
        
        return df
    
    def add_eta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add ETA (Estimated Time of Arrival) features to the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with ETA features added
        """
        if self.eta_dicts is None:
            self.eta_dicts = self.load_and_process_eta()
        
        def get_eta(row):
            hour = row.ORDER_DATE.hour
            is_fast_delivery = row['IS_fast_delivery']
            pincode = row['CENTRAL_PINCODE']
            return self.eta_dicts[(hour, is_fast_delivery)].loc[pincode].to_list()
        
        df['eta'] = df.apply(get_eta, axis=1)
        
        return df
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features (hour encoding) to the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with temporal features added
        """
        df['HOUR'] = df.ORDER_DATE.dt.hour
        df['SIN_HOUR'] = np.sin(df['HOUR'] * 2 * np.pi / 23)
        df['COS_HOUR'] = np.cos(df['HOUR'] * 2 * np.pi / 23)
        
        return df
    
    def create_combined_inputs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine all features into a single input vector.
        
        Args:
            df: Input dataframe with all features
            
        Returns:
            Dataframe with combined input vectors
        """
        def combine_features(row):
            return np.concatenate([
                [row['IS_fast_delivery']],
                [row['COLD_STORAGE'], row['SIN_HOUR'], row['COS_HOUR']],
                np.array(row['eta']) / 240,  # Normalize ETA
                row['dist'],
                row['inv']
            ])
        
        df['inputs'] = df.apply(combine_features, axis=1)
        
        return df
    
    def preprocess_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess and split groups for batching.
        
        Args:
            df: Input dataframe
            
        Returns:
            Preprocessed dataframe with group splits
        """
        df = df.drop_duplicates(['GROUP_ID', 'ORDER_ID', 'SKU_ID'])
        
        # Rank SKUs within each group
        df['sku_rank'] = df.groupby('GROUP_ID').SKU_ID.rank()
        
        # Split groups into sub-groups of 10
        df['split_11'] = df['sku_rank'] // 10
        df['GROUP_ID'] = df['GROUP_ID'].astype(str) + df['split_11'].astype(str)
        
        return df
    
    def prepare_data(self, df: pd.DataFrame, expected_input_size: int = 149) -> List[Dict]:
        """
        Main data preparation pipeline.
        
        Args:
            df: Input dataframe
            expected_input_size: Expected size of input vectors
            
        Returns:
            List of batched data dictionaries
        """
        # Add all features
        df = self.add_inventory_features(df)
        df = self.add_distance_features(df)
        df = self.add_eta_features(df)
        df = self.add_temporal_features(df)
        df = self.create_combined_inputs(df)
        df = self.preprocess_groups(df)
        
        return df
    
    def create_batches(self, 
                      df: pd.DataFrame, 
                      input_size: int,
                      num_of_batch: int = 5, 
                      max_seq_len: int = 10,
                      device: str = 'cpu') -> List[Dict[str, Any]]:
        """
        Create batched data for model training/inference.
        
        Args:
            df: Input dataframe
            input_size: Size of each input vector
            num_of_batch: Number of batches to create
            max_seq_len: Maximum sequence length (for padding)
            device: Device to place tensors on
            
        Returns:
            List of batch dictionaries containing inputs, masks, and metadata
        """
        batch_list = []
        unique_groups = df.GROUP_ID.unique()
        batch_size = (len(unique_groups) // num_of_batch) + 1
        
        # Split groups into batches
        batches = [
            unique_groups[i * batch_size : (i + 1) * batch_size] 
            for i in range(num_of_batch)
        ]
        
        # Add SKU count per group
        df['sku_count'] = df.groupby('GROUP_ID').SKU_ID.transform('count')
        df = df.sort_values('GROUP_ID')
        
        for batch_groups in tqdm(batches, desc="Creating batches"):
            batch_data = self._process_single_batch(
                df, batch_groups, input_size, max_seq_len, device
            )
            batch_list.append(batch_data)
        
        return batch_list
    
    def _process_single_batch(self,
                             df: pd.DataFrame,
                             batch_groups: np.ndarray,
                             input_size: int,
                             max_seq_len: int,
                             device: str) -> Dict[str, Any]:
        """
        Process a single batch of data.
        
        Args:
            df: Input dataframe
            batch_groups: Group IDs for this batch
            input_size: Size of input vectors
            max_seq_len: Maximum sequence length
            device: Device for tensors
            
        Returns:
            Dictionary containing batch data
        """
        df_batch = df[df.GROUP_ID.isin(batch_groups)]
        
        # Create padded input sequences
        sku_inputs = []
        for group_id, group_inputs in df_batch.groupby('GROUP_ID').inputs:
            input_array = np.array(group_inputs.to_list()).reshape((len(group_inputs), input_size))
            padding = np.zeros((max_seq_len - len(group_inputs), input_size))
            padded_input = np.concatenate([input_array, padding])
            sku_inputs.append(padded_input)
        
        sku_inputs = np.array(sku_inputs)
        
        # Create attention mask
        mask = df_batch.groupby('GROUP_ID').SKU_ID.apply(
            lambda x: np.concatenate([
                np.ones(len(x)),
                np.zeros(max_seq_len - len(x))
            ])
        )
        
        # Convert to tensors
        mask_tensor = torch.tensor(mask.to_list(), dtype=torch.float).to(device)
        inputs_tensor = torch.tensor(sku_inputs, dtype=torch.float).to(device)
        
        # Get SKU counts
        sku_counts = df_batch.groupby('GROUP_ID').sku_count.mean().to_list()
        
        return {
            'df': df_batch,
            'input': inputs_tensor,
            'mask': mask_tensor,
            'poid': batch_groups,
            'sku_count': sku_counts
        }

