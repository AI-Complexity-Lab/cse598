import logging
import os
from pathlib import Path
from typing import Union, List

from torch.utils.data import DataLoader

from .pt_dataset import PTDataset
from .web_utils import download_from_zenodo_record

from neuralop.utils import get_project_root

logger = logging.Logger(logging.root.level)

class DarcyDataset(PTDataset):
    """
    DarcyDataset stores data generated according to Darcy's Law.
    Input is a coefficient function and outputs describe flow. 
    """
    def __init__(self,
                 root_dir: Union[Path, str],
                 n_train: int,
                 n_tests: List[int],
                 batch_size: int,
                 test_batch_sizes: List[int],
                 train_resolution: int,
                 test_resolutions: int=[16,32],
                 grid_boundaries: List[int]=[[0,1],[0,1]],
                 positional_encoding: bool=True,
                 encode_input: bool=False, 
                 encode_output: bool=True, 
                 encoding="channel-wise",
                 channel_dim=1,
                 subsampling_rate=None,
                 download: bool=True):
        
        """
        Darcy Flow dataset
        Data source: https://zenodo.org/records/10994262
        """
        # convert root dir to Path
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        if not root_dir.exists():
            root_dir.mkdir(parents=True)

        # Zenodo record ID for Darcy-Flow dataset
        zenodo_record_id = "10994262"

        # List of resolutions needed for dataset object
        resolutions = set(test_resolutions + [train_resolution])

        # We store data at these resolutions on the Zenodo archive
        available_resolutions = [16, 32, 64, 128, 421]
        for res in resolutions:
            assert res in available_resolutions, f"Error: resolution {res} not available"

        # download darcy data from zenodo archive if passed
        if download:
            files_to_download = []
            already_downloaded_files = [x for x in root_dir.iterdir()]
            for res in resolutions:
                if f"darcy_train_{res}.pt" not in already_downloaded_files or \
                f"darcy_test_{res}.pt" not in already_downloaded_files:    
                    files_to_download.append(f"darcy_{res}.tgz")
            download_from_zenodo_record(record_id=zenodo_record_id,
                                        root=root_dir,
                                        files_to_download=files_to_download)
            
        # once downloaded/if files already exist, init PTDataset
        super().__init__(root_dir=root_dir,
                       dataset_name="darcy",
                       n_train=n_train,
                       n_tests=n_tests,
                       batch_size=batch_size,
                       test_batch_sizes=test_batch_sizes,
                       train_resolution=train_resolution,
                       test_resolutions=test_resolutions,
                       grid_boundaries=grid_boundaries,
                       positional_encoding=positional_encoding,
                       encode_input=encode_input,
                       encode_output=encode_output,
                       encoding=encoding,
                       channel_dim=channel_dim,
                       input_subsampling_rate=subsampling_rate,
                       output_subsampling_rate=subsampling_rate)
        
# legacy Small Darcy Flow example
example_data_root = get_project_root() / "neuralop/data/datasets/data"
def load_darcy_flow_small(n_train,
    n_tests,
    batch_size,
    test_batch_sizes,
    data_root = example_data_root,
    test_resolutions=[16, 32],
    grid_boundaries=[[0, 1], [0, 1]],
    positional_encoding=True,
    encode_input=False,
    encode_output=True,
    encoding="channel-wise",
    channel_dim=1,):

    dataset = DarcyDataset(root_dir = data_root,
                           n_train=n_train,
                           n_tests=n_tests,
                           batch_size=batch_size,
                           test_batch_sizes=test_batch_sizes,
                           train_resolution=16,
                           test_resolutions=test_resolutions,
                           grid_boundaries=grid_boundaries,
                           positional_encoding=positional_encoding,
                           encode_input=encode_input,
                           encode_output=encode_output,
                           encoding=encoding,
                           channel_dim=channel_dim,
                           download=False)
    
    # return dataloaders for backwards compat
    train_loader = DataLoader(dataset.train_db,
                              batch_size=batch_size,
                              num_workers=0,
                              pin_memory=True,
                              persistent_workers=False,)
    
    test_loaders = {}
    for res,test_bsize in zip(test_resolutions, test_batch_sizes):
        test_loaders[res] = DataLoader(dataset.test_dbs[res],
                                       batch_size=test_bsize,
                                       shuffle=False,
                                       num_workers=0,
                                       pin_memory=True,
                                       persistent_workers=False,)
    
    return train_loader, test_loaders, dataset.data_processor
    
# legacy pt Darcy Flow loader
def load_darcy_pt(n_train,
                  n_tests,
                  batch_size,
                  test_batch_sizes,
                  data_root = "./neuralop/data/datasets/data",
                  train_resolution=16,
                  test_resolutions=[16, 32],
                  grid_boundaries=[[0, 1], [0, 1]],
                  positional_encoding=True,
                  encode_input=False,
                  encode_output=True,
                  encoding="channel-wise",
                  channel_dim=1,):

    dataset = DarcyDataset(root_dir = data_root,
                           n_train=n_train,
                           n_tests=n_tests,
                           batch_size=batch_size,
                           test_batch_sizes=test_batch_sizes,
                           train_resolution=train_resolution,
                           test_resolutions=test_resolutions,
                           grid_boundaries=grid_boundaries,
                           positional_encoding=positional_encoding,
                           encode_input=encode_input,
                           encode_output=encode_output,
                           encoding=encoding,
                           channel_dim=channel_dim,
                           download=False)
    
    # return dataloaders for backwards compat
    train_loader = DataLoader(dataset.train_db,
                              batch_size=batch_size,
                              num_workers=0,
                              pin_memory=True,
                              persistent_workers=False,)
    
    test_loaders = {}
    for res,test_bsize in zip(test_resolutions, test_batch_sizes):
        test_loaders[res] = DataLoader(dataset.test_dbs[res],
                                       batch_size=test_bsize,
                                       shuffle=False,
                                       num_workers=0,
                                       pin_memory=True,
                                       persistent_workers=False,)
    
    return train_loader, test_loaders, dataset.data_processor