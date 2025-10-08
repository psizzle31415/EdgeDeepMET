"""
    PyTorch specification for the hit graph dataset, extended with edge features.
"""

# System imports
import os
import glob
import os.path as osp

# External imports
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import random_split
from torch_geometric.data import Data, Dataset
from torch_geometric.data import DataLoader
from torch_cluster import knn_graph
import torch_geometric.transforms as T


class METDataset(Dataset):
    """PyTorch geometric dataset from processed hit information."""

    def __init__(self, root):
        super(METDataset, self).__init__(root)

    def download(self):
        # Download from remote if needed
        pass

    @property
    def raw_file_names(self):
        if not hasattr(self, 'input_files'):
            self.input_files = sorted(glob.glob(osp.join(self.raw_dir, '*.npz')))
        return [osp.basename(f) for f in self.input_files]

    @property
    def existing_pt_names(self):
        if not hasattr(self, 'pt_files'):
            self.pt_files = sorted(glob.glob(osp.join(self.processed_dir, '*file*slice*nevent*pt')))
        return [osp.basename(f) for f in self.pt_files]

    @property
    def processed_file_names(self):
        if not hasattr(self, 'processed_files'):
            proc_names = [idx for idx in self.existing_pt_names]
            self.processed_files = [osp.join(self.processed_dir, name) for name in proc_names]
        return self.processed_files

    def __len__(self):
        return len(self.processed_file_names)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(self.processed_files[idx])
        return data

    def process(self):
        """
        Converts the .npz files into PyTorch Geometric Data objects and saves them to disk.
        Adds kNN-based edges and 5D edge features per event.
        """
        k = 10  # number of nearest neighbors for graph connectivity

        for idx, raw_path in enumerate(tqdm(self.raw_paths)):
            npzfile = np.load(raw_path, allow_pickle=True)

            for ievt in range(np.shape(npzfile['x'])[1]):
                # Load event inputs
                inputs = np.array(npzfile['x'][:, ievt, :]).astype(np.float32).T

                # Build node features:
                # L1MET variables: pt, eta, phi, puppiWeight, pdgId, charge
                # We want: pt, px, py, eta, phi, puppiWeight, pdgId, charge
                x = inputs[:, 1:6]
                x = np.insert(x, 0, inputs[:, 0], axis=1)
                x = np.insert(x, 1, inputs[:, 0] * np.cos(inputs[:, 2]), axis=1)
                x = np.insert(x, 2, inputs[:, 0] * np.sin(inputs[:, 2]), axis=1)
                x = x[x[:, 6] != -999]
                x = x[x[:, 7] != -999]
                x = x[np.abs(x[:, 0]) <= 500.]

                x = np.nan_to_num(x)
                x = np.clip(x, -5000., 5000.)
                assert not np.any(np.isnan(x))

                x_torch = torch.from_numpy(x).float()
                N = x_torch.size(0)
                if N < 2:
                    continue

                # Build kNN graph using eta/phi as coordinates
                coords = x_torch[:, 3:5]  # (eta, phi)
                edge_index = knn_graph(coords, k=k, loop=False)

                # Build edge features
                src, dst = edge_index
                eta_i, phi_i = x_torch[src, 3], x_torch[src, 4] #check indices
                eta_j, phi_j = x_torch[dst, 3], x_torch[dst, 4]
                pt_i, pt_j = x_torch[src, 0], x_torch[dst, 0]
                ch_i, ch_j = x_torch[src, 7], x_torch[dst, 7]

                d_eta = eta_j - eta_i
                d_phi = phi_j - phi_i
                d_phi = torch.remainder(d_phi + np.pi, 2 * np.pi) - np.pi  # wrap to [-pi, pi]
                dR = torch.sqrt(d_eta ** 2 + d_phi ** 2 + 1e-8) #check 1e-8
                pt_ratio = pt_j / (pt_i + 1e-8)
                ch_diff = ch_j - ch_i

                edge_attr = torch.stack([d_eta, d_phi, dR, pt_ratio, ch_diff], dim=1) #need to process data
                #use jupyter notebook, import the functions, run line by line, call the functions with the data
                #we have, understand what the functions do. see if we can get the features from data

                # Target
                y = torch.from_numpy(np.array(npzfile['y'][ievt, :]).astype(np.float32)[None])

                # Save graph
                data = Data(
                    x=x_torch,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y
                )
                torch.save(
                    data,
                    osp.join(
                        self.processed_dir,
                        (raw_path.replace('.npz', f'_{ievt}.pt')).split('/')[-1]
                    )
                )


def fetch_dataloader(data_dir, batch_size, validation_split):
    """
    Creates train/test dataloaders for the MET dataset with edge features.
    """
    transform = T.Cartesian(cat=False)
    dataset = METDataset(data_dir)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    print(f"Dataset split: {split} for validation")

    random_seed = 42
    torch.manual_seed(random_seed)
    train_subset, val_subset = torch.utils.data.random_split(
        dataset,
        [dataset_size - split, split],
    )
    print('Train/val sizes:', len(train_subset), len(val_subset))

    dataloaders = {
        'train': DataLoader(train_subset, batch_size=batch_size, num_workers=4,
                            pin_memory=True, shuffle=True),
        'test': DataLoader(val_subset, batch_size=batch_size, num_workers=4,
                           pin_memory=True, shuffle=False)
    }
    return dataloaders
