import json
import typing
from pathlib import Path

import torch
from torch.utils.data import Dataset

from util.globals import *

REMOTE_ROOT = f"{REMOTE_ROOT_URL}/data/dsets"

class MQuAKEDataset_CF_3k_A(Dataset):
    """
    Dataset class for loading MQuAKE-CF data.
    """
    def __init__(self, data_dir: str, size: typing.Optional[int] = None, *args, **kwargs):
        data_dir = Path(data_dir)
        mquake_loc = data_dir / "MQuAKE-CF-3k-A.json"
        if not mquake_loc.exists():
            remote_url = f"{REMOTE_ROOT}/MQuAKE-CF-3k.json"
            print(f"{mquake_loc} does not exist. Downloading from {remote_url}")
            data_dir.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(remote_url, mquake_loc)
        
        with open(mquake_loc, "r") as f:
            self.data = json.load(f)
        if size is not None:
            self.data = self.data[:size]
        
        print(f"Loaded MQuAKE-CF dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
