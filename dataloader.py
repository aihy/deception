import os
import pandas as pd
import numpy as np

import torch.utils.data as data
from PIL import Image


class XixiDataset(data.Dataset):
    def __init__(self, csvpath, transform=None):
        self.data = pd.read_csv(csvpath)
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data.iloc[index, 0]))
        if self.transform is not None:
            img = self.transform(img)

        if self.data.iloc[index, 1] == 0:  # deception
            label = 0
        else:
            label = 1
        target = np.asarray(label, dtype=np.int64)

        return img, target

    def __len__(self):
        return len(self.data)
