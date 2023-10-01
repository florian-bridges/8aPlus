import os
import requests
import json
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.globals import DATA_PATH, GRADE_DICT


class BoulderDataLoader:
    def __init__(self, batch_size, split=[0.8, 0.1, 0.1], seed=1337):
        raw_data = self.load_raw_data()

        X = []
        y = []
        for key in raw_data.keys():
            grade = GRADE_DICT[raw_data[key]["meta_info"]["grade"]]
            boulder = raw_data[key]["boulder"]

            X.append(np.sum(boulder, axis=0))
            y.append([float(grade)])

        y = np.array(y)
        y = y / 18
        X = torch.Tensor(np.array(X).astype(float))
        y = torch.Tensor(np.array(y).astype(float))

        torch.manual_seed(seed)
        train_tensor, validation_tensor, test_tensor = random_split(
            TensorDataset(X, y), split
        )
        self.train_loader = DataLoader(
            train_tensor, batch_size=batch_size, shuffle=True
        )
        self.validation_loader = DataLoader(
            train_tensor, batch_size=batch_size, shuffle=True
        )
        self.test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=True)

    def load_raw_data(self):
        ids = [
            file_name.replace(".json", "")
            for file_name in os.listdir(DATA_PATH)
            if file_name.find(".json") != -1
        ]

        data_dict = {}
        for id in ids:
            with open(os.path.join(DATA_PATH, id + ".json"), "r") as f:
                js = json.load(f)

            boulder = np.load(os.path.join(DATA_PATH, id + ".npz"))["arr_0"]

            data_dict[id] = {
                "meta_info": js,
                "boulder": boulder,
            }
        return data_dict

    def get(self, split_type):
        if split_type == "train":
            return self.train_loader
        if split_type == "validation":
            return self.validation_loader
        if split_type == "test":
            return self.test_loader
