import torch

from src.model import NeuralNetwork
from src.globals import *


def load_model(path):
    model = NeuralNetwork()
    model.load_state_dict(torch.load(path))
    model.eval()

    return model


def gt_to_grade(gt):
    return [
        INV_GRADE_DICT[grade]
        for grade in (18 * gt).round().to("cpu").detach().numpy().flatten()
    ]
