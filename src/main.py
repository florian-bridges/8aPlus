
import wandb
import torch
from torch import nn

from src.model import NeuralNetwork
from src.data_loading import BoulderDataLoader
from src.train import train

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

## init wandb
run = wandb.init(
        project="8aPlus",
    )

boudler_dl = BoulderDataLoader(batch_size=4)

model = NeuralNetwork().to(device)
print(model)
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=3 * 1e-2)
train(boudler_dl.get("train"), model, loss_fn, optimizer, epochs=1000)
