import numpy as np
import wandb

from src.test import test

def train(dataloader, model, loss_fn, optimizer, epochs=3, device="cuda"):
    size = len(dataloader.dataset)
    model.train()

    for epoch in range(epochs):
        print("Epoch: ", epoch)
        train_acc = 0
        train_loss = 0
        dp_count = 0

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            train_loss += loss.float().sum()
            train_acc += ((18 * pred).round() == (18 * y).round()).sum()
            dp_count += pred.size(dim=0)
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        train_acc = train_acc.cpu().detach().numpy()/dp_count
        train_loss = train_loss.cpu().detach().numpy()/dp_count
        
        validation_loss, validation_acc = test(dataloader, model, loss_fn, device)
        print(f"train acc:\t {train_acc}, train loss:\t {train_loss}, test acc:\t {validation_acc}, test loss:\t {validation_loss}")
        wandb.log({
            "train_accuracy": train_acc,
            "train_loss": train_loss,
            "validation_accuracy": validation_acc,
            "validation_loss": validation_loss,
            "epoch": epoch,
        })