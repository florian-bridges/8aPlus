import os
import numpy as np
import wandb
import torch

from src.test import test

MODEL_OUTPUT_PATH = "output"

def train(dataloader, model, loss_fn, optimizer, epochs=3, device="cuda"):

    size = len(dataloader.dataset)
    model.train()

    max_val_acc = 0

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


        train_acc = train_acc.cpu().detach().numpy()/dp_count
        train_loss = train_loss.cpu().detach().numpy()/dp_count        
        validation_loss, validation_acc = test(dataloader, model, loss_fn, device)
        log_training_progress(train_acc, train_loss, validation_acc, validation_loss, epoch)
        #log_model(f"model_epoch{epoch}.pt", model)
        if  validation_acc > max_val_acc:
            log_model(f"best_model.pt", model)
            max_val_acc = validation_acc

def create_dir(PATH):
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    

def log_model(model_name, model):
    create_dir(os.path.join(MODEL_OUTPUT_PATH, wandb.run.name))
    torch.save(model.state_dict(), os.path.join(MODEL_OUTPUT_PATH, wandb.run.name, model_name))
    

def log_training_progress(train_acc, train_loss, validation_acc, validation_loss, epoch):
    print(f"train acc:\t {train_acc}, train loss:\t {train_loss}, test acc:\t {validation_acc}, test loss:\t {validation_loss}")
    wandb.log({
        "train_accuracy": train_acc,
        "train_loss": train_loss,
        "validation_accuracy": validation_acc,
        "validation_loss": validation_loss,
        "epoch": epoch,
    })