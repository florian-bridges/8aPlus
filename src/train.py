

def train(dataloader, model, loss_fn, optimizer, epochs=3, device="cuda"):
    size = len(dataloader.dataset)
    model.train()
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
    
            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)
    
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")