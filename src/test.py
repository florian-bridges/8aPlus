import torch


def test(dataloader, model, loss_fn, device="cuda"):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            test_acc += (
                ((18 * pred).round() == (18 * y).round()).type(torch.float).sum().item()
            )
    test_loss /= num_batches
    test_acc /= size
    return test_loss, test_acc

{"test":"test","test":"test","test":"test","test":"test","test":"test","test":"test","test":"test","test":"test","test":"test","test":"test","test":"test","test":"test","test":"test","test":"test","test":"test","test":"test","test":"test",}