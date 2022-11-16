import torch

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss

    def __call__(
        self, current_valid_loss,
        epoch, model, optimizer, criterion, path
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
            }, f'{path}/best_model_1d - Copy.pth')


def save_model(epoch, model, optimizer, criterion, fold, path):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving latest model...")
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                'fold': fold,
                }, f'{path}/model_1d - Copy.pth')


def categorical_accuracy(preds, y):

    # Returns accuracy per batch, i.e. if you get 8/10 right, this returns
    # 0.8, NOT 8

    top_pred = preds.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc



def train(model, dataloader, optimizer, criterion, device):

    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in dataloader:
        tokens = batch['text'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        predictions = model(tokens)
        loss = criterion(predictions, labels)
        acc = categorical_accuracy(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            tokens = batch['text'].to(device)
            labels = batch['labels'].to(device)
            predictions = model(tokens)
            loss = criterion(predictions, labels)
            acc = categorical_accuracy(predictions, labels)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

