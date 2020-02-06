import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import accuracy_score

from utils import AverageMeter
from models import BaselineCNN


MODEL_WEIGHTS_PATH = 'weights/baseline_weigths.pt'

def str2numpy(string):
    """
        Convert the string representation of the image to numpy
    """
    return np.array(string.split(' ')).astype(np.float32).reshape(48, 48) / 255.0


def df2dataloader(df):
    """ 
      Create a pytorch dataloader from a pandas dataframe containing a column
      'pixel' of numpy array representing the image. The label must be in the 
      'emotion' column
    """
    X = np.stack(df.numpy.tolist())
    Y = np.array(df.emotion)
    X = torch.from_numpy(X).unsqueeze(1)
    Y = torch.from_numpy(Y)
    dataset = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    return dataloader


if __name__ == "__main__":
    # Check running device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if "cuda" in str(device):
        print(f"Training runing on {torch.cuda.get_device_name()}")
    else:
        print(f"Training on {device}")

    # Loading data
    df = pd.read_csv('data/fer2013.csv')
    df['numpy'] = df['pixels'].apply(str2numpy)
    train_dataloader = df2dataloader(df[df['Usage'] == 'Training'])
    val_dataloader = df2dataloader(df[df['Usage']  == 'PublicTest'])
    test_dataloader = df2dataloader(df[df['Usage']  == 'PrivateTest'])
    
    
    # define the model
    model = BaselineCNN()
    model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Training loop
    best_loss = float('Infinity')
    train_losses = []
    test_losses = []
    losses = AverageMeter('Loss', ':.4e')
    accuracy = AverageMeter('Acc', ':.4e')
    max_epochs = 500
    
    for epoch in range(max_epochs):  # loop over the dataset multiple times
        running_loss = 0.0

        # Traning phase
        losses.reset()
        model.train()
        pbar = tqdm(total=len(train_dataloader))
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), labels.shape[0])
            pbar.set_description(str(losses))
            pbar.update(1)

        train_losses.append(losses.avg)
        pbar.close()

        losses.reset()
        accuracy.reset()
        print("")
        # Testing phase
        for i, data in enumerate(test_dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            model.eval()
            with torch.no_grad():
                outputs = model(inputs)
            loss = criterion(outputs, labels)
            pred = nn.Softmax(dim=1)(outputs).argmax(1)
            losses.update(loss.item(), labels.shape[0])
            acc = accuracy_score(labels.cpu().numpy(), pred.cpu().numpy())
            accuracy.update(acc, labels.shape[0])
        print(f"On test set {str(losses)} and {str(accuracy)}")
        test_losses.append(losses.avg)

        if losses.avg < best_loss:
            print(f"Best loss improved from {best_loss} to {losses.avg}, saving model")
            torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
            best_loss = losses.avg
    
    plt.plot(train_losses, label="Training loss")
    plt.plot(test_losses, label="Validation loss")
    plt.legend()
    plt.imsave('outputs/training_loss.png')
    
    print('Finished Training')
