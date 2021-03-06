from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import heapq

import os

'''
This code is adapted from two sources:
(i) The official PyTorch MNIST example (https://github.com/pytorch/examples/blob/master/mnist/main.py)
(ii) Starter code from Yisong Yue's CS 155 Course (http://www.yisongyue.com/courses/cs155/2020_winter/)
'''

class fcNet(nn.Module):
    '''
    Design your model with fully connected layers (convolutional layers are not
    allowed here). Initial model is designed to have a poor performance. These
    are the sample units you can try:
        Linear, Dropout, activation layers (ReLU, softmax)
    '''
    def __init__(self):
        # Define the units that you will use in your model
        # Note that this has nothing to do with the order in which operations
        # are applied - that is defined in the forward function below.
        super(fcNet, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=20)
        self.fc2 = nn.Linear(20, 10)
        self.dropout1 = nn.Dropout(p=0.5)

    def forward(self, x):
        # Define the sequence of operations your model will apply to an input x
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.relu(x)

        output = F.log_softmax(x, dim=1)
        return output


class ConvNet(nn.Module):
    '''
    Design your model with convolutional layers.
    '''
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(8, 8, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(200, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


class Net(nn.Module):
    '''
    Build the best MNIST classifier.
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(8, 8, 3, 1)
        self.conv3 = nn.Conv2d(8, 16, 3, 1)
        self.conv4 = nn.Conv2d(16, 16, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 10)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(16)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x_emb = self.fc2(x)

        output = F.log_softmax(x_emb, dim=1)
        return output, x_emb


def train(args, model, device, train_loader, optimizer, epoch):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    train_loss = 0.
    train_num = 0.
    model.train()   # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Clear the gradient
        output, emb = model(data)                # Make predictions
        loss = F.nll_loss(output, target)   # Compute loss
        train_loss += F.nll_loss(output, target, reduction='sum').item()
        train_num += len(data)
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))
    train_loss /= train_num
    return train_loss


def test(model, device, test_loader, get_extra=False):
    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0
    target_lst = []
    pred_lst = []
    embeddings = []
    targets = []
    datas = []
    with torch.no_grad():   # For the inference step, gradient is not computed
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output, emb = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_num += len(data)
            # Make list of predictions and targets for confusion matrix.
            # Also get embeddings
            if get_extra:
                target_lst += target.flatten().tolist()
                pred_lst += pred.flatten().tolist()
                for i in range(np.shape(emb)[0]):
                    embeddings.append(emb[i].tolist())
                    targets.append(target[i])
                    datas.append(data[i,0])

    test_loss /= test_num

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, test_num,
        100. * correct / test_num))
    test_acc = correct / test_num
    return test_loss, test_acc, target_lst, pred_lst, np.asarray(embeddings), targets, datas

# Plots training and val loss as a function of the epoch.
def plot_train_test_loss_epoch(train_loss, test_loss, num_epochs):
    fig, ax = plt.subplots()
    epochs = list(range(1,num_epochs+1))
    ax.plot(epochs, train_loss, color='r', marker='.', label='Train')
    ax.plot(epochs, test_loss, color='b', marker='.', label='Validation')
    ax.set(xlabel='Epoch', ylabel='NLL Loss', yscale='log', title='Loss By Epoch')
    ax.legend(title='Type of Loss')
    plt.show()
    return


# Get the training and test loss and accuracies.
def get_acc_loss(sizes, device, train_loader, test_loader):
    train_loss_lst = []
    train_acc_lst = []
    test_loss_lst = []
    test_acc_lst = []
    for size in sizes:
        model = Net().to(device)
        model.load_state_dict(torch.load('mnist_model_' + str(size) + '.pt'))
        train_loss, train_acc, target_lst, pred_lst, emb, targets, datas = test(model, device, train_loader)
        test_loss, test_acc, target_lst, pred_lst, emb, targets, datas = test(model, device, test_loader)
        train_loss_lst.append(train_loss)
        train_acc_lst.append(train_acc)
        test_loss_lst.append(test_loss)
        test_acc_lst.append(test_acc)
    return train_loss_lst, train_acc_lst, test_loss_lst, test_acc_lst


# Plot the training and test error as a function of the number of training examples.
def plot_train_test_loss_subset(sizes, train_loss_lst, test_loss_lst):
    # Only trained on 0.85 on dataset.
    sizes_part = [int(0.85 * size) for size in sizes]
    fig, ax = plt.subplots()
    ax.plot(sizes_part, train_loss_lst, color='r', marker='.', label='Train')
    ax.plot(sizes_part, test_loss_lst, color='b', marker='.', label='Test')
    ax.set(xlabel='Number of Training Examples', ylabel='NLL Loss', title='Loss vs. Number of Training Examples')
    ax.set(xscale='log', yscale='log')
    ax.set_xticks(sizes_part)
    ax.set_xticklabels(sizes_part)
    ax.legend(title='Type of Loss')
    plt.savefig('error_vs_num_ex.png')
    plt.show()
    return


# Plots 9 examples from the test set where the classifier made a mistake.
def plot_mistakes(model, device, test_loader):
    model.eval()    # Set the model to inference mode
    lim_mistakes = 9
    mistakes = []
    fig, axes = plt.subplots(3, 3)
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, emb = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            for i in range(len(pred)):
                if pred[i,0] != target[i]:
                    mistakes.append(data[i,0])
                    ax = axes[(len(mistakes)-1)//3, (len(mistakes)-1)%3]
                    ax.imshow(data[i,0], cmap='gray')
                    ax.set_title('Actual: {} Pred: {}'.format(target[i], pred[i,0]))
                    if len(mistakes) >= lim_mistakes:
                        plt.tight_layout()
                        plt.savefig('mistakes.png')
                        plt.show()
                        return mistakes
    return


# Viualizes 8 of the learned kernels from the first layer of the network.
def plot_kernels(kernels):
    num_rows = 2
    num_cols = 4
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(1.5*num_cols,2*num_rows))
    for i in range(len(kernels)):
        kernel = kernels[i,0]
        ax = axes[i//num_cols, i%num_cols]
        # Normalize input
        ker = (kernel - torch.min(kernel)) / (torch.max(kernel) - torch.min(kernel))
        im = ax.imshow(ker, cmap='gray')
    fig.suptitle('First Layer Learned Kernels')
    plt.tight_layout()
    plt.savefig('learned_kernels.png')
    plt.show()
    return


# Generates a confusion matrix for the test set.
def plot_confusion_matrix(target_lst, pred_lst):
    labels = list(range(10))
    # Calculate confusion matrix.
    conf_mat = confusion_matrix(target_lst, pred_lst, labels=labels)

    # Plot.
    fig, axs = plt.subplots(1,1)
    axs.axis('tight')
    axs.axis('off')
    table = axs.table(cellText=conf_mat, rowLabels=labels, colLabels=labels, loc='center')
    plt.suptitle('Confusion Matrix (ground truth vs. predictions)')
    plt.savefig('confusion_mat.png')
    plt.show()
    return


# Visualizes the tSNE embedding.
def plot_embeddings(embeddings, targets):
    # do again with separate test function
    # separate by class list and thn
    colors = cm.rainbow(np.linspace(0, 1, 10))
    embeddings_2 = TSNE(n_components=2).fit_transform(embeddings)
    embeddings_2_classes = [[] for i in range(10)]
    # Sort out classes.
    for i in range(len(targets)):
        embeddings_2_classes[targets[i]].append(embeddings_2[i,:])
    embeddings_2_classes_arr = [np.asarray(class_arr) for class_arr in embeddings_2_classes]
    for j in range(10):
        plt.scatter(embeddings_2_classes_arr[j][:,0], embeddings_2_classes_arr[j][:,1],
                    color=colors[j], marker='.', alpha=0.5, label=j)
    plt.legend(title='Class')
    plt.title('Feature Vectors By Class')
    plt.savefig('tSNE_embedding.png')
    plt.show()
    return


# Finds feature vectors that are close in distance.
def find_similar_vectors(embeddings, datas):
    num_ims = 8
    fig, axes = plt.subplots(4, 9)
    for i in range(4):
        emb = embeddings[(i+1)*1000]
        embeddings_dist = np.linalg.norm(embeddings - emb, axis=1)
        closest_emb = heapq.nsmallest(num_ims, embeddings_dist)
        closest_emb_inds = []
        for c_emb in closest_emb:
            closest_emb_inds += list(np.argwhere(embeddings_dist == c_emb).flatten())
        # If there's more than one vector that has the same distance.
        # We can cut it off like this because nsmallest returns an ordered list.
        closest_emb_inds = closest_emb_inds[:num_ims]
        # Visualize I_0.
        ax = axes[i, 0]
        ax.imshow(datas[(i+1)*1000], cmap='gray')
        ax.set_title('I_0')
        # Visualize other 8 images with closest feature vectors.
        for k in range(8):
            ax = axes[i, k+1]
            ax.imshow(datas[closest_emb_inds[k]], cmap='gray')
            ax.set_title('I_{}'.format(k+1))
    plt.tight_layout()
    plt.savefig('sim_feature_vecs.png')
    plt.show()
    return


def main():
    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step', type=int, default=1, metavar='N',
                        help='number of epochs between learning rate reductions (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate your model on the official test set')
    parser.add_argument('--load-model', type=str,
                        help='model file path')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Evaluate on the official test set
    if args.evaluate:
        assert os.path.exists(args.load_model)

        # Set the test model
        model = Net().to(device)
        model.load_state_dict(torch.load(args.load_model))

        test_dataset = datasets.MNIST('../data', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        test_loss, test_acc, target_lst, pred_lst, emb, targets, datas = \
                                test(model, device, test_loader, get_extra=True)

        train_dataset = datasets.MNIST('../data', train=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

        sizes = [3750, 7500, 15000, 30000, 60000]
        train_loss_lst, train_acc_lst, test_loss_lst, test_acc_lst = \
                        get_acc_loss(sizes, device, train_loader, test_loader)
        plot_train_test_loss_subset(sizes, train_loss_lst, test_loss_lst)

        plot_mistakes(model, device, test_loader)

        plot_kernels(model.conv1.weight.data)

        plot_confusion_matrix(target_lst, pred_lst)

        plot_embeddings(emb, targets)
        find_similar_vectors(emb, datas)
        return

    # Pytorch has default MNIST dataloader which loads data at each iteration
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                transform=transforms.Compose([       # Data preprocessing
                    transforms.RandomApply([transforms.RandomResizedCrop(28,(0.8,1.))]),
                    transforms.RandomAffine(degrees=(-3,3),translate=(0.05,0.05)),
                    transforms.ToTensor(),           # Add data augmentation here
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))

    valid_dataset = datasets.MNIST('../data', train=True, download=True,
                transform=transforms.Compose([       # Data preprocessing
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))

    # You can assign indices for training/validation or use a random subset for
    # training by using SubsetRandomSampler. Right now the train and validation
    # sets are built from the same indices - this is bad! Change it so that
    # the training and validation sets are disjoint and have the correct relative sizes.
    indices_all = np.asarray(range(len(train_dataset)))
    np.random.shuffle(indices_all)
    # Decide amount of data to train on.
    data_frac = float(1)
    indices_all = indices_all[:int(data_frac * len(indices_all))]
    valid_per_class = np.ones(10) * int(0.15 * len(indices_all) * 0.1)

    subset_indices_train = []
    subset_indices_valid = []
    i = 0
    while np.any(valid_per_class > 0):
        num = train_dataset[indices_all[i]][1]
        if valid_per_class[num] > 0:
            subset_indices_valid.append(indices_all[i])
            valid_per_class[num] -= 1
        else:
            subset_indices_train.append(indices_all[i])
        i += 1
    subset_indices_train = subset_indices_train + list(indices_all[i:])
    assert(len(subset_indices_train) + len(subset_indices_valid) == len(indices_all))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=SubsetRandomSampler(subset_indices_train)
    )
    val_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.test_batch_size,
        sampler=SubsetRandomSampler(subset_indices_valid)
    )

    # Load your model [fcNet, ConvNet, Net]
    model = Net().to(device)

    # Try different optimzers here [Adam, SGD, RMSprop]
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Set your learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    train_loss_all = []
    val_loss_all = []
    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        val_loss, val_acc, target_lst, pred_lst, emb, targets, datas = test(model, device, val_loader)
        train_loss_all.append(train_loss)
        val_loss_all.append(val_loss)
        scheduler.step()    # learning rate scheduler

        # You may optionally save your model at each epoch here

    if args.save_model:
        torch.save(model.state_dict(), "mnist_model_" + str(len(indices_all)) + ".pt")

    # Plot training and val loss as a function of epoch.
    plot_train_test_loss_epoch(train_loss_all, val_loss_all, args.epochs)




if __name__ == '__main__':
    main()
