import torch
import numpy as np
import torch.nn as nn
from algo_utils import load_object
from patch_aggregator import PatchAggregator
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader


class HistoNet(nn.Module):

    def __init__(self):
        super(HistoNet, self).__init__()

        # CNN state

        self._currently_training = False
        self._trained = False

        # Loss of CNN
        self.criterion = nn.CrossEntropyLoss()

        # Layers
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=4, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False,
                                     ceil_mode=False)

        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False,
                                     ceil_mode=False)

        self.FC1 = nn.Linear(in_features=8 * 8 * 8, out_features=128)
        self.relu3 = nn.ReLU()

        self.FC2 = nn.Linear(in_features=128, out_features=2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.view(-1, x[0].numel())
        x = self.FC1(x)
        x = self.relu3(x)
        x = self.FC2(x)
        x = self.softmax(x)
        return x

    def train_nn(self,
                 dataloader,
                 lr=0.01,
                 n_epoch=5,
                 test_im=None,
                 test_lab=None):
        """
        if train_images.shape[0] % batch_size != 0:
            print('Recognized ' + str(train_images.shape[0]) + ' images but batch size is ' + str(batch_size)
                  + '. Network needs number_of_images % batch_size == 0')
            samples_to_add = batch_size - train_images.shape[0] % batch_size
            train_images = torch.tensor(np.concatenate((train_images, train_images[:samples_to_add])), dtype=torch.float)
            train_labels = torch.tensor(np.concatenate((train_labels, train_labels[:samples_to_add])), dtype=torch.long)
            assert train_images.shape[0] % batch_size == 0, 'Something went wrong when concatenating train_images'
        """

        # Needed in case of hyperparameters tuning
        self.apply(self.weight_reset)
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        n_iter = len(dataloader)
        self._currently_training = True
        accuracy = []
        for epoch in range(n_epoch):
            # y_hat_train_label = np.empty(train_labels.shape)
            # y_hat_train_label[:] = np.nan
            y_hat_train_label = []
            for local_batch, local_label in dataloader:
                print(str(np.sum(local_label.detach().numpy())))
                optimizer.zero_grad()
                output = self(local_batch)
                loss = self.criterion(output, local_label)
                loss.backward()
                optimizer.step()

                p_train_label_batch_numpy = output.detach().numpy()
                #y_hat_train_label.extend(np.argmax(p_train_label_batch_numpy, axis=1))

                #batch_accuracy = np.divide(np.sum(np.argmax(p_train_label_batch_numpy, axis=1) == local_label.detach().numpy()), np.float(batch_size))
                #running_accuracy = np.divide(np.sum(y_hat_train_label[batch_size * epoch + m] == local_label.detach().numpy()), np.float((m + 1) * batch_size))
                #if m % np.floor(n_iter / 10.0).astype('int') == 0:
                #    print('Epoch [{}/{}], Step [{}/{}], Batch Loss: {:.4f}, Batch Accuracy: {:.2f}%, Running Accuracy: {:.2f}%'
                #          .format(epoch + 1, n_epoch, m + 1, n_iter, loss.item(), 100 * batch_accuracy,
                #                  100 * running_accuracy))
            print('Results for epoch number ' + str(epoch + 1) + ':')
            if test_im is not None and test_lab is not None:
                accuracy.append(self.test(test_im, test_lab))
            else:
                accuracy.append(-1)
        print('*** Neural network is trained ***'.upper())
        self._trained = True
        self._currently_training = False
        return accuracy

    @staticmethod
    def weight_reset(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()


if __name__ == '__main__':
    mri_patches = load_object('/Users/arnaud.marcoux/histo_mri/pickled_data/aggregator_test')

    histo_net_cnn = HistoNet()

    labs = mri_patches.all_labels
    labs[labs == 2] = 0

    # Parameters
    params = {'batch_size': 32,
              'shuffle': True,
              'num_workers': 8}

    dataset = TensorDataset(torch.from_numpy(mri_patches.all_patches),
                            torch.from_numpy(labs))
    dataloader = DataLoader(dataset, **params)

    # dtype Long is necessary for labels
    histo_net_cnn.train_nn(dataloader,
                           lr=0.01,
                           n_epoch=1)
