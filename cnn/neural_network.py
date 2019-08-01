import torch
import numpy as np
import torch.nn as nn
from algorithms.algo_utils import load_object
from colorama import Fore
from patch.patch_aggregator import PatchAggregator
from torch.utils.data.dataset import TensorDataset, Subset
from torch.utils.data.dataloader import DataLoader
from os.path import join, isfile, exists
from os import remove, mkdir


class HistoNet(nn.Module):

    def __init__(self):
        super(HistoNet, self).__init__()

        # CNN state

        self._currently_training = False
        self._trained = False

        # Loss of CNN
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(np.array([0.045, 0.955]), dtype=torch.float))

        # Image normalization over batch size, for each channel
        self.normalize = nn.BatchNorm2d(5)

        # Layers
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=4, kernel_size=3, stride=1, padding=1)
        # Wo = (Wi − kernel_size + 2 * Padding)/Stride + 1
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False,
                                     ceil_mode=False)

        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False,
                                     ceil_mode=False)

        self.FC1 = nn.Linear(in_features=4 * 4 * 4, out_features=256)
        self.relu3 = nn.ReLU()

        self.FC2 = nn.Linear(in_features=256, out_features=2)
        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.normalize(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        # x = self.conv2(x)
        # x = self.relu2(x)
        # x = self.maxpool2(x)
        x = x.view(-1, x[0].numel())

        x = self.FC1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.FC2(x)
        x = self.softmax(x)
        return x

    def train_nn(self,
                 dataloader,
                 lr=0.01,
                 n_epoch=5,
                 val_data=None):
        """
        :param dataloader: torch.dataloader object
        :param lr:
        :param n_epoch:
        :param val_data: must be a TensorDataset
        :return:
        """

        # Needed in case of hyperparameters tuning
        self.apply(self.weight_reset)
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        n_iter = len(dataloader)
        self._currently_training = True
        accuracy = []
        for epoch in range(n_epoch):
            # This array is only to generate acc performance
            y_train_label = []

            y_hat_train_label = []
            for m, (local_batch, local_label) in enumerate(dataloader):
                optimizer.zero_grad()
                output = self(local_batch)
                loss = self.criterion(output, local_label)
                loss.backward()
                optimizer.step()

                p_train_label_batch_numpy = output.detach().numpy()
                y_hat_train_label.extend(list(np.argmax(p_train_label_batch_numpy, axis=1)))
                y_train_label.extend(list(local_label.detach().numpy()))

                if m % np.floor(n_iter / 15.0).astype('int') == 0:
                    batch_accuracy = np.divide(np.sum(np.argmax(p_train_label_batch_numpy, axis=1) == local_label.detach().numpy()),
                                               np.float(dataloader.batch_size))
                    sensitivity = np.divide(np.sum((np.array(y_hat_train_label) == np.array(y_train_label)) * (np.array(y_train_label) == 1)),
                                            np.sum(np.array(y_train_label) == 1))
                    specificity = np.divide(np.sum((np.array(y_hat_train_label) == np.array(y_train_label)) * (np.array(y_train_label) == 0)),
                                            np.sum(np.array(y_train_label) == 0))
                    balanced_accuracy = (sensitivity + specificity) / 2
                    running_accuracy = np.divide(np.sum(np.array(y_hat_train_label) == np.array(y_train_label)),
                                                 np.float(len(y_hat_train_label)))
                    print('Epoch [{}/{}], Step [{}/{}], Batch Loss: {:.4f}, Batch Accuracy: {:.2f}%, Running Accuracy: {:.2f}%, Balanced Accuracy: {:.2f}%'
                          .format(epoch + 1, n_epoch, m + 1, n_iter, loss.item(), 100 * batch_accuracy, 100 * running_accuracy, 100 * balanced_accuracy))
            print('Results for epoch number ' + str(epoch + 1) + ':')
            if val_data is not None:
                accuracy.append(self.test(val_data.tensors[0], val_data.tensors[1]))
            else:
                accuracy.append(-1)
        print(Fore.GREEN + '*** Neural network is trained ***'.upper() + Fore.RESET)
        self._trained = True
        self._currently_training = False
        return np.array(accuracy)

    @staticmethod
    def weight_reset(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

    def test(self, test_images, test_labels):
        """
        :param test_images:
        :param test_labels:
        :return:
        """
        # Check that that the CNN is trained using attribute
        if not self._trained and not self._currently_training:
            raise RuntimeError('CNN is not trained. Train it with CNN.train()')
        if not (isinstance(test_images, torch.Tensor) & isinstance(test_labels, torch.Tensor)):
            raise ValueError('test data and label must be provided as torch.Tensor objects')
        if not test_images.shape[0] == test_labels.shape[0]:
            raise ValueError('Test set contains ' + str(test_images.shape[0]) + ' image(s) but has ' + str(test_labels.shape[0]) + ' label(s)')
        with torch.no_grad():
            probability = self(test_images)

        # Need to calculate balanced accuracy
        probability_numpy = probability.detach().numpy()
        test_labels_numpy = test_labels.detach().numpy()
        y_hat = np.argmax(probability_numpy, axis=1)
        accuracy = np.divide(np.sum(y_hat == test_labels_numpy),
                             np.float(test_labels_numpy.shape[0]))
        sensitivity = np.divide(np.sum((y_hat == test_labels_numpy) * (test_labels_numpy == 1)),
                                np.sum(test_labels_numpy == 1))
        specificity = np.divide(np.sum((y_hat == test_labels_numpy) * (test_labels_numpy == 0)),
                                np.sum(test_labels_numpy == 0))
        balanced_accuracy = (sensitivity + specificity) / 2

        # Cohen's kappa measures how much better the classifier is compared with guessing with the target distribution
        p_both_dnf = (np.sum(y_hat == 1) / y_hat.size) * (np.sum(test_labels_numpy == 1) / test_labels_numpy.size)
        p_both_no_dnf = (np.sum(y_hat == 0) / y_hat.size) * (np.sum(test_labels_numpy == 0) / test_labels_numpy.size)
        p_random_agreement = p_both_dnf + p_both_no_dnf
        kappa_cohen = (accuracy - p_random_agreement) / (1 - p_random_agreement)

        if balanced_accuracy != balanced_accuracy:
            print('** Nan problem **')
            print('acc ' + str(accuracy))
            print('balanced acc ' + str(balanced_accuracy))
            print('sensitivity  ' + str(sensitivity))
            print('specificity  ' + str(specificity))
            print('np.sum(test_labels_numpy == 1)) ' + str(np.sum(test_labels_numpy == 1)))
            print('np.sum(test_labels_numpy == 0))' + str(np.sum(test_labels_numpy == 0)))

        print(Fore.YELLOW)
        print('\tAccuracy of model on validation set is {:.2f}%'.format(100 * accuracy))
        print('\tBalanced accuracy of model on test set is {:.2f}%'.format(100 * balanced_accuracy))
        print('\tSensitivity of model on test set is {:.2f}%'.format(100 * sensitivity))
        print('\tSpecificity of model on test set is {:.2f}%'.format(100 * specificity))
        print('\tCohen\'s Kappa : {:.3f}'.format(kappa_cohen))
        print(Fore.RESET)
        return balanced_accuracy


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

    #hyperparameters = histo_net_cnn.find_hyper_parameter(k_fold=5,
    #                                                     n_epochs=3,
    #                                                     data=dataset,
    #                                                     output_directory='/Users/arnaud.marcoux/histo_mri/pickled_data/cnn/hyperparameters')

    # dtype Long is necessary for labels
    histo_net_cnn.train_nn(dataloader,
                           lr=0.01,
                           n_epoch=5)
