import torch
import numpy as np
import torch.nn as nn
from algo_utils import load_object
from patch_aggregator import PatchAggregator
from torch.utils.data.dataset import TensorDataset, Subset
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import StratifiedKFold



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
                 val_data=None):
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
                accuracy.append(self.test(val_data.tensors[0].numpy(), val_data.tensors[1].numpy()))
            else:
                accuracy.append(-1)
        print('*** Neural network is trained ***'.upper())
        self._trained = True
        self._currently_training = False
        return np.array(accuracy)

    @staticmethod
    def weight_reset(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

    def find_hyper_parameter(self, k_fold, n_epochs, data):

        # search parameter range
        lr_range = np.logspace(-3, -0.7, 5)
        batch_size_range = np.array([2 ** power for power in [3, 4, 5, 6]]).astype('int')
        accuracy_hyperparameters = np.zeros((k_fold,
                                             lr_range.size,
                                             batch_size_range.size,
                                             n_epochs)).astype('float')

        # Stratiefied k_fold ensures to keep the folds balanced
        skf = StratifiedKFold(k_fold, shuffle=True, random_state=0)

        # k_fold grid
        for idx_k, (train_idx, val_idx) in enumerate(skf.split(data.tensors[0], data.tensors[1])):
            # Those lines displays information of the repartition of dnf labels within each train / val t
            # print('fold ' + str(i) + ' % of DNF in test set: ' + str(np.sum(test_label) / test_label.shape[0]))
            # print('fold ' + str(i) + ' % of DNF in train set: ' + str(np.sum(train_label) / train_label.shape[0]))

            subset_train = Subset(data, train_idx)
            subset_val = Subset(data, val_idx)

            for idx_lr, lr in enumerate(lr_range):
                for idx_bs, batch_size in enumerate(batch_size_range):
                    # Already shuffled by stratifiedKsplit
                    current_params = {'batch_size': batch_size,
                                      'num_workers': 8}
                    dataloader_train = DataLoader(subset_train, **current_params)
                    # dataloader_val = DataLoader(subset_val, **current_params)
                    accuracy_hyperparameters[idx_k, idx_lr, idx_bs, :] = self.train_nn(dataloader_train,
                                                                                       lr=lr,
                                                                                       n_epoch=n_epochs,
                                                                                       val_data=subset_val)
        return accuracy_hyperparameters

    def test(self, test_images, test_labels):
        # Check that that the CNN is trained using attribute
        if not self._trained and not self._currently_training:
            raise RuntimeError('CNN is not trained. Train it with CNN.train()')
        if not (isinstance(test_images, torch.Tensor) & isinstance(test_labels, torch.Tensor)):
            raise ValueError('test data and label must be provided as torch.Tensor objects')
        if not test_images.shape[0] == test_labels.shape[0]:
            raise ValueError('Test set contains ' + str(test_images.shape[0]) + ' image(s) but has ' + str(
                test_labels.shape[0]) + ' label(s)')
        with torch.no_grad():
            probability = self(test_images)
        probability_numpy = probability.detach().numpy()
        y_hat = np.argmax(probability_numpy, axis=1)
        accuracy = np.divide(np.sum(y_hat == test_labels.detach().numpy()),
                             np.float(test_labels.shape[0]))
        print('Accuracy of model on test set is {:.2f}%'.format(100 * accuracy))
        return accuracy

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
    #dataloader = DataLoader(dataset, **params)

    hyperparameters = histo_net_cnn.find_hyper_parameter(k_fold=5, n_epochs=3, data=dataset)

    # dtype Long is necessary for labels
    #histo_net_cnn.train_nn(dataloader,
    #                       lr=0.01,
    #                       n_epoch=1)
