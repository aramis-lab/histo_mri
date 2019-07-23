import torch
import numpy as np
import torch.nn as nn
from algo_utils import load_object
from patch_aggregator import PatchAggregator
from torch.utils.data.dataset import TensorDataset, Subset
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import StratifiedKFold, KFold
from os.path import join, isfile, exists
from os import remove, mkdir


class HistoNet(nn.Module):

    def __init__(self):
        super(HistoNet, self).__init__()

        # CNN state

        self._currently_training = False
        self._trained = False

        # Loss of CNN
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(np.array([0.05, 0.95]), dtype=torch.float))

        # Image normalization over batch size, for each channel
        self.normalize = nn.BatchNorm2d(5)

        # Layers
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=4, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False,
                                     ceil_mode=False)

        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False,
                                     ceil_mode=False)

        self.FC1 = nn.Linear(in_features=8 * 8 * 4, out_features=128)
        self.relu3 = nn.ReLU()

        self.FC2 = nn.Linear(in_features=128, out_features=2)
        self.softmax = nn.Softmax(dim=1)

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
        x = self.FC2(x)
        x = self.softmax(x)
        return x

    def train_nn(self,
                 dataloader,
                 lr=0.01,
                 n_epoch=5,
                 val_data=None):
        """
        :param dataloader:
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
        print('*** Neural network is trained ***'.upper())
        self._trained = True
        self._currently_training = False
        return np.array(accuracy)

    @staticmethod
    def weight_reset(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

    @staticmethod
    def write_informations(n_epoch, lr_range, batch_size_range, train_set, val_set, in_mat, output_file):

        mat = np.load(in_mat)
        # Averaging along folds
        mat_avg = np.mean(mat, axis=0)
        argmax = np.unravel_index(np.argmax(mat_avg), mat_avg.shape)

        best_accuracy_fold = mat[:, argmax[0], argmax[1], :]
        idx_best_fold = np.unravel_index(np.argmax(best_accuracy_fold), best_accuracy_fold.shape)[0]

        if isfile(output_file):
            remove(output_file)

        with open(output_file, 'w') as file:
            file.write('n_epoch : ' + str(n_epoch))
            file.write('\nlr_range : ' + str(lr_range))
            file.write('\nbatch_size_range : ' + str(batch_size_range))
            file.write('\nmatrix file : ' + str(in_mat))
            file.write('\ntrain set : ' + str(train_set))
            file.write('\ntest set : ' + str(val_set))
            file.write('\n\n** Best model :\nlr : ' + str(lr_range[argmax[0]]))
            file.write('\nbatch size : ' + str(batch_size_range[argmax[1]]))
            file.write('\nepoch : ' + str(argmax[2] + 1))
            file.write('\nidx best fold : ' + str(idx_best_fold))
            file.write('\nBest balanced accuracy : ' + str(np.max(mat_avg)))

    def find_hyper_parameter(self, n_epochs, data, output_directory):
        """
        :param n_epochs:
        :param data:
        :param output_directory:
        :return:
        """

        if not exists(output_directory):
            mkdir(output_directory)

        # CV model
        k_fold = 4
        # search parameter range
        lr_range = np.logspace(-3, -2, 5)
        # batch_size_range = np.array([2 ** power for power in [4, 5, 6]]).astype('int')
        batch_size_range = np.array([32, 64])
        accuracy_hyperparameters = np.zeros((4,
                                             lr_range.size,
                                             batch_size_range.size,
                                             n_epochs)).astype('float')

        train_slices, val_slices = self.divide_population(4, 4)
        train_slices_name = []
        val_slices_name = []
        for train_slice, val_slice in zip(train_slices, val_slices):
            train_slices_name.append(['TG0' + str(n + 3) for n in train_slice[0]]
                                     + ['WT0' + str(n + 3) for n in train_slice[1]])
            val_slices_name.append(['TG0' + str(n + 3) for n in val_slice[0]]
                                   + ['WT0' + str(n + 3) for n in val_slice[1]])
        train_idx = []
        val_idx = []
        for k, (train_slice_name, val_slice_name) in enumerate(zip(train_slices_name, val_slices_name)):
            train_idx.append([i for i in range(len(data.mouse_name)) if data.mouse_name[i] in train_slice_name])
            val_idx.append([i for i in range(len(data.mouse_name)) if data.mouse_name[i] in val_slice_name])

        dataset = TensorDataset(torch.from_numpy(data.all_patches),
                                torch.from_numpy(data.all_labels))
        # k_fold grid
        for idx_k in range(k_fold):
            # Those lines displays information of the repartition of dnf labels within each train / val t

            subset_train = Subset(dataset, train_idx[idx_k])

            for idx_lr, lr in enumerate(lr_range):
                for idx_bs, batch_size in enumerate(batch_size_range):
                    current_params = {'batch_size': int(batch_size),
                                      'shuffle': True,
                                      'num_workers': 8}
                    dataloader_train = DataLoader(subset_train, **current_params)
                    dataset_val = TensorDataset(torch.from_numpy(data.all_patches[val_idx[idx_k]]),
                                                torch.from_numpy(data.all_labels[val_idx[idx_k]]))
                    accuracy_hyperparameters[idx_k, idx_lr, idx_bs, :] = self.train_nn(dataloader_train,
                                                                                       lr=lr,
                                                                                       n_epoch=n_epochs,
                                                                                       val_data=dataset_val)
                    np.save(join(output_directory, 'hyperparameter_matrix.npy'), accuracy_hyperparameters)
                    self.write_informations(n_epochs, lr_range, batch_size_range,
                                            train_slices_name,
                                            val_slices_name,
                                            join(output_directory, 'hyperparameter_matrix.npy'),
                                            join(output_directory, 'results_cnn.txt'))
        return accuracy_hyperparameters

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
        y_hat = np.argmax(probability_numpy, axis=1)
        accuracy = np.divide(np.sum(y_hat == test_labels.detach().numpy()),
                             np.float(test_labels.shape[0]))
        sensitivity = np.divide(np.sum((np.array(y_hat) == np.array(test_labels)) * (np.array(test_labels) == 1)),
                                np.sum(np.array(test_labels) == 1))
        specificity = np.divide(np.sum((np.array(y_hat) == np.array(test_labels)) * (np.array(test_labels) == 0)),
                                np.sum(np.array(test_labels) == 0))
        balanced_accuracy = (sensitivity + specificity) / 2
        print('Accuracy of model on test set is {:.2f}%'.format(100 * accuracy))
        print('Balanced accuracy of model on test set is {:.2f}%'.format(100 * balanced_accuracy))
        return balanced_accuracy

    @staticmethod
    def divide_population(n_tg, n_wt):
        assert n_tg == n_wt, 'This cross validation procedure is designed to work only when n_tg == n_wt'
        kf_tg = KFold(n_splits=n_tg, shuffle=True)
        kf_wt = KFold(n_splits=n_wt, shuffle=True)

        train_set = []
        val_set = []

        for (idx_train_tg, idx_val_tg), (idx_train_wt, idx_val_wt) in zip(kf_tg.split(range(4)), kf_wt.split(range(4))):
            train_set.append((idx_train_tg, idx_train_wt))
            val_set.append((idx_val_tg, idx_val_wt))

        return train_set, val_set


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
