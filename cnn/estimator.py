import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from cnn.neural_network import HistoNet
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
import torch
from colorama import Fore

class CnnClassifier(BaseEstimator, ClassifierMixin):
    """ An example classifier which implements a 1-NN algorithm.
    For more information regarding how to build your own classifier, read more
    in the :ref:`User Guide <user_guide>`.
    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.
    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """
    def __init__(self, learning_rate=0.1, batch_size=32, n_epochs=5):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.net = HistoNet()

    def fit(self, X, y):
        """A reference implementation of a fitting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """

        X, y = check_X_y(X, y, accept_sparse=True, multi_output=True, allow_nd=True)
        params = {'batch_size': self.batch_size,
                  'shuffle': True,
                  'num_workers': 8}
        dataset = TensorDataset(torch.from_numpy(X),
                                torch.from_numpy(y))
        dataloader = DataLoader(dataset, **params)
        print('Training CNN with parameters: '
              + 'batch size: ' + str(self.batch_size)
              + ' learning rate: ' + str(self.learning_rate)
              + ' n epochs: ' + str(self.n_epochs))
        self.net.train_nn(dataloader=dataloader,
                          lr=self.learning_rate,
                          n_epoch=self.n_epochs)
        self.is_fitted_ = True

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        # Return the classifier
        return self

    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X, ensure_2d=False, allow_nd=True)
        with torch.no_grad():
            probability = self.net(torch.from_numpy(X))
        probability_numpy = probability.detach().numpy()
        y_hat_numpy = np.argmax(probability_numpy, axis=1)
        return y_hat_numpy
