class MajorityVoting:

    def __init__(self, parameters_dict, splits, cnn):
        # get best hyperparameters
        # then vote
        self.k_fold = len