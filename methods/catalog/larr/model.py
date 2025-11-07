from methods.api.recourse_method import RecourseMethod


class Larr(RecourseMethod):
    """
    Implementation of LARR (Learning-Augmented Robust Recourse) [1]_.

    Parameters
    ----------
    mlmodel : model.MLModel
        Black-Box-Model
    hyperparams : dict
        Dictionary containing hyperparameters. See Notes below to see its content.

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.

    Notes
    -----
    - Hyperparams
        Hyperparameter contains important information for the recourse method to initialize.
        Please make sure to pass all values as dict with the following keys.

        * "lr": float, default: 0.01
            Learning rate for gradient descent.


    .. [1] Kayastha, K., Gkatzelis, V., Jabbari, S. (2025). Learning-Augmented Robust Algorithmic Recourse. Drexel University.
    """

    def __init__(self, mlmodel):
        super().__init__(mlmodel)
    

    def get_counterfactuals(self, factuals):
        

        return ...