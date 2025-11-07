import numpy as np
import tensorflow as tf
import torch
from tensorflow import keras
from tensorflow.keras.layers import Dense

# Set TensorFlow logging level to ERROR
tf.logging.set_verbosity(tf.logging.ERROR)


# Custom Pytorch Module for Neural Networks
class PyTorchNeuralNetwork(torch.nn.Module):
    """
    Initializes a PyTorch neural network model with specified number of inputs, outputs, and neurons.

    Parameters
    ----------
    n_inputs (int): Number of input features.
    n_outputs (int): Number of output classes.
    n_neurons (int): Number of neurons in hidden layers.

    Returns
    -------
    PyTorchNeuralNetwork.

    Raises
    -------
    None.
    """

    # Constructor
    def __init__(self, n_inputs, n_outputs, n_neurons):
        super(PyTorchNeuralNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(n_inputs, n_neurons)
        self.fc2 = torch.nn.Linear(n_neurons, n_neurons)
        self.fc3 = torch.nn.Linear(n_neurons, n_outputs)

    # Predictions
    def forward(self, x):
        """
        Performs the forward pass of the neural network.

        Parameters
        -------
        x (torch.Tensor): Input tensor to the neural network.

        Returns
        -------
        torch.Tensor: Predicted output tensor.

        Raises
        -------
        None.
        """
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        y_pred = torch.nn.functional.softmax(self.fc3(x))
        return y_pred

    def fit(self, x_train, y_train):
        """
        Fits the neural network to the training data.

        Parameters
        ----------
        x_train (array-like): Input training data.
        y_train (array-like): Target training data.

        Returns
        -------
        PyTorchNeuralNetwork: Trained neural network instance.

        Raises
        ------
        None.
        """
        x_train_tensor = torch.from_numpy(np.array(x_train).astype(np.float32))
        y_train_tensor = torch.from_numpy(np.array(y_train)).type(torch.LongTensor)

        train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=1000, shuffle=True
        )

        # defining the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        # defining Cross-Entropy loss
        criterion = torch.nn.NLLLoss()

        epochs = 1  # TODO increase epochs for better training/ allow as parameter
        for _ in range(epochs):
            for i, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self(data)
                output = torch.log(output)
                loss = criterion(output, target)

                loss.backward()
                optimizer.step()

        return self

    def predict(self, test):
        """
        Predicts using the trained neural network.

        Parameters
        -------
        test (torch.Tensor): Input tensor for prediction.

        Returns
        -------
        torch.Tensor: Predicted output tensor.

        Raises
        -------
        None.
        """
        self.eval()
        y_train_pred = []
        with torch.no_grad():
            output = self(test)

            y_train_pred.extend(output)

        y_train_pred = torch.stack(y_train_pred)
        return y_train_pred


# Custom Pytorch Module for Logistic Regression
class PyTorchLogisticRegression(torch.nn.Module):
    """
    Initializes a PyTorch logistic regression linear model with specified number of inputs, outputs.

    Parameters
    ----------
      n_inputs (int): Number of input features.
      n_outputs (int): Number of output classes.

    Returns
    -------
    PyTorchLogisticRegression.

    Raises
    -------
    None.
    """

    # Constructor
    def __init__(self, n_inputs, n_outputs):
        super(PyTorchLogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(n_inputs, n_outputs)

    # Predictions
    def forward(self, x):
        """
        Performs the forward pass of the logistic regression model.

        Parameters
        -------
        x (torch.Tensor): Input tensor to the logistic regression model.

        Returns
        -------
        torch.Tensor: Predicted output tensor.

        Raises
        -------
        None.
        """
        y_pred = torch.nn.functional.softmax(self.linear(x))
        return y_pred

    def fit(self, x_train, y_train):
        """
        Fits the logistic regression model to the training data.

        Parameters
        ----------
        x_train (array-like): Input training data.
        y_train (array-like): Target training data.

        Returns
        -------
        PyTorchLogisticRegression: Trained logistic regression model instance.

        Raises
        ------
        None.
        """
        x_train_tensor = torch.from_numpy(np.array(x_train).astype(np.float32))
        y_train_tensor = torch.from_numpy(np.array(y_train)).type(torch.LongTensor)

        train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=1000, shuffle=True
        )

        # Defining the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        # Defining Cross-Entropy loss
        criterion = torch.nn.NLLLoss()

        epochs = 1
        for _ in range(epochs):
            for i, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self(data)
                output = torch.log(output)
                loss = criterion(output, target)

                loss.backward()
                optimizer.step()

        return self

    def predict(self, test):
        """
        Predicts using the trained logistic regression model.

        Parameters
        -------
        test (torch.Tensor): Input tensor for prediction.

        Returns
        -------
        torch.Tensor: Predicted output tensor.

        Raises
        -------
        None.
        """
        self.eval()
        y_train_pred = []
        with torch.no_grad():
            output = self(test)

            y_train_pred.extend(output)

        y_train_pred = torch.stack(y_train_pred)
        return y_train_pred


# Custom TensorFlow Module for Neural Networks
class TensorflowNeuralNetwork(keras.Model):
    """
    Initializes a Tensor neural network model with specified number of inputs, outputs, and neurons.

    Parameters
    ----------
    n_inputs (int): Number of input features.
    n_outputs (int): Number of output classes.
    n_neurons (int): Number of neurons in hidden layers.

    Returns
    -------
    TensorflowNeuralNetwork.

    Raises
    -------
    None.
    """

    def __init__(self, n_inputs, n_outputs, n_neurons):
        super(TensorflowNeuralNetwork, self).__init__()
        self.fc1 = Dense(n_neurons, activation="relu", input_dim=n_inputs)
        self.fc2 = Dense(n_neurons, activation="relu", input_dim=n_neurons)
        self.fc3 = Dense(n_outputs, activation="softmax", input_dim=n_neurons)

    def call(self, inputs, training=False):
        """
        Performs the forward pass of the neural network.

        Parameters
        -------
        inputs (tf.Tensor): Input tensor to the neural network.

        Returns
        -------
        tf.Tensor: Predicted output tensor.

        Raises
        -------
        None.
        """
        fc1_out = self.fc1(inputs)
        fc2_out = self.fc2(fc1_out)
        y_pred = self.fc3(fc2_out)
        return y_pred

    def fit(self, x_train, y_train):
        """
        Fits the neural network to the training data.

        Parameters
        ----------
        x_train (array-like): Input training data.
        y_train (array-like): Target training data.

        Returns
        -------
        TensorflowNeuralNetwork: Trained neural network instance.

        Raises
        ------
        None.
        """
        self.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        super(TensorflowNeuralNetwork, self).fit(x_train, y_train)

        return self

    def predict(self, test):
        """
        Predicts using the trained neural network.

        Parameters
        -------
        test (tf.Tensor): Input tensor for prediction.

        Returns
        -------
        tf.Tensor: Predicted output tensor.

        Raises
        -------
        None.
        """
        # Predict method
        output = super(TensorflowNeuralNetwork, self).predict(test)
        return output


# Custom TensorFlow Module for Logistic Regression
class TensorflowLogisticRegression(keras.Model):
    """
    Initializes a Tensorflow logistic regression linear model with specified number of inputs, outputs.

    Parameters
    ----------
      n_inputs (int): Number of input features.
      n_outputs (int): Number of output classes.

    Returns
    -------
    TensorflowLogisticRegression.

    Raises
    -------
    None.
    """

    def __init__(self, n_inputs, n_outputs):
        super(TensorflowLogisticRegression, self).__init__()
        self.linear = Dense(n_outputs, activation="softmax", input_dim=n_inputs)

    def call(self, inputs, training=False):
        """
        Performs the forward pass of the logistic regression model.

        Parameters
        -------
        inputs (tf.Tensor): Input tensor to the logistic regression model.

        Returns
        -------
        tf.Tensor: Predicted output tensor.

        Raises
        -------
        None.
        """
        y_pred = self.linear(inputs)
        return y_pred

    def fit(self, x_train, y_train):
        """
        Fits the logistic regression model to the training data.

        Parameters
        ----------
        x_train (array-like): Input training data.
        y_train (array-like): Target training data.

        Returns
        -------
        TensorflowLogisticRegression: Trained logistic regression model instance.

        Raises
        ------
        None.
        """
        self.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        super(TensorflowLogisticRegression, self).fit(x_train, y_train)

        return self

    def predict(self, test):
        """
        Predicts using the trained logistic regression model.

        Parameters
        -------
        test (tf.Tensor): Input tensor for prediction.

        Returns
        -------
        tf.Tensor: Predicted output tensor.

        Raises
        -------
        None.
        """
        # Predict method
        output = super(TensorflowLogisticRegression, self).predict(test)
        return output
