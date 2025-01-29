import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Data(ABC):
    """
    Abstract class to implement arbitrary datasets, which are provided by the user. This is the general data object
    that is used in
    """

    @property
    @abstractmethod
    def categorical(self):
        """
        Provides the column names of categorical data.
        Column names do not contain encoded information as provided by a get_dummy() method (e.g., sex_female)

        Label name is not included.

        Returns
        -------
        list of Strings
            List of all categorical columns
        """
        pass

    @property
    @abstractmethod
    def continuous(self):
        """
        Provides the column names of continuous data.

        Label name is not included.

        Returns
        -------
        list of Strings
            List of all continuous columns
        """
        pass

    @property
    @abstractmethod
    def immutables(self):
        """
        Provides the column names of immutable data.

        Label name is not included.

        Returns
        -------
        list of Strings
            List of all immutable columns
        """
        pass

    @property
    @abstractmethod
    def target(self):
        """
        Provides the name of the label column.

        Returns
        -------
        str
            Target label name
        """
        pass

    @property
    @abstractmethod
    def df(self):
        """
        The full Dataframe.

        Returns
        -------
        pd.DataFrame
        """
        pass

    @property
    @abstractmethod
    def df_train(self):
        """
        The training split Dataframe.

        Returns
        -------
        pd.DataFrame
        """
        pass

    @property
    @abstractmethod
    def df_test(self):
        """
        The testing split Dataframe.

        Returns
        -------
        pd.DataFrame
        """
        pass

    @property
    @abstractmethod
    def feature_input_order(self):
        """
        Saves the required order of features as list.

        Prevents confusion about correct order of input features in evaluation

        Returns
        -------
        list of str
        """
        pass

    @abstractmethod
    def transform(self, df):
        """
        Data transformation, for example normalization of continuous features and encoding of categorical features.

        Parameters
        ----------
        df: pd.DataFrame

        Returns
        -------
        pd.Dataframe
        """
        pass

    @abstractmethod
    def inverse_transform(self, df):
        """
        Inverts transform operation.

        Parameters
        ----------
        df: pd.DataFrame

        Returns
        -------
        pd.Dataframe
        """
        pass

    def get_ordered_features(self, x):
        """
        Restores the correct input feature order for the ML model, this also drops the columns not in the
        feature order. So it drops the target column, and possibly other features, e.g. categorical.

        Only works for encoded data

        Parameters
        ----------
        x : pd.DataFrame
            Data we want to order

        Returns
        -------
        output : pd.DataFrame
            Whole DataFrame with ordered feature
        """
        if isinstance(x, pd.DataFrame):
            return x[self.feature_input_order]
        else:
            warnings.warn(
                f"cannot re-order features for non dataframe input: {type(x)}"
            )
            return x

    def get_mutable_mask(self):
        """
        Get mask of mutable features.

        For example with mutable feature "income" and immutable features "age", the
        mask would be [True, False] for feature_input_order ["income", "age"].

        This mask can then be used to index data to only get the columns that are (im)mutable.

        Returns
        -------
        mutable_mask: np.array(bool)
        """
        # get categorical features
        categorical = self.categorical
        # get the binary encoded categorical features
        encoded_categorical = categorical
        # get the immutables, where the categorical features are in encoded format
        immutable = [
            encoded_categorical[categorical.index(i)] if i in categorical else i
            for i in self.immutables
        ]
        # find the index of the immutables in the feature input order
        immutable = [self.feature_input_order.index(col) for col in immutable]
        # make a mask
        mutable_mask = np.ones(len(self.feature_input_order), dtype=bool)
        # set the immutables to False
        mutable_mask[immutable] = False
        return mutable_mask
