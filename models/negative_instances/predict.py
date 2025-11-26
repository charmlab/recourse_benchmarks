from typing import Any, Union

import numpy as np
import pandas as pd

from data.catalog import DataCatalog


def predict_negative_instances(
    model: Any, data: Union[pd.DataFrame, DataCatalog]
) -> pd.DataFrame:
    """
    Predicts negative instances using the specified model and test data.

    Parameters
    ----------
    model (Any): The machine learning model used for prediction.
    data (Union[pd.DataFrame, DataCatalog]): Test data for prediction.

    Returns
    -------
    pd.DataFrame: DataFrame containing the negative instances.

    Raises
    ------
    ValueError: If the test set is empty.

    """
    # get processed data and remove target
    df = data.df_test.copy() if isinstance(data, DataCatalog) else data.copy()
    if df.empty:
        raise ValueError("Empty Test Set")
    df["y_neg"] = predict_label(model, df)
    df = df[df["y_neg"] == 0]
    df = df.drop("y_neg", axis="columns")

    return df


def predict_label(model: Any, df: pd.DataFrame, as_prob: bool = False) -> np.ndarray:
    """Predicts the data target

    Assumption: Positive class label is at position 1

    Parameters
    ----------
    name : Tensorflow or PyTorch Model
        Model object retrieved by :func:`load_model`
    df : pd.DataFrame
        Dataset used for predictions
    Returns
    -------
    predictions :  2d numpy array with predictions
    """

    predictions = model.predict(df)

    if not as_prob:
        predictions = predictions.round()

    return predictions
