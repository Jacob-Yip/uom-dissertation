import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader


def load_data(filename: str, column_names: list, output_column_name: str, normalise_data=True, remove_outliers=False) -> tuple:
    """
    Load data from csv file
    Make sure you remove all outliers before normalising data
    TODO: Currently, we are focusing on regression task

    :param: filename: The name of the csv file
    :param: column_names: The column names of the csv file
    :param: output_column_name: The name of the column that represents the output label
    :param: normalise_data: Whether we want to normalise data
    :param: remove_outliers: Whether we want to remove outliers
    :return: (X, y), where each item is a numpy array
    """
    dataframe = pd.read_csv(filename, header=None,
                            delimiter=r"\s+", names=column_names)

    if remove_outliers:
        # Remove outliers
        dataframe = remove_dataframe_outliers(dataframe=dataframe)

    # Separate input and output data from the dataframe
    X = dataframe.drop(output_column_name, axis=1).values.astype(
        np.float32)  # Input column
    y = dataframe[output_column_name].values.astype(
        np.float32).reshape(-1, 1)  # Output/target column

    if normalise_data:
        # Scale input features (X) to [0, 1]
        x_scaler = MinMaxScaler(feature_range=(0, 1))
        X = x_scaler.fit_transform(X)

        # Scale output target (y) to [-1, 1]
        y_scaler = MinMaxScaler(feature_range=(-1, 1))
        y = y_scaler.fit_transform(y)

    return (X, y)


def get_data_loaders(filename: str, column_names: list, output_column_name: str, test_size: float, random_state: int, batch_size: int, validation_size=None, normalise_data=True, remove_outliers=True) -> tuple:
    """
    Return data loaders of training and testing data and possibly validation data
    Return (training_data_loader, testing_data_loader) or (training_data_loader, validation_data_loader, testing_data_loader)

    :return: (training_data_loader, testing_data_loader)
    """
    X_raw, y_raw = load_data(
        filename=filename,
        column_names=column_names,
        output_column_name=output_column_name,
        normalise_data=normalise_data,
        remove_outliers=remove_outliers
    )

    # Split data set to training and testing
    X_train, X_test, y_train, y_test = to_train_test_data(
        X=X_raw,
        y=y_raw,
        test_size=test_size,
        random_state=random_state
    )

    if validation_size is None:
        # No validation set

        # To PyTorch tensor
        X_train, X_test, y_train, y_test = to_tensor(
            X_train,
            X_test,
            y_train,
            y_test
        )

        # Create dataset
        dataset_train = to_dataset(X_train, y_train)
        dataset_test = to_dataset(X_test, y_test)

        # Create data loader
        # Always shuffle data
        data_loader_train = to_data_loader(
            dataset=dataset_train, batch_size=batch_size)
        data_loader_test = to_data_loader(
            dataset=dataset_test, batch_size=batch_size)

        return (data_loader_train, data_loader_test)
    else:
        # Create validation set
        X_train, X_validation, y_train, y_validation = to_train_test_data(
            X=X_train,
            y=y_train,
            test_size=validation_size,
            random_state=random_state
        )

        # To PyTorch tensor
        X_train, X_validation, X_test, y_train, y_validation, y_test = to_tensor(
            X_train,
            X_validation,
            X_test,
            y_train,
            y_validation,
            y_test
        )

        # Create dataset
        dataset_train = to_dataset(X_train, y_train)
        dataset_validation = to_dataset(X_validation, y_validation)
        dataset_test = to_dataset(X_test, y_test)

        # Create data loader
        # Always shuffle data
        data_loader_train = to_data_loader(
            dataset=dataset_train, batch_size=batch_size)
        data_loader_validation = to_data_loader(
            dataset=dataset_validation, batch_size=batch_size)
        data_loader_test = to_data_loader(
            dataset=dataset_test, batch_size=batch_size)

        return (data_loader_train, data_loader_validation, data_loader_test)


def remove_dataframe_outliers(dataframe):
    """
    Remove outliers from a dataframe

    :param: dataframe: The original dataframe instance
    :return: The cleaned dataframe instance
    """
    for column in dataframe.columns:
        Q1 = dataframe[column].quantile(0.25)
        Q3 = dataframe[column].quantile(0.75)
        IQR = Q3 - Q1  # Inter-quartile range
        lower_bound = Q1 - 1.5 * IQR  # Inclusive lower bound of valid range
        upper_bound = Q3 + 1.5 * IQR  # Inclusive upper bound of valid range

        # Keep rows where values are within bounds for this column
        dataframe = dataframe[(dataframe[column] >= lower_bound) & (
            dataframe[column] <= upper_bound)]

    return dataframe


def to_train_test_data(X, y, test_size=0.2, random_state=42) -> tuple:
    """
    Split X and y to (X_train, X_test, y_train, y_test)

    :param: X: All input data
    :param: y: All output data
    :param: test_size: Proportion of test data compared to the whole data
    :param: random_state: For randomness
    :return: (X_train, X_test, y_train, y_test)
    """
    assert test_size > 0 and test_size < 1, f"Invalid value of test_size (expect value between 0 and 1 exclusively): {test_size}"

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def to_tensor(*data) -> tuple:
    """
    Convert data from numpy array to PyTorch tensor

    :data: The list of data items
    :return: Tuple of corresponding PyTorch tensors
    """
    return (torch.from_numpy(data_item) for data_item in data)


def to_dataset(X_tensor: torch.Tensor, y_tensor: torch.Tensor):
    """
    Return a dataset of the corresponding set of data
    Expect X_tensor and y_tensor belong to training or testing together

    :param: X_tensor: The first data in the form of PyTorch tensor
    :param: y_tensor: The second data in the form of PyTorch tensor
    :return: A dataset instance of these 2 data
    """
    # Create Dataset and DataLoader
    return TensorDataset(X_tensor, y_tensor)


def to_data_loader(dataset, batch_size=32, shuffle=True):
    """
    Return a data loader of the corresponding dataset
    A data loader is either for training or testing

    :param: dataset: The dataset instance
    :param: The size of each batch of data
    :param: shuffle: True if we want to randomise the order of data
    :return: The corresponding data loader instance
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def apply_dataframe_filter(df: pd.DataFrame, filter: dict) -> pd.DataFrame:
    """
    Filter a dataframe with the given filter
    The format of filter is expected to be {column_name: [acceptable_column_value]}
    """
    mask = pd.Series(True, index=df.index)

    for column_name, acceptable_column_values in filter.items():
        mask &= df[column_name].isin(acceptable_column_values)
    
    return df[mask]


def assert_frame_equal_ignore_order(df1, df2, list_columns=None):
    """
    Compare two DataFrames ignoring row order
    list_columns: list of column names that contain lists
    """
    df1_copy = df1.copy()
    df2_copy = df2.copy()

    # Convert list columns to tuple so they can be sorted
    if list_columns:
        for col in list_columns:
            df1_copy[col] = df1_copy[col].apply(tuple)
            df2_copy[col] = df2_copy[col].apply(tuple)

    # Sort by all columns and reset index
    sort_cols = list(df1_copy.columns)
    df1_sorted = df1_copy.sort_values(by=sort_cols).reset_index(drop=True)
    df2_sorted = df2_copy.sort_values(by=sort_cols).reset_index(drop=True)

    try:
        assert_frame_equal(df1_sorted, df2_sorted)
    except AssertionError:
        raise Exception(f"Expected {df2_sorted} but received: {df1_sorted}")