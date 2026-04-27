import ast
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from src.utils import data_helper, logger

"""
Run: python -m src.utils.experiment.data_plotter_mlp

NOTE: If you run data_plotter_mlp.py now, the generated graphs are not the same as the ones used in the report, which have a different value of learning_rate
"""


# Constants
# Path to project root
ROOT_PATH = os.path.join(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Path to each csv file in csv/
CSV_PATH_MLP = os.path.join(
    ROOT_PATH, f"data", f"experiment", f"mlp", f"csv", f"experiment_runner_mlp.csv")

# Path to img/
IMG_REPOSITORY_PATH_MLP = os.path.join(
    ROOT_PATH, f"data", f"experiment", f"mlp", f"img")

# Global configurations of experiment graphs
FONT_SIZE = 18
FONT_FAMILY = "serif"
FONT_SERIF = ["Times New Roman"]
AXES_TITLESIZE = 18
AXES_TITLEWEIGHT = "normal"
AXES_LABELSIZE = 18
LEGEND_FONTSIZE = 12

# Customise plt appearence
plt.rcParams.update({
    "font.size": FONT_SIZE,
    "font.family": FONT_FAMILY,
    "font.serif": FONT_SERIF,
    "axes.titlesize": AXES_TITLESIZE,
    "axes.titleweight": AXES_TITLEWEIGHT,
    "axes.labelsize": AXES_LABELSIZE,
    "legend.fontsize": LEGEND_FONTSIZE,
    # whitegrid style equivalent
    "axes.grid": True,
    "grid.color": "gray",
    "grid.alpha": 0.3
})

# ========================================================================


def mse_against_epoch_num_10_1000_fixed_learning_rate_00001(raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_mlp = {
        "hidden_size": [
            [2],
            [2, 2, 2],
            [2, 2, 2, 2, 2, 2],
            [6],
            [12]
        ],
        "learning_rate": [
            0.001
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"epoch_num"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs epoch_num\n(learning_rate: 0.001)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"mlp_mse_against_epoch_num_10_1000_fixed_learning_rate_00001.png"
    # ====================================================================

    df_mlp = data_helper.apply_dataframe_filter(
        df=raw_df_mlp,
        filter=filter_mlp
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(df_mlp["epoch_num"].unique())

    # Plot lines
    # MLP - hidden_size = [2]
    mse_mlp_2 = np.array(df_mlp[df_mlp["hidden_size"].apply(
        lambda x: x == [2])]["loss_test"])
    assert len(mse_mlp_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp_2 has length {len(mse_mlp_2)}"
    axis.plot(data_x, mse_mlp_2, marker="o", label=f"hidden_size - [2]")

    # MLP - hidden_size = [2, 2, 2]
    mse_mlp_2_2_2 = np.array(df_mlp[df_mlp["hidden_size"].apply(
        lambda x: x == [2, 2, 2])]["loss_test"])
    assert len(mse_mlp_2_2_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp_2_2_2 has length {len(mse_mlp_2_2_2)}"
    axis.plot(data_x, mse_mlp_2_2_2, marker="o",
              label=f"hidden_size - [2, 2, 2]")

    # MLP - hidden_size = [2, 2, 2, 2, 2, 2]
    mse_mlp_2_2_2_2_2_2 = np.array(df_mlp[df_mlp["hidden_size"].apply(
        lambda x: x == [2, 2, 2, 2, 2, 2])]["loss_test"])
    assert len(mse_mlp_2_2_2_2_2_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp_2_2_2_2_2_2 has length {len(mse_mlp_2_2_2_2_2_2)}"
    axis.plot(data_x, mse_mlp_2_2_2_2_2_2, marker="o",
              label=f"hidden_size - [2, 2, 2, 2, 2, 2]")

    # MLP - hidden_size = [6]
    mse_mlp_6 = np.array(df_mlp[df_mlp["hidden_size"].apply(
        lambda x: x == [6])]["loss_test"])
    assert len(mse_mlp_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp_6 has length {len(mse_mlp_6)}"
    axis.plot(data_x, mse_mlp_6, marker="o", label=f"hidden_size - [6]")

    # MLP - hidden_size = [12]
    mse_mlp_12 = np.array(df_mlp[df_mlp["hidden_size"].apply(
        lambda x: x == [12])]["loss_test"])
    assert len(mse_mlp_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp_12 has length {len(mse_mlp_12)}"
    axis.plot(data_x, mse_mlp_12, marker="o", label=f"hidden_size - [12]")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend()

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_epoch_num_10_1000_fixed_learning_rate_00001_hidden_size_all(raw_df_mlp: pd.DataFrame, img_repository_path: str):
    """
    It plots a graph of MSE against epoch_num for all models of all values of hidden_size
    For internal evaluation, i.e. not for report
    It allows me to decide the best value of epoch_num for traditional NCL and static NCL experiments
    """
    assert raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_mlp = {
        "hidden_size": [
            [2],
            [2, 2],
            [2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
            [4],
            [6],
            [8],
            [10],
            [12]
        ],
        "learning_rate": [
            0.001
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"epoch_num"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs epoch_num\n(learning_rate: 0.001)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"mlp_mse_against_epoch_num_10_1000_fixed_learning_rate_00001_hidden_size_all.png"
    # ====================================================================

    df_mlp = data_helper.apply_dataframe_filter(
        df=raw_df_mlp,
        filter=filter_mlp
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(df_mlp["epoch_num"].unique())

    # Plot lines
    # MLP - hidden_size = [2]
    mse_mlp_2 = np.array(df_mlp[df_mlp["hidden_size"].apply(
        lambda x: x == [2])]["loss_test"])
    assert len(mse_mlp_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp_2 has length {len(mse_mlp_2)}"
    axis.plot(data_x, mse_mlp_2, marker="o", label=f"hidden_size - [2]")

    # MLP - hidden_size = [2, 2, 2]
    mse_mlp_2_2_2 = np.array(df_mlp[df_mlp["hidden_size"].apply(
        lambda x: x == [2, 2, 2])]["loss_test"])
    assert len(mse_mlp_2_2_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp_2_2_2 has length {len(mse_mlp_2_2_2)}"
    axis.plot(data_x, mse_mlp_2_2_2, marker="o",
              label=f"hidden_size - [2, 2, 2]")

    # MLP - hidden_size = [2, 2, 2, 2, 2, 2]
    mse_mlp_2_2_2_2_2_2 = np.array(df_mlp[df_mlp["hidden_size"].apply(
        lambda x: x == [2, 2, 2, 2, 2, 2])]["loss_test"])
    assert len(mse_mlp_2_2_2_2_2_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp_2_2_2_2_2_2 has length {len(mse_mlp_2_2_2_2_2_2)}"
    axis.plot(data_x, mse_mlp_2_2_2_2_2_2, marker="o",
              label=f"hidden_size - [2, 2, 2, 2, 2, 2]")

    # MLP - hidden_size = [6]
    mse_mlp_6 = np.array(df_mlp[df_mlp["hidden_size"].apply(
        lambda x: x == [6])]["loss_test"])
    assert len(mse_mlp_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp_6 has length {len(mse_mlp_6)}"
    axis.plot(data_x, mse_mlp_6, marker="o", label=f"hidden_size - [6]")

    # MLP - hidden_size = [12]
    mse_mlp_12 = np.array(df_mlp[df_mlp["hidden_size"].apply(
        lambda x: x == [12])]["loss_test"])
    assert len(mse_mlp_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp_12 has length {len(mse_mlp_12)}"
    axis.plot(data_x, mse_mlp_12, marker="o", label=f"hidden_size - [12]")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend()

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_epoch_num_10_1000_fixed_learning_rate_00001_hidden_size_2(raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_mlp = {
        "hidden_size": [
            [2]
        ],
        "learning_rate": [
            0.001
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"epoch_num"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs epoch_num\n(learning_rate: 0.001; hidden_size: [2])"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"mlp_mse_against_epoch_num_10_1000_fixed_learning_rate_00001_hidden_size_2.png"
    # ====================================================================

    df_mlp = data_helper.apply_dataframe_filter(
        df=raw_df_mlp,
        filter=filter_mlp
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(df_mlp["epoch_num"].unique())

    # Plot lines
    # MLP - hidden_size = [2]
    mse_test_mlp_2 = np.array(
        df_mlp[df_mlp["hidden_size"].apply(lambda x: x == [2])]["loss_test"])
    assert len(mse_test_mlp_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_test_mlp_2 has length {len(mse_test_mlp_2)}"
    axis.plot(data_x, mse_test_mlp_2, marker="o", label=f"Testing")

    # MLP - hidden_size = [2]
    mse_train_mlp_2 = np.array(
        df_mlp[df_mlp["hidden_size"].apply(lambda x: x == [2])]["loss_train"])
    assert len(mse_train_mlp_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_train_mlp_2 has length {len(mse_train_mlp_2)}"
    axis.plot(data_x, mse_train_mlp_2, marker="o", label=f"Training")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend()

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_epoch_num_10_1000_fixed_learning_rate_00001_hidden_size_2_2_2(raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_mlp = {
        "hidden_size": [
            [2, 2, 2]
        ],
        "learning_rate": [
            0.001
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"epoch_num"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs epoch_num\n(learning_rate: 0.001; hidden_size: [2, 2, 2])"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"mlp_mse_against_epoch_num_10_1000_fixed_learning_rate_00001_hidden_size_2_2_2.png"
    # ====================================================================

    df_mlp = data_helper.apply_dataframe_filter(
        df=raw_df_mlp,
        filter=filter_mlp
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(df_mlp["epoch_num"].unique())

    # Plot lines
    # MLP - hidden_size = [2, 2, 2]
    mse_test_mlp_2_2_2 = np.array(
        df_mlp[df_mlp["hidden_size"].apply(lambda x: x == [2, 2, 2])]["loss_test"])
    assert len(mse_test_mlp_2_2_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_test_mlp_2_2_2 has length {len(mse_test_mlp_2_2_2)}"
    axis.plot(data_x, mse_test_mlp_2_2_2, marker="o", label=f"Testing")

    # MLP - hidden_size = [2, 2, 2]
    mse_train_mlp_2_2_2 = np.array(
        df_mlp[df_mlp["hidden_size"].apply(lambda x: x == [2, 2, 2])]["loss_train"])
    assert len(mse_train_mlp_2_2_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_train_mlp_2_2_2 has length {len(mse_train_mlp_2_2_2)}"
    axis.plot(data_x, mse_train_mlp_2_2_2, marker="o", label=f"Training")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend()

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_epoch_num_10_1000_fixed_learning_rate_00001_hidden_size_2_2_2_2_2_2(raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_mlp = {
        "hidden_size": [
            [2, 2, 2, 2, 2, 2]
        ],
        "learning_rate": [
            0.001
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"epoch_num"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs epoch_num\n(learning_rate: 0.001; hidden_size: [2, 2, 2, 2, 2, 2])"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"mlp_mse_against_epoch_num_10_1000_fixed_learning_rate_00001_hidden_size_2_2_2_2_2_2.png"
    # ====================================================================

    df_mlp = data_helper.apply_dataframe_filter(
        df=raw_df_mlp,
        filter=filter_mlp
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(df_mlp["epoch_num"].unique())

    # Plot lines
    # MLP - hidden_size = [2, 2, 2, 2, 2, 2]
    mse_test_mlp_2_2_2_2_2_2 = np.array(df_mlp[df_mlp["hidden_size"].apply(
        lambda x: x == [2, 2, 2, 2, 2, 2])]["loss_test"])
    assert len(mse_test_mlp_2_2_2_2_2_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_test_mlp_2_2_2_2_2_2 has length {len(mse_test_mlp_2_2_2_2_2_2)}"
    axis.plot(data_x, mse_test_mlp_2_2_2_2_2_2, marker="o", label=f"Testing")

    # MLP - hidden_size = [2, 2, 2, 2, 2, 2]
    mse_train_mlp_2_2_2_2_2_2 = np.array(df_mlp[df_mlp["hidden_size"].apply(
        lambda x: x == [2, 2, 2, 2, 2, 2])]["loss_train"])
    assert len(mse_train_mlp_2_2_2_2_2_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_train_mlp_2_2_2_2_2_2 has length {len(mse_train_mlp_2_2_2_2_2_2)}"
    axis.plot(data_x, mse_train_mlp_2_2_2_2_2_2, marker="o", label=f"Training")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend()

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_epoch_num_10_1000_fixed_learning_rate_00001_hidden_size_6(raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_mlp = {
        "hidden_size": [
            [6]
        ],
        "learning_rate": [
            0.001
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"epoch_num"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs epoch_num\n(learning_rate: 0.001; hidden_size: [6])"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"mlp_mse_against_epoch_num_10_1000_fixed_learning_rate_00001_hidden_size_6.png"
    # ====================================================================

    df_mlp = data_helper.apply_dataframe_filter(
        df=raw_df_mlp,
        filter=filter_mlp
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(df_mlp["epoch_num"].unique())

    # Plot lines
    # MLP - hidden_size = [6]
    mse_test_mlp_6 = np.array(
        df_mlp[df_mlp["hidden_size"].apply(lambda x: x == [6])]["loss_test"])
    assert len(mse_test_mlp_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_test_mlp_6 has length {len(mse_test_mlp_6)}"
    axis.plot(data_x, mse_test_mlp_6, marker="o", label=f"Testing")

    # MLP - hidden_size = [6]
    mse_train_mlp_6 = np.array(
        df_mlp[df_mlp["hidden_size"].apply(lambda x: x == [6])]["loss_train"])
    assert len(mse_train_mlp_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_train_mlp_6 has length {len(mse_train_mlp_6)}"
    axis.plot(data_x, mse_train_mlp_6, marker="o", label=f"Training")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend()

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_epoch_num_10_1000_fixed_learning_rate_00001_hidden_size_12(raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_mlp = {
        "hidden_size": [
            [12]
        ],
        "learning_rate": [
            0.001
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"epoch_num"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs epoch_num\n(learning_rate: 0.001; hidden_size: [12])"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"mlp_mse_against_epoch_num_10_1000_fixed_learning_rate_00001_hidden_size_12.png"
    # ====================================================================

    df_mlp = data_helper.apply_dataframe_filter(
        df=raw_df_mlp,
        filter=filter_mlp
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(df_mlp["epoch_num"].unique())

    # Plot lines
    # MLP - hidden_size = [12]
    mse_test_mlp_12 = np.array(
        df_mlp[df_mlp["hidden_size"].apply(lambda x: x == [12])]["loss_test"])
    assert len(mse_test_mlp_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_test_mlp_12 has length {len(mse_test_mlp_12)}"
    axis.plot(data_x, mse_test_mlp_12, marker="o", label=f"Testing")

    # MLP - hidden_size = [12]
    mse_train_mlp_12 = np.array(
        df_mlp[df_mlp["hidden_size"].apply(lambda x: x == [12])]["loss_train"])
    assert len(mse_train_mlp_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_train_mlp_12 has length {len(mse_train_mlp_12)}"
    axis.plot(data_x, mse_train_mlp_12, marker="o", label=f"Training")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend()

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


if __name__ == "__main__":
    df_mlp = pd.read_csv(CSV_PATH_MLP)
    # NOTE: If we read from csv, df_mlp["hidden_size"] will be of type class "str" instead of "list"
    df_mlp["hidden_size"] = df_mlp["hidden_size"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    mse_against_epoch_num_10_1000_fixed_learning_rate_00001(
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_MLP
    )

    mse_against_epoch_num_10_1000_fixed_learning_rate_00001_hidden_size_all(
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_MLP
    )

    mse_against_epoch_num_10_1000_fixed_learning_rate_00001_hidden_size_2(
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_MLP
    )

    mse_against_epoch_num_10_1000_fixed_learning_rate_00001_hidden_size_2_2_2(
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_MLP
    )

    mse_against_epoch_num_10_1000_fixed_learning_rate_00001_hidden_size_2_2_2_2_2_2(
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_MLP
    )

    mse_against_epoch_num_10_1000_fixed_learning_rate_00001_hidden_size_6(
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_MLP
    )

    mse_against_epoch_num_10_1000_fixed_learning_rate_00001_hidden_size_12(
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_MLP
    )

    logger.log(f"Graphs of MLP experiment are saved ...")
