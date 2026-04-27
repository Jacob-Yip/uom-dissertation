import ast
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from src.utils import data_helper, logger

"""
Run: python -m src.utils.experiment.data_plotter_voter_static_ncl
"""


# Constants
# Path to project root
ROOT_PATH = os.path.join(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Path to each csv file in csv/
CSV_PATH_ARITHMETIC_MEAN = os.path.join(
    ROOT_PATH, f"data", f"experiment", f"static-ncl", f"csv", f"experiment_runner_static_ncl_arithmetic_mean.csv")
CSV_PATH_MEDIAN = os.path.join(
    ROOT_PATH, f"data", f"experiment", f"static-ncl", f"csv", f"experiment_runner_static_ncl_median.csv")
CSV_PATH_NN = os.path.join(
    ROOT_PATH, f"data", f"experiment", f"static-ncl", f"csv", f"experiment_runner_static_ncl_nn.csv")

# Path to img/
IMG_REPOSITORY_PATH_VOTER_STATIC_NCL = os.path.join(
    ROOT_PATH, f"data", f"experiment", f"voter", f"static-ncl", f"img")

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


def mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_fixed_epoch_num_400(raw_df_arithmetic_mean: pd.DataFrame, raw_df_median: pd.DataFrame, raw_df_nn: pd.DataFrame, img_repository_path: str):
    assert raw_df_arithmetic_mean is not None and raw_df_median is not None and raw_df_nn is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_arithmetic_mean, raw_df_median, raw_df_nn, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_arithmetic_mean = {
        "hidden_size": [
            [2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_median = {
        "hidden_size": [
            [2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_nn = {
        "hidden_size": [
            [2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(each base learner is [2] trained with 400 epochs)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"voter_static_ncl_mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_fixed_epoch_num_400.png"
    # ====================================================================

    df_arithmetic_mean = data_helper.apply_dataframe_filter(
        df=raw_df_arithmetic_mean,
        filter=filter_arithmetic_mean
    )
    df_median = data_helper.apply_dataframe_filter(
        df=raw_df_median,
        filter=filter_median
    )
    df_nn = data_helper.apply_dataframe_filter(
        df=raw_df_nn,
        filter=filter_nn
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_arithmetic_mean["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Arithmetic mean - ensemble_size = 2
    mse_arithmetic_mean_2 = np.array(
        df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 2]["loss_test"])
    assert len(mse_arithmetic_mean_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_2 has length {len(mse_arithmetic_mean_2)}"
    axis.plot(data_x, mse_arithmetic_mean_2,
              marker="o", label=f"Arithmetic mean - ensemble_size: 2")

    # Arithmetic mean - ensemble_size = 6
    mse_arithmetic_mean_6 = np.array(
        df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 6]["loss_test"])
    assert len(mse_arithmetic_mean_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_6 has length {len(mse_arithmetic_mean_6)}"
    axis.plot(data_x, mse_arithmetic_mean_6,
              marker="o", label=f"Arithmetic mean - ensemble_size: 6")

    # Arithmetic mean - ensemble_size = 12
    mse_arithmetic_mean_12 = np.array(
        df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 12]["loss_test"])
    assert len(mse_arithmetic_mean_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_12 has length {len(mse_arithmetic_mean_12)}"
    axis.plot(data_x, mse_arithmetic_mean_12,
              marker="o", label=f"Arithmetic mean - ensemble_size: 12")

    # Median - ensemble_size = 2
    mse_median_2 = np.array(
        df_median[df_median["ensemble_size"] == 2]["loss_test"])
    assert len(mse_median_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_2 has length {len(mse_median_2)}"
    axis.plot(data_x, mse_median_2, marker="o",
              label=f"Median - ensemble_size: 2")

    # Median - ensemble_size = 6
    mse_median_6 = np.array(
        df_median[df_median["ensemble_size"] == 6]["loss_test"])
    assert len(mse_median_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_6 has length {len(mse_median_6)}"
    axis.plot(data_x, mse_median_6, marker="o",
              label=f"Median - ensemble_size: 6")

    # Median - ensemble_size = 12
    mse_median_12 = np.array(
        df_median[df_median["ensemble_size"] == 12]["loss_test"])
    assert len(mse_median_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_12 has length {len(mse_median_12)}"
    axis.plot(data_x, mse_median_12, marker="o",
              label=f"Median - ensemble_size: 12")

    # NN - ensemble_size = 2
    mse_nn_2 = np.array(df_nn[df_nn["ensemble_size"] == 2]["loss_test"])
    assert len(mse_nn_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_2 has length {len(mse_nn_2)}"
    axis.plot(data_x, mse_nn_2, marker="o",
              label=f"Neural network - ensemble_size: 2")

    # NN - ensemble_size = 6
    mse_nn_6 = np.array(df_nn[df_nn["ensemble_size"] == 6]["loss_test"])
    assert len(mse_nn_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_6 has length {len(mse_nn_6)}"
    axis.plot(data_x, mse_nn_6, marker="o", label=f"Neural network - 6")

    # NN - ensemble_size = 12
    mse_nn_12 = np.array(df_nn[df_nn["ensemble_size"] == 12]["loss_test"])
    assert len(mse_nn_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_12 has length {len(mse_nn_12)}"
    axis.plot(data_x, mse_nn_12, marker="o", label=f"Neural network - 12")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend(loc=f"upper center", bbox_to_anchor=(
        0.5, -0.25), ncol=1, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_2_2_fixed_epoch_num_400(raw_df_arithmetic_mean: pd.DataFrame, raw_df_median: pd.DataFrame, raw_df_nn: pd.DataFrame, img_repository_path: str):
    assert raw_df_arithmetic_mean is not None and raw_df_median is not None and raw_df_nn is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_arithmetic_mean, raw_df_median, raw_df_nn, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_arithmetic_mean = {
        "hidden_size": [
            [2, 2, 2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_median = {
        "hidden_size": [
            [2, 2, 2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_nn = {
        "hidden_size": [
            [2, 2, 2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(each base learner is [2, 2, 2] trained with 400 epochs)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"voter_static_ncl_mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_2_2_fixed_epoch_num_400.png"
    # ====================================================================

    df_arithmetic_mean = data_helper.apply_dataframe_filter(
        df=raw_df_arithmetic_mean,
        filter=filter_arithmetic_mean
    )
    df_median = data_helper.apply_dataframe_filter(
        df=raw_df_median,
        filter=filter_median
    )
    df_nn = data_helper.apply_dataframe_filter(
        df=raw_df_nn,
        filter=filter_nn
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_arithmetic_mean["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Arithmetic mean - ensemble_size = 2
    mse_arithmetic_mean_2 = np.array(
        df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 2]["loss_test"])
    assert len(mse_arithmetic_mean_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_2 has length {len(mse_arithmetic_mean_2)}"
    axis.plot(data_x, mse_arithmetic_mean_2,
              marker="o", label=f"Arithmetic mean - ensemble_size: 2")

    # Arithmetic mean - ensemble_size = 6
    mse_arithmetic_mean_6 = np.array(
        df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 6]["loss_test"])
    assert len(mse_arithmetic_mean_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_6 has length {len(mse_arithmetic_mean_6)}"
    axis.plot(data_x, mse_arithmetic_mean_6,
              marker="o", label=f"Arithmetic mean - ensemble_size: 6")

    # Arithmetic mean - ensemble_size = 12
    mse_arithmetic_mean_12 = np.array(
        df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 12]["loss_test"])
    assert len(mse_arithmetic_mean_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_12 has length {len(mse_arithmetic_mean_12)}"
    axis.plot(data_x, mse_arithmetic_mean_12,
              marker="o", label=f"Arithmetic mean - ensemble_size: 12")

    # Median - ensemble_size = 2
    mse_median_2 = np.array(
        df_median[df_median["ensemble_size"] == 2]["loss_test"])
    assert len(mse_median_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_2 has length {len(mse_median_2)}"
    axis.plot(data_x, mse_median_2, marker="o",
              label=f"Median - ensemble_size: 2")

    # Median - ensemble_size = 6
    mse_median_6 = np.array(
        df_median[df_median["ensemble_size"] == 6]["loss_test"])
    assert len(mse_median_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_6 has length {len(mse_median_6)}"
    axis.plot(data_x, mse_median_6, marker="o",
              label=f"Median - ensemble_size: 6")

    # Median - ensemble_size = 12
    mse_median_12 = np.array(
        df_median[df_median["ensemble_size"] == 12]["loss_test"])
    assert len(mse_median_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_12 has length {len(mse_median_12)}"
    axis.plot(data_x, mse_median_12, marker="o",
              label=f"Median - ensemble_size: 12")

    # NN - ensemble_size = 2
    mse_nn_2 = np.array(df_nn[df_nn["ensemble_size"] == 2]["loss_test"])
    assert len(mse_nn_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_2 has length {len(mse_nn_2)}"
    axis.plot(data_x, mse_nn_2, marker="o",
              label=f"Neural network - ensemble_size: 2")

    # NN - ensemble_size = 6
    mse_nn_6 = np.array(df_nn[df_nn["ensemble_size"] == 6]["loss_test"])
    assert len(mse_nn_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_6 has length {len(mse_nn_6)}"
    axis.plot(data_x, mse_nn_6, marker="o", label=f"Neural network - 6")

    # NN - ensemble_size = 12
    mse_nn_12 = np.array(df_nn[df_nn["ensemble_size"] == 12]["loss_test"])
    assert len(mse_nn_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_12 has length {len(mse_nn_12)}"
    axis.plot(data_x, mse_nn_12, marker="o", label=f"Neural network - 12")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend(loc=f"upper center", bbox_to_anchor=(
        0.5, -0.25), ncol=1, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_2_2_2_2_2_fixed_epoch_num_400(raw_df_arithmetic_mean: pd.DataFrame, raw_df_median: pd.DataFrame, raw_df_nn: pd.DataFrame, img_repository_path: str):
    assert raw_df_arithmetic_mean is not None and raw_df_median is not None and raw_df_nn is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_arithmetic_mean, raw_df_median, raw_df_nn, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_arithmetic_mean = {
        "hidden_size": [
            [2, 2, 2, 2, 2, 2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_median = {
        "hidden_size": [
            [2, 2, 2, 2, 2, 2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_nn = {
        "hidden_size": [
            [2, 2, 2, 2, 2, 2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(each base learner is [2, 2, 2, 2, 2, 2] trained with 400 epochs)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"voter_static_ncl_mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_2_2_2_2_2_fixed_epoch_num_400.png"
    # ====================================================================

    df_arithmetic_mean = data_helper.apply_dataframe_filter(
        df=raw_df_arithmetic_mean,
        filter=filter_arithmetic_mean
    )
    df_median = data_helper.apply_dataframe_filter(
        df=raw_df_median,
        filter=filter_median
    )
    df_nn = data_helper.apply_dataframe_filter(
        df=raw_df_nn,
        filter=filter_nn
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_arithmetic_mean["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Arithmetic mean - ensemble_size = 2
    mse_arithmetic_mean_2 = np.array(
        df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 2]["loss_test"])
    assert len(mse_arithmetic_mean_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_2 has length {len(mse_arithmetic_mean_2)}"
    axis.plot(data_x, mse_arithmetic_mean_2,
              marker="o", label=f"Arithmetic mean - ensemble_size: 2")

    # Arithmetic mean - ensemble_size = 6
    mse_arithmetic_mean_6 = np.array(
        df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 6]["loss_test"])
    assert len(mse_arithmetic_mean_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_6 has length {len(mse_arithmetic_mean_6)}"
    axis.plot(data_x, mse_arithmetic_mean_6,
              marker="o", label=f"Arithmetic mean - ensemble_size: 6")

    # Arithmetic mean - ensemble_size = 12
    mse_arithmetic_mean_12 = np.array(
        df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 12]["loss_test"])
    assert len(mse_arithmetic_mean_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_12 has length {len(mse_arithmetic_mean_12)}"
    axis.plot(data_x, mse_arithmetic_mean_12,
              marker="o", label=f"Arithmetic mean - ensemble_size: 12")

    # Median - ensemble_size = 2
    mse_median_2 = np.array(
        df_median[df_median["ensemble_size"] == 2]["loss_test"])
    assert len(mse_median_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_2 has length {len(mse_median_2)}"
    axis.plot(data_x, mse_median_2, marker="o",
              label=f"Median - ensemble_size: 2")

    # Median - ensemble_size = 6
    mse_median_6 = np.array(
        df_median[df_median["ensemble_size"] == 6]["loss_test"])
    assert len(mse_median_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_6 has length {len(mse_median_6)}"
    axis.plot(data_x, mse_median_6, marker="o",
              label=f"Median - ensemble_size: 6")

    # Median - ensemble_size = 12
    mse_median_12 = np.array(
        df_median[df_median["ensemble_size"] == 12]["loss_test"])
    assert len(mse_median_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_12 has length {len(mse_median_12)}"
    axis.plot(data_x, mse_median_12, marker="o",
              label=f"Median - ensemble_size: 12")

    # NN - ensemble_size = 2
    mse_nn_2 = np.array(df_nn[df_nn["ensemble_size"] == 2]["loss_test"])
    assert len(mse_nn_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_2 has length {len(mse_nn_2)}"
    axis.plot(data_x, mse_nn_2, marker="o",
              label=f"Neural network - ensemble_size: 2")

    # NN - ensemble_size = 6
    mse_nn_6 = np.array(df_nn[df_nn["ensemble_size"] == 6]["loss_test"])
    assert len(mse_nn_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_6 has length {len(mse_nn_6)}"
    axis.plot(data_x, mse_nn_6, marker="o", label=f"Neural network - 6")

    # NN - ensemble_size = 12
    mse_nn_12 = np.array(df_nn[df_nn["ensemble_size"] == 12]["loss_test"])
    assert len(mse_nn_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_12 has length {len(mse_nn_12)}"
    axis.plot(data_x, mse_nn_12, marker="o", label=f"Neural network - 12")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend(loc=f"upper center", bbox_to_anchor=(
        0.5, -0.25), ncol=1, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_6_fixed_epoch_num_400(raw_df_arithmetic_mean: pd.DataFrame, raw_df_median: pd.DataFrame, raw_df_nn: pd.DataFrame, img_repository_path: str):
    assert raw_df_arithmetic_mean is not None and raw_df_median is not None and raw_df_nn is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_arithmetic_mean, raw_df_median, raw_df_nn, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_arithmetic_mean = {
        "hidden_size": [
            [6]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_median = {
        "hidden_size": [
            [6]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_nn = {
        "hidden_size": [
            [6]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(each base learner is [6] trained with 400 epochs)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"voter_static_ncl_mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_6_fixed_epoch_num_400.png"
    # ====================================================================

    df_arithmetic_mean = data_helper.apply_dataframe_filter(
        df=raw_df_arithmetic_mean,
        filter=filter_arithmetic_mean
    )
    df_median = data_helper.apply_dataframe_filter(
        df=raw_df_median,
        filter=filter_median
    )
    df_nn = data_helper.apply_dataframe_filter(
        df=raw_df_nn,
        filter=filter_nn
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_arithmetic_mean["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Arithmetic mean - ensemble_size = 2
    mse_arithmetic_mean_2 = np.array(
        df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 2]["loss_test"])
    assert len(mse_arithmetic_mean_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_2 has length {len(mse_arithmetic_mean_2)}"
    axis.plot(data_x, mse_arithmetic_mean_2,
              marker="o", label=f"Arithmetic mean - ensemble_size: 2")

    # Arithmetic mean - ensemble_size = 6
    mse_arithmetic_mean_6 = np.array(
        df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 6]["loss_test"])
    assert len(mse_arithmetic_mean_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_6 has length {len(mse_arithmetic_mean_6)}"
    axis.plot(data_x, mse_arithmetic_mean_6,
              marker="o", label=f"Arithmetic mean - ensemble_size: 6")

    # Arithmetic mean - ensemble_size = 12
    mse_arithmetic_mean_12 = np.array(
        df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 12]["loss_test"])
    assert len(mse_arithmetic_mean_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_12 has length {len(mse_arithmetic_mean_12)}"
    axis.plot(data_x, mse_arithmetic_mean_12,
              marker="o", label=f"Arithmetic mean - ensemble_size: 12")

    # Median - ensemble_size = 2
    mse_median_2 = np.array(
        df_median[df_median["ensemble_size"] == 2]["loss_test"])
    assert len(mse_median_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_2 has length {len(mse_median_2)}"
    axis.plot(data_x, mse_median_2, marker="o",
              label=f"Median - ensemble_size: 2")

    # Median - ensemble_size = 6
    mse_median_6 = np.array(
        df_median[df_median["ensemble_size"] == 6]["loss_test"])
    assert len(mse_median_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_6 has length {len(mse_median_6)}"
    axis.plot(data_x, mse_median_6, marker="o",
              label=f"Median - ensemble_size: 6")

    # Median - ensemble_size = 12
    mse_median_12 = np.array(
        df_median[df_median["ensemble_size"] == 12]["loss_test"])
    assert len(mse_median_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_12 has length {len(mse_median_12)}"
    axis.plot(data_x, mse_median_12, marker="o",
              label=f"Median - ensemble_size: 12")

    # NN - ensemble_size = 2
    mse_nn_2 = np.array(df_nn[df_nn["ensemble_size"] == 2]["loss_test"])
    assert len(mse_nn_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_2 has length {len(mse_nn_2)}"
    axis.plot(data_x, mse_nn_2, marker="o",
              label=f"Neural network - ensemble_size: 2")

    # NN - ensemble_size = 6
    mse_nn_6 = np.array(df_nn[df_nn["ensemble_size"] == 6]["loss_test"])
    assert len(mse_nn_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_6 has length {len(mse_nn_6)}"
    axis.plot(data_x, mse_nn_6, marker="o", label=f"Neural network - 6")

    # NN - ensemble_size = 12
    mse_nn_12 = np.array(df_nn[df_nn["ensemble_size"] == 12]["loss_test"])
    assert len(mse_nn_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_12 has length {len(mse_nn_12)}"
    axis.plot(data_x, mse_nn_12, marker="o", label=f"Neural network - 12")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend(loc=f"upper center", bbox_to_anchor=(
        0.5, -0.25), ncol=1, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_12_fixed_epoch_num_400(raw_df_arithmetic_mean: pd.DataFrame, raw_df_median: pd.DataFrame, raw_df_nn: pd.DataFrame, img_repository_path: str):
    assert raw_df_arithmetic_mean is not None and raw_df_median is not None and raw_df_nn is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_arithmetic_mean, raw_df_median, raw_df_nn, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_arithmetic_mean = {
        "hidden_size": [
            [12]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_median = {
        "hidden_size": [
            [12]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_nn = {
        "hidden_size": [
            [12]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(each base learner is [12] trained with 400 epochs)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"voter_static_ncl_mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_12_fixed_epoch_num_400.png"
    # ====================================================================

    df_arithmetic_mean = data_helper.apply_dataframe_filter(
        df=raw_df_arithmetic_mean,
        filter=filter_arithmetic_mean
    )
    df_median = data_helper.apply_dataframe_filter(
        df=raw_df_median,
        filter=filter_median
    )
    df_nn = data_helper.apply_dataframe_filter(
        df=raw_df_nn,
        filter=filter_nn
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_arithmetic_mean["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Arithmetic mean - ensemble_size = 2
    mse_arithmetic_mean_2 = np.array(
        df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 2]["loss_test"])
    assert len(mse_arithmetic_mean_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_2 has length {len(mse_arithmetic_mean_2)}"
    axis.plot(data_x, mse_arithmetic_mean_2,
              marker="o", label=f"Arithmetic mean - ensemble_size: 2")

    # Arithmetic mean - ensemble_size = 6
    mse_arithmetic_mean_6 = np.array(
        df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 6]["loss_test"])
    assert len(mse_arithmetic_mean_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_6 has length {len(mse_arithmetic_mean_6)}"
    axis.plot(data_x, mse_arithmetic_mean_6,
              marker="o", label=f"Arithmetic mean - ensemble_size: 6")

    # Arithmetic mean - ensemble_size = 12
    mse_arithmetic_mean_12 = np.array(
        df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 12]["loss_test"])
    assert len(mse_arithmetic_mean_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_12 has length {len(mse_arithmetic_mean_12)}"
    axis.plot(data_x, mse_arithmetic_mean_12,
              marker="o", label=f"Arithmetic mean - ensemble_size: 12")

    # Median - ensemble_size = 2
    mse_median_2 = np.array(
        df_median[df_median["ensemble_size"] == 2]["loss_test"])
    assert len(mse_median_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_2 has length {len(mse_median_2)}"
    axis.plot(data_x, mse_median_2, marker="o",
              label=f"Median - ensemble_size: 2")

    # Median - ensemble_size = 6
    mse_median_6 = np.array(
        df_median[df_median["ensemble_size"] == 6]["loss_test"])
    assert len(mse_median_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_6 has length {len(mse_median_6)}"
    axis.plot(data_x, mse_median_6, marker="o",
              label=f"Median - ensemble_size: 6")

    # Median - ensemble_size = 12
    mse_median_12 = np.array(
        df_median[df_median["ensemble_size"] == 12]["loss_test"])
    assert len(mse_median_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_12 has length {len(mse_median_12)}"
    axis.plot(data_x, mse_median_12, marker="o",
              label=f"Median - ensemble_size: 12")

    # NN - ensemble_size = 2
    mse_nn_2 = np.array(df_nn[df_nn["ensemble_size"] == 2]["loss_test"])
    assert len(mse_nn_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_2 has length {len(mse_nn_2)}"
    axis.plot(data_x, mse_nn_2, marker="o",
              label=f"Neural network - ensemble_size: 2")

    # NN - ensemble_size = 6
    mse_nn_6 = np.array(df_nn[df_nn["ensemble_size"] == 6]["loss_test"])
    assert len(mse_nn_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_6 has length {len(mse_nn_6)}"
    axis.plot(data_x, mse_nn_6, marker="o", label=f"Neural network - 6")

    # NN - ensemble_size = 12
    mse_nn_12 = np.array(df_nn[df_nn["ensemble_size"] == 12]["loss_test"])
    assert len(mse_nn_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_12 has length {len(mse_nn_12)}"
    axis.plot(data_x, mse_nn_12, marker="o", label=f"Neural network - 12")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend(loc=f"upper center", bbox_to_anchor=(
        0.5, -0.25), ncol=1, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_fixed_epoch_num_400_fixed_ensemble_size_2(raw_df_arithmetic_mean: pd.DataFrame, raw_df_median: pd.DataFrame, raw_df_nn: pd.DataFrame, img_repository_path: str):
    assert raw_df_arithmetic_mean is not None and raw_df_median is not None and raw_df_nn is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_arithmetic_mean, raw_df_median, raw_df_nn, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_arithmetic_mean = {
        "hidden_size": [
            [2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_median = {
        "hidden_size": [
            [2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_nn = {
        "hidden_size": [
            [2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(each base learner is [2] trained with 400 epochs)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"voter_static_ncl_mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_fixed_epoch_num_400_fixed_ensemble_size_2.png"
    # ====================================================================

    df_arithmetic_mean = data_helper.apply_dataframe_filter(
        df=raw_df_arithmetic_mean,
        filter=filter_arithmetic_mean
    )
    df_median = data_helper.apply_dataframe_filter(
        df=raw_df_median,
        filter=filter_median
    )
    df_nn = data_helper.apply_dataframe_filter(
        df=raw_df_nn,
        filter=filter_nn
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_arithmetic_mean["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Arithmetic mean - ensemble_size = 2
    mse_arithmetic_mean_2 = np.array(
        df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 2]["loss_test"])
    assert len(mse_arithmetic_mean_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_2 has length {len(mse_arithmetic_mean_2)}"
    axis.plot(data_x, mse_arithmetic_mean_2,
              marker="o", label=f"Arithmetic mean - ensemble_size: 2")

    # # Arithmetic mean - ensemble_size = 6
    # mse_arithmetic_mean_6 = np.array(
    #     df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 6]["loss_test"])
    # assert len(mse_arithmetic_mean_6) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_6 has length {len(mse_arithmetic_mean_6)}"
    # axis.plot(data_x, mse_arithmetic_mean_6,
    #           marker="o", label=f"Arithmetic mean - ensemble_size: 6")

    # # Arithmetic mean - ensemble_size = 12
    # mse_arithmetic_mean_12 = np.array(
    #     df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 12]["loss_test"])
    # assert len(mse_arithmetic_mean_12) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_12 has length {len(mse_arithmetic_mean_12)}"
    # axis.plot(data_x, mse_arithmetic_mean_12,
    #           marker="o", label=f"Arithmetic mean - ensemble_size: 12")

    # Median - ensemble_size = 2
    mse_median_2 = np.array(
        df_median[df_median["ensemble_size"] == 2]["loss_test"])
    assert len(mse_median_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_2 has length {len(mse_median_2)}"
    axis.plot(data_x, mse_median_2, marker="o",
              label=f"Median - ensemble_size: 2")

    # # Median - ensemble_size = 6
    # mse_median_6 = np.array(
    #     df_median[df_median["ensemble_size"] == 6]["loss_test"])
    # assert len(mse_median_6) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_6 has length {len(mse_median_6)}"
    # axis.plot(data_x, mse_median_6, marker="o",
    #           label=f"Median - ensemble_size: 6")

    # # Median - ensemble_size = 12
    # mse_median_12 = np.array(
    #     df_median[df_median["ensemble_size"] == 12]["loss_test"])
    # assert len(mse_median_12) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_12 has length {len(mse_median_12)}"
    # axis.plot(data_x, mse_median_12, marker="o",
    #           label=f"Median - ensemble_size: 12")

    # NN - ensemble_size = 2
    mse_nn_2 = np.array(df_nn[df_nn["ensemble_size"] == 2]["loss_test"])
    assert len(mse_nn_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_2 has length {len(mse_nn_2)}"
    axis.plot(data_x, mse_nn_2, marker="o",
              label=f"Neural network - ensemble_size: 2")

    # # NN - ensemble_size = 6
    # mse_nn_6 = np.array(df_nn[df_nn["ensemble_size"] == 6]["loss_test"])
    # assert len(mse_nn_6) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_6 has length {len(mse_nn_6)}"
    # axis.plot(data_x, mse_nn_6, marker="o", label=f"Neural network - 6")

    # # NN - ensemble_size = 12
    # mse_nn_12 = np.array(df_nn[df_nn["ensemble_size"] == 12]["loss_test"])
    # assert len(mse_nn_12) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_12 has length {len(mse_nn_12)}"
    # axis.plot(data_x, mse_nn_12, marker="o", label=f"Neural network - 12")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend(loc=f"upper center", bbox_to_anchor=(
        0.5, -0.25), ncol=1, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_2_2_fixed_epoch_num_400_fixed_ensemble_size_2(raw_df_arithmetic_mean: pd.DataFrame, raw_df_median: pd.DataFrame, raw_df_nn: pd.DataFrame, img_repository_path: str):
    assert raw_df_arithmetic_mean is not None and raw_df_median is not None and raw_df_nn is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_arithmetic_mean, raw_df_median, raw_df_nn, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_arithmetic_mean = {
        "hidden_size": [
            [2, 2, 2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_median = {
        "hidden_size": [
            [2, 2, 2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_nn = {
        "hidden_size": [
            [2, 2, 2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(each base learner is [2, 2, 2] trained with 400 epochs)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"voter_static_ncl_mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_2_2_fixed_epoch_num_400_fixed_ensemble_size_2.png"
    # ====================================================================

    df_arithmetic_mean = data_helper.apply_dataframe_filter(
        df=raw_df_arithmetic_mean,
        filter=filter_arithmetic_mean
    )
    df_median = data_helper.apply_dataframe_filter(
        df=raw_df_median,
        filter=filter_median
    )
    df_nn = data_helper.apply_dataframe_filter(
        df=raw_df_nn,
        filter=filter_nn
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_arithmetic_mean["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Arithmetic mean - ensemble_size = 2
    mse_arithmetic_mean_2 = np.array(
        df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 2]["loss_test"])
    assert len(mse_arithmetic_mean_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_2 has length {len(mse_arithmetic_mean_2)}"
    axis.plot(data_x, mse_arithmetic_mean_2,
              marker="o", label=f"Arithmetic mean - ensemble_size: 2")

    # # Arithmetic mean - ensemble_size = 6
    # mse_arithmetic_mean_6 = np.array(
    #     df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 6]["loss_test"])
    # assert len(mse_arithmetic_mean_6) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_6 has length {len(mse_arithmetic_mean_6)}"
    # axis.plot(data_x, mse_arithmetic_mean_6,
    #           marker="o", label=f"Arithmetic mean - ensemble_size: 6")

    # # Arithmetic mean - ensemble_size = 12
    # mse_arithmetic_mean_12 = np.array(
    #     df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 12]["loss_test"])
    # assert len(mse_arithmetic_mean_12) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_12 has length {len(mse_arithmetic_mean_12)}"
    # axis.plot(data_x, mse_arithmetic_mean_12,
    #           marker="o", label=f"Arithmetic mean - ensemble_size: 12")

    # Median - ensemble_size = 2
    mse_median_2 = np.array(
        df_median[df_median["ensemble_size"] == 2]["loss_test"])
    assert len(mse_median_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_2 has length {len(mse_median_2)}"
    axis.plot(data_x, mse_median_2, marker="o",
              label=f"Median - ensemble_size: 2")

    # # Median - ensemble_size = 6
    # mse_median_6 = np.array(
    #     df_median[df_median["ensemble_size"] == 6]["loss_test"])
    # assert len(mse_median_6) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_6 has length {len(mse_median_6)}"
    # axis.plot(data_x, mse_median_6, marker="o",
    #           label=f"Median - ensemble_size: 6")

    # # Median - ensemble_size = 12
    # mse_median_12 = np.array(
    #     df_median[df_median["ensemble_size"] == 12]["loss_test"])
    # assert len(mse_median_12) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_12 has length {len(mse_median_12)}"
    # axis.plot(data_x, mse_median_12, marker="o",
    #           label=f"Median - ensemble_size: 12")

    # NN - ensemble_size = 2
    mse_nn_2 = np.array(df_nn[df_nn["ensemble_size"] == 2]["loss_test"])
    assert len(mse_nn_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_2 has length {len(mse_nn_2)}"
    axis.plot(data_x, mse_nn_2, marker="o",
              label=f"Neural network - ensemble_size: 2")

    # # NN - ensemble_size = 6
    # mse_nn_6 = np.array(df_nn[df_nn["ensemble_size"] == 6]["loss_test"])
    # assert len(mse_nn_6) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_6 has length {len(mse_nn_6)}"
    # axis.plot(data_x, mse_nn_6, marker="o", label=f"Neural network - 6")

    # # NN - ensemble_size = 12
    # mse_nn_12 = np.array(df_nn[df_nn["ensemble_size"] == 12]["loss_test"])
    # assert len(mse_nn_12) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_12 has length {len(mse_nn_12)}"
    # axis.plot(data_x, mse_nn_12, marker="o", label=f"Neural network - 12")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend(loc=f"upper center", bbox_to_anchor=(
        0.5, -0.25), ncol=1, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_2_2_2_2_2_fixed_epoch_num_400_fixed_ensemble_size_2(raw_df_arithmetic_mean: pd.DataFrame, raw_df_median: pd.DataFrame, raw_df_nn: pd.DataFrame, img_repository_path: str):
    assert raw_df_arithmetic_mean is not None and raw_df_median is not None and raw_df_nn is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_arithmetic_mean, raw_df_median, raw_df_nn, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_arithmetic_mean = {
        "hidden_size": [
            [2, 2, 2, 2, 2, 2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_median = {
        "hidden_size": [
            [2, 2, 2, 2, 2, 2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_nn = {
        "hidden_size": [
            [2, 2, 2, 2, 2, 2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(each base learner is [2, 2, 2, 2, 2, 2] trained with 400 epochs)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"voter_static_ncl_mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_2_2_2_2_2_fixed_epoch_num_400_fixed_ensemble_size_2.png"
    # ====================================================================

    df_arithmetic_mean = data_helper.apply_dataframe_filter(
        df=raw_df_arithmetic_mean,
        filter=filter_arithmetic_mean
    )
    df_median = data_helper.apply_dataframe_filter(
        df=raw_df_median,
        filter=filter_median
    )
    df_nn = data_helper.apply_dataframe_filter(
        df=raw_df_nn,
        filter=filter_nn
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_arithmetic_mean["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Arithmetic mean - ensemble_size = 2
    mse_arithmetic_mean_2 = np.array(
        df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 2]["loss_test"])
    assert len(mse_arithmetic_mean_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_2 has length {len(mse_arithmetic_mean_2)}"
    axis.plot(data_x, mse_arithmetic_mean_2,
              marker="o", label=f"Arithmetic mean - ensemble_size: 2")

    # # Arithmetic mean - ensemble_size = 6
    # mse_arithmetic_mean_6 = np.array(
    #     df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 6]["loss_test"])
    # assert len(mse_arithmetic_mean_6) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_6 has length {len(mse_arithmetic_mean_6)}"
    # axis.plot(data_x, mse_arithmetic_mean_6,
    #           marker="o", label=f"Arithmetic mean - ensemble_size: 6")

    # # Arithmetic mean - ensemble_size = 12
    # mse_arithmetic_mean_12 = np.array(
    #     df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 12]["loss_test"])
    # assert len(mse_arithmetic_mean_12) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_12 has length {len(mse_arithmetic_mean_12)}"
    # axis.plot(data_x, mse_arithmetic_mean_12,
    #           marker="o", label=f"Arithmetic mean - ensemble_size: 12")

    # Median - ensemble_size = 2
    mse_median_2 = np.array(
        df_median[df_median["ensemble_size"] == 2]["loss_test"])
    assert len(mse_median_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_2 has length {len(mse_median_2)}"
    axis.plot(data_x, mse_median_2, marker="o",
              label=f"Median - ensemble_size: 2")

    # # Median - ensemble_size = 6
    # mse_median_6 = np.array(
    #     df_median[df_median["ensemble_size"] == 6]["loss_test"])
    # assert len(mse_median_6) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_6 has length {len(mse_median_6)}"
    # axis.plot(data_x, mse_median_6, marker="o",
    #           label=f"Median - ensemble_size: 6")

    # # Median - ensemble_size = 12
    # mse_median_12 = np.array(
    #     df_median[df_median["ensemble_size"] == 12]["loss_test"])
    # assert len(mse_median_12) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_12 has length {len(mse_median_12)}"
    # axis.plot(data_x, mse_median_12, marker="o",
    #           label=f"Median - ensemble_size: 12")

    # NN - ensemble_size = 2
    mse_nn_2 = np.array(df_nn[df_nn["ensemble_size"] == 2]["loss_test"])
    assert len(mse_nn_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_2 has length {len(mse_nn_2)}"
    axis.plot(data_x, mse_nn_2, marker="o",
              label=f"Neural network - ensemble_size: 2")

    # # NN - ensemble_size = 6
    # mse_nn_6 = np.array(df_nn[df_nn["ensemble_size"] == 6]["loss_test"])
    # assert len(mse_nn_6) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_6 has length {len(mse_nn_6)}"
    # axis.plot(data_x, mse_nn_6, marker="o", label=f"Neural network - 6")

    # # NN - ensemble_size = 12
    # mse_nn_12 = np.array(df_nn[df_nn["ensemble_size"] == 12]["loss_test"])
    # assert len(mse_nn_12) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_12 has length {len(mse_nn_12)}"
    # axis.plot(data_x, mse_nn_12, marker="o", label=f"Neural network - 12")

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


def mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_6_fixed_epoch_num_400_fixed_ensemble_size_2(raw_df_arithmetic_mean: pd.DataFrame, raw_df_median: pd.DataFrame, raw_df_nn: pd.DataFrame, img_repository_path: str):
    assert raw_df_arithmetic_mean is not None and raw_df_median is not None and raw_df_nn is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_arithmetic_mean, raw_df_median, raw_df_nn, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_arithmetic_mean = {
        "hidden_size": [
            [6]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_median = {
        "hidden_size": [
            [6]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_nn = {
        "hidden_size": [
            [6]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(each base learner is [6] trained with 400 epochs)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"voter_static_ncl_mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_6_fixed_epoch_num_400_fixed_ensemble_size_2.png"
    # ====================================================================

    df_arithmetic_mean = data_helper.apply_dataframe_filter(
        df=raw_df_arithmetic_mean,
        filter=filter_arithmetic_mean
    )
    df_median = data_helper.apply_dataframe_filter(
        df=raw_df_median,
        filter=filter_median
    )
    df_nn = data_helper.apply_dataframe_filter(
        df=raw_df_nn,
        filter=filter_nn
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_arithmetic_mean["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Arithmetic mean - ensemble_size = 2
    mse_arithmetic_mean_2 = np.array(
        df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 2]["loss_test"])
    assert len(mse_arithmetic_mean_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_2 has length {len(mse_arithmetic_mean_2)}"
    axis.plot(data_x, mse_arithmetic_mean_2,
              marker="o", label=f"Arithmetic mean - ensemble_size: 2")

    # # Arithmetic mean - ensemble_size = 6
    # mse_arithmetic_mean_6 = np.array(
    #     df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 6]["loss_test"])
    # assert len(mse_arithmetic_mean_6) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_6 has length {len(mse_arithmetic_mean_6)}"
    # axis.plot(data_x, mse_arithmetic_mean_6,
    #           marker="o", label=f"Arithmetic mean - ensemble_size: 6")

    # # Arithmetic mean - ensemble_size = 12
    # mse_arithmetic_mean_12 = np.array(
    #     df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 12]["loss_test"])
    # assert len(mse_arithmetic_mean_12) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_12 has length {len(mse_arithmetic_mean_12)}"
    # axis.plot(data_x, mse_arithmetic_mean_12,
    #           marker="o", label=f"Arithmetic mean - ensemble_size: 12")

    # Median - ensemble_size = 2
    mse_median_2 = np.array(
        df_median[df_median["ensemble_size"] == 2]["loss_test"])
    assert len(mse_median_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_2 has length {len(mse_median_2)}"
    axis.plot(data_x, mse_median_2, marker="o",
              label=f"Median - ensemble_size: 2")

    # # Median - ensemble_size = 6
    # mse_median_6 = np.array(
    #     df_median[df_median["ensemble_size"] == 6]["loss_test"])
    # assert len(mse_median_6) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_6 has length {len(mse_median_6)}"
    # axis.plot(data_x, mse_median_6, marker="o",
    #           label=f"Median - ensemble_size: 6")

    # # Median - ensemble_size = 12
    # mse_median_12 = np.array(
    #     df_median[df_median["ensemble_size"] == 12]["loss_test"])
    # assert len(mse_median_12) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_12 has length {len(mse_median_12)}"
    # axis.plot(data_x, mse_median_12, marker="o",
    #           label=f"Median - ensemble_size: 12")

    # NN - ensemble_size = 2
    mse_nn_2 = np.array(df_nn[df_nn["ensemble_size"] == 2]["loss_test"])
    assert len(mse_nn_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_2 has length {len(mse_nn_2)}"
    axis.plot(data_x, mse_nn_2, marker="o",
              label=f"Neural network - ensemble_size: 2")

    # # NN - ensemble_size = 6
    # mse_nn_6 = np.array(df_nn[df_nn["ensemble_size"] == 6]["loss_test"])
    # assert len(mse_nn_6) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_6 has length {len(mse_nn_6)}"
    # axis.plot(data_x, mse_nn_6, marker="o", label=f"Neural network - 6")

    # # NN - ensemble_size = 12
    # mse_nn_12 = np.array(df_nn[df_nn["ensemble_size"] == 12]["loss_test"])
    # assert len(mse_nn_12) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_12 has length {len(mse_nn_12)}"
    # axis.plot(data_x, mse_nn_12, marker="o", label=f"Neural network - 12")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend(loc=f"upper center", bbox_to_anchor=(
        0.5, -0.25), ncol=1, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_12_fixed_epoch_num_400_fixed_ensemble_size_2(raw_df_arithmetic_mean: pd.DataFrame, raw_df_median: pd.DataFrame, raw_df_nn: pd.DataFrame, img_repository_path: str):
    assert raw_df_arithmetic_mean is not None and raw_df_median is not None and raw_df_nn is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_arithmetic_mean, raw_df_median, raw_df_nn, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_arithmetic_mean = {
        "hidden_size": [
            [12]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_median = {
        "hidden_size": [
            [12]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_nn = {
        "hidden_size": [
            [12]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(each base learner is [12] trained with 400 epochs)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"voter_static_ncl_mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_12_fixed_epoch_num_400_fixed_ensemble_size_2.png"
    # ====================================================================

    df_arithmetic_mean = data_helper.apply_dataframe_filter(
        df=raw_df_arithmetic_mean,
        filter=filter_arithmetic_mean
    )
    df_median = data_helper.apply_dataframe_filter(
        df=raw_df_median,
        filter=filter_median
    )
    df_nn = data_helper.apply_dataframe_filter(
        df=raw_df_nn,
        filter=filter_nn
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_arithmetic_mean["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Arithmetic mean - ensemble_size = 2
    mse_arithmetic_mean_2 = np.array(
        df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 2]["loss_test"])
    assert len(mse_arithmetic_mean_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_2 has length {len(mse_arithmetic_mean_2)}"
    axis.plot(data_x, mse_arithmetic_mean_2,
              marker="o", label=f"Arithmetic mean - ensemble_size: 2")

    # # Arithmetic mean - ensemble_size = 6
    # mse_arithmetic_mean_6 = np.array(
    #     df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 6]["loss_test"])
    # assert len(mse_arithmetic_mean_6) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_6 has length {len(mse_arithmetic_mean_6)}"
    # axis.plot(data_x, mse_arithmetic_mean_6,
    #           marker="o", label=f"Arithmetic mean - ensemble_size: 6")

    # # Arithmetic mean - ensemble_size = 12
    # mse_arithmetic_mean_12 = np.array(
    #     df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 12]["loss_test"])
    # assert len(mse_arithmetic_mean_12) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_12 has length {len(mse_arithmetic_mean_12)}"
    # axis.plot(data_x, mse_arithmetic_mean_12,
    #           marker="o", label=f"Arithmetic mean - ensemble_size: 12")

    # Median - ensemble_size = 2
    mse_median_2 = np.array(
        df_median[df_median["ensemble_size"] == 2]["loss_test"])
    assert len(mse_median_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_2 has length {len(mse_median_2)}"
    axis.plot(data_x, mse_median_2, marker="o",
              label=f"Median - ensemble_size: 2")

    # # Median - ensemble_size = 6
    # mse_median_6 = np.array(
    #     df_median[df_median["ensemble_size"] == 6]["loss_test"])
    # assert len(mse_median_6) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_6 has length {len(mse_median_6)}"
    # axis.plot(data_x, mse_median_6, marker="o",
    #           label=f"Median - ensemble_size: 6")

    # # Median - ensemble_size = 12
    # mse_median_12 = np.array(
    #     df_median[df_median["ensemble_size"] == 12]["loss_test"])
    # assert len(mse_median_12) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_12 has length {len(mse_median_12)}"
    # axis.plot(data_x, mse_median_12, marker="o",
    #           label=f"Median - ensemble_size: 12")

    # NN - ensemble_size = 2
    mse_nn_2 = np.array(df_nn[df_nn["ensemble_size"] == 2]["loss_test"])
    assert len(mse_nn_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_2 has length {len(mse_nn_2)}"
    axis.plot(data_x, mse_nn_2, marker="o",
              label=f"Neural network - ensemble_size: 2")

    # # NN - ensemble_size = 6
    # mse_nn_6 = np.array(df_nn[df_nn["ensemble_size"] == 6]["loss_test"])
    # assert len(mse_nn_6) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_6 has length {len(mse_nn_6)}"
    # axis.plot(data_x, mse_nn_6, marker="o", label=f"Neural network - 6")

    # # NN - ensemble_size = 12
    # mse_nn_12 = np.array(df_nn[df_nn["ensemble_size"] == 12]["loss_test"])
    # assert len(mse_nn_12) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_12 has length {len(mse_nn_12)}"
    # axis.plot(data_x, mse_nn_12, marker="o", label=f"Neural network - 12")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend(loc=f"upper center", bbox_to_anchor=(
        0.5, -0.25), ncol=1, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_fixed_epoch_num_400_fixed_ensemble_size_6(raw_df_arithmetic_mean: pd.DataFrame, raw_df_median: pd.DataFrame, raw_df_nn: pd.DataFrame, img_repository_path: str):
    assert raw_df_arithmetic_mean is not None and raw_df_median is not None and raw_df_nn is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_arithmetic_mean, raw_df_median, raw_df_nn, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_arithmetic_mean = {
        "hidden_size": [
            [2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_median = {
        "hidden_size": [
            [2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_nn = {
        "hidden_size": [
            [2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(each base learner is [2] trained with 400 epochs)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"voter_static_ncl_mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_fixed_epoch_num_400_fixed_ensemble_size_6.png"
    # ====================================================================

    df_arithmetic_mean = data_helper.apply_dataframe_filter(
        df=raw_df_arithmetic_mean,
        filter=filter_arithmetic_mean
    )
    df_median = data_helper.apply_dataframe_filter(
        df=raw_df_median,
        filter=filter_median
    )
    df_nn = data_helper.apply_dataframe_filter(
        df=raw_df_nn,
        filter=filter_nn
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_arithmetic_mean["correlation_penalty_coefficient"].unique())

    # Plot lines
    # # Arithmetic mean - ensemble_size = 2
    # mse_arithmetic_mean_2 = np.array(
    #     df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 2]["loss_test"])
    # assert len(mse_arithmetic_mean_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_2 has length {len(mse_arithmetic_mean_2)}"
    # axis.plot(data_x, mse_arithmetic_mean_2,
    #           marker="o", label=f"Arithmetic mean - ensemble_size: 2")

    # Arithmetic mean - ensemble_size = 6
    mse_arithmetic_mean_6 = np.array(
        df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 6]["loss_test"])
    assert len(mse_arithmetic_mean_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_6 has length {len(mse_arithmetic_mean_6)}"
    axis.plot(data_x, mse_arithmetic_mean_6,
              marker="o", label=f"Arithmetic mean - ensemble_size: 6")

    # # Arithmetic mean - ensemble_size = 12
    # mse_arithmetic_mean_12 = np.array(
    #     df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 12]["loss_test"])
    # assert len(mse_arithmetic_mean_12) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_12 has length {len(mse_arithmetic_mean_12)}"
    # axis.plot(data_x, mse_arithmetic_mean_12,
    #           marker="o", label=f"Arithmetic mean - ensemble_size: 12")

    # # Median - ensemble_size = 2
    # mse_median_2 = np.array(
    #     df_median[df_median["ensemble_size"] == 2]["loss_test"])
    # assert len(mse_median_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_2 has length {len(mse_median_2)}"
    # axis.plot(data_x, mse_median_2, marker="o",
    #           label=f"Median - ensemble_size: 2")

    # Median - ensemble_size = 6
    mse_median_6 = np.array(
        df_median[df_median["ensemble_size"] == 6]["loss_test"])
    assert len(mse_median_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_6 has length {len(mse_median_6)}"
    axis.plot(data_x, mse_median_6, marker="o",
              label=f"Median - ensemble_size: 6")

    # # Median - ensemble_size = 12
    # mse_median_12 = np.array(
    #     df_median[df_median["ensemble_size"] == 12]["loss_test"])
    # assert len(mse_median_12) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_12 has length {len(mse_median_12)}"
    # axis.plot(data_x, mse_median_12, marker="o",
    #           label=f"Median - ensemble_size: 12")

    # # NN - ensemble_size = 2
    # mse_nn_2 = np.array(df_nn[df_nn["ensemble_size"] == 2]["loss_test"])
    # assert len(mse_nn_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_2 has length {len(mse_nn_2)}"
    # axis.plot(data_x, mse_nn_2, marker="o",
    #           label=f"Neural network - ensemble_size: 2")

    # NN - ensemble_size = 6
    mse_nn_6 = np.array(df_nn[df_nn["ensemble_size"] == 6]["loss_test"])
    assert len(mse_nn_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_6 has length {len(mse_nn_6)}"
    axis.plot(data_x, mse_nn_6, marker="o", label=f"Neural network - 6")

    # # NN - ensemble_size = 12
    # mse_nn_12 = np.array(df_nn[df_nn["ensemble_size"] == 12]["loss_test"])
    # assert len(mse_nn_12) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_12 has length {len(mse_nn_12)}"
    # axis.plot(data_x, mse_nn_12, marker="o", label=f"Neural network - 12")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend(loc=f"upper center", bbox_to_anchor=(
        0.5, -0.25), ncol=1, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_2_2_fixed_epoch_num_400_fixed_ensemble_size_6(raw_df_arithmetic_mean: pd.DataFrame, raw_df_median: pd.DataFrame, raw_df_nn: pd.DataFrame, img_repository_path: str):
    assert raw_df_arithmetic_mean is not None and raw_df_median is not None and raw_df_nn is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_arithmetic_mean, raw_df_median, raw_df_nn, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_arithmetic_mean = {
        "hidden_size": [
            [2, 2, 2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_median = {
        "hidden_size": [
            [2, 2, 2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_nn = {
        "hidden_size": [
            [2, 2, 2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(each base learner is [2, 2, 2] trained with 400 epochs)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"voter_static_ncl_mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_2_2_fixed_epoch_num_400_fixed_ensemble_size_6.png"
    # ====================================================================

    df_arithmetic_mean = data_helper.apply_dataframe_filter(
        df=raw_df_arithmetic_mean,
        filter=filter_arithmetic_mean
    )
    df_median = data_helper.apply_dataframe_filter(
        df=raw_df_median,
        filter=filter_median
    )
    df_nn = data_helper.apply_dataframe_filter(
        df=raw_df_nn,
        filter=filter_nn
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_arithmetic_mean["correlation_penalty_coefficient"].unique())

    # Plot lines
    # # Arithmetic mean - ensemble_size = 2
    # mse_arithmetic_mean_2 = np.array(
    #     df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 2]["loss_test"])
    # assert len(mse_arithmetic_mean_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_2 has length {len(mse_arithmetic_mean_2)}"
    # axis.plot(data_x, mse_arithmetic_mean_2,
    #           marker="o", label=f"Arithmetic mean - ensemble_size: 2")

    # Arithmetic mean - ensemble_size = 6
    mse_arithmetic_mean_6 = np.array(
        df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 6]["loss_test"])
    assert len(mse_arithmetic_mean_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_6 has length {len(mse_arithmetic_mean_6)}"
    axis.plot(data_x, mse_arithmetic_mean_6,
              marker="o", label=f"Arithmetic mean - ensemble_size: 6")

    # # Arithmetic mean - ensemble_size = 12
    # mse_arithmetic_mean_12 = np.array(
    #     df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 12]["loss_test"])
    # assert len(mse_arithmetic_mean_12) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_12 has length {len(mse_arithmetic_mean_12)}"
    # axis.plot(data_x, mse_arithmetic_mean_12,
    #           marker="o", label=f"Arithmetic mean - ensemble_size: 12")

    # # Median - ensemble_size = 2
    # mse_median_2 = np.array(
    #     df_median[df_median["ensemble_size"] == 2]["loss_test"])
    # assert len(mse_median_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_2 has length {len(mse_median_2)}"
    # axis.plot(data_x, mse_median_2, marker="o",
    #           label=f"Median - ensemble_size: 2")

    # Median - ensemble_size = 6
    mse_median_6 = np.array(
        df_median[df_median["ensemble_size"] == 6]["loss_test"])
    assert len(mse_median_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_6 has length {len(mse_median_6)}"
    axis.plot(data_x, mse_median_6, marker="o",
              label=f"Median - ensemble_size: 6")

    # # Median - ensemble_size = 12
    # mse_median_12 = np.array(
    #     df_median[df_median["ensemble_size"] == 12]["loss_test"])
    # assert len(mse_median_12) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_12 has length {len(mse_median_12)}"
    # axis.plot(data_x, mse_median_12, marker="o",
    #           label=f"Median - ensemble_size: 12")

    # # NN - ensemble_size = 2
    # mse_nn_2 = np.array(df_nn[df_nn["ensemble_size"] == 2]["loss_test"])
    # assert len(mse_nn_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_2 has length {len(mse_nn_2)}"
    # axis.plot(data_x, mse_nn_2, marker="o",
    #           label=f"Neural network - ensemble_size: 2")

    # NN - ensemble_size = 6
    mse_nn_6 = np.array(df_nn[df_nn["ensemble_size"] == 6]["loss_test"])
    assert len(mse_nn_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_6 has length {len(mse_nn_6)}"
    axis.plot(data_x, mse_nn_6, marker="o", label=f"Neural network - 6")

    # # NN - ensemble_size = 12
    # mse_nn_12 = np.array(df_nn[df_nn["ensemble_size"] == 12]["loss_test"])
    # assert len(mse_nn_12) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_12 has length {len(mse_nn_12)}"
    # axis.plot(data_x, mse_nn_12, marker="o", label=f"Neural network - 12")

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


def mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_2_2_2_2_2_fixed_epoch_num_400_fixed_ensemble_size_6(raw_df_arithmetic_mean: pd.DataFrame, raw_df_median: pd.DataFrame, raw_df_nn: pd.DataFrame, img_repository_path: str):
    assert raw_df_arithmetic_mean is not None and raw_df_median is not None and raw_df_nn is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_arithmetic_mean, raw_df_median, raw_df_nn, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_arithmetic_mean = {
        "hidden_size": [
            [2, 2, 2, 2, 2, 2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_median = {
        "hidden_size": [
            [2, 2, 2, 2, 2, 2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_nn = {
        "hidden_size": [
            [2, 2, 2, 2, 2, 2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(each base learner is [2, 2, 2, 2, 2, 2] trained with 400 epochs)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"voter_static_ncl_mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_2_2_2_2_2_fixed_epoch_num_400_fixed_ensemble_size_6.png"
    # ====================================================================

    df_arithmetic_mean = data_helper.apply_dataframe_filter(
        df=raw_df_arithmetic_mean,
        filter=filter_arithmetic_mean
    )
    df_median = data_helper.apply_dataframe_filter(
        df=raw_df_median,
        filter=filter_median
    )
    df_nn = data_helper.apply_dataframe_filter(
        df=raw_df_nn,
        filter=filter_nn
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_arithmetic_mean["correlation_penalty_coefficient"].unique())

    # Plot lines
    # # Arithmetic mean - ensemble_size = 2
    # mse_arithmetic_mean_2 = np.array(
    #     df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 2]["loss_test"])
    # assert len(mse_arithmetic_mean_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_2 has length {len(mse_arithmetic_mean_2)}"
    # axis.plot(data_x, mse_arithmetic_mean_2,
    #           marker="o", label=f"Arithmetic mean - ensemble_size: 2")

    # Arithmetic mean - ensemble_size = 6
    mse_arithmetic_mean_6 = np.array(
        df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 6]["loss_test"])
    assert len(mse_arithmetic_mean_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_6 has length {len(mse_arithmetic_mean_6)}"
    axis.plot(data_x, mse_arithmetic_mean_6,
              marker="o", label=f"Arithmetic mean - ensemble_size: 6")

    # # Arithmetic mean - ensemble_size = 12
    # mse_arithmetic_mean_12 = np.array(
    #     df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 12]["loss_test"])
    # assert len(mse_arithmetic_mean_12) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_12 has length {len(mse_arithmetic_mean_12)}"
    # axis.plot(data_x, mse_arithmetic_mean_12,
    #           marker="o", label=f"Arithmetic mean - ensemble_size: 12")

    # # Median - ensemble_size = 2
    # mse_median_2 = np.array(
    #     df_median[df_median["ensemble_size"] == 2]["loss_test"])
    # assert len(mse_median_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_2 has length {len(mse_median_2)}"
    # axis.plot(data_x, mse_median_2, marker="o",
    #           label=f"Median - ensemble_size: 2")

    # Median - ensemble_size = 6
    mse_median_6 = np.array(
        df_median[df_median["ensemble_size"] == 6]["loss_test"])
    assert len(mse_median_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_6 has length {len(mse_median_6)}"
    axis.plot(data_x, mse_median_6, marker="o",
              label=f"Median - ensemble_size: 6")

    # # Median - ensemble_size = 12
    # mse_median_12 = np.array(
    #     df_median[df_median["ensemble_size"] == 12]["loss_test"])
    # assert len(mse_median_12) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_12 has length {len(mse_median_12)}"
    # axis.plot(data_x, mse_median_12, marker="o",
    #           label=f"Median - ensemble_size: 12")

    # # NN - ensemble_size = 2
    # mse_nn_2 = np.array(df_nn[df_nn["ensemble_size"] == 2]["loss_test"])
    # assert len(mse_nn_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_2 has length {len(mse_nn_2)}"
    # axis.plot(data_x, mse_nn_2, marker="o",
    #           label=f"Neural network - ensemble_size: 2")

    # NN - ensemble_size = 6
    mse_nn_6 = np.array(df_nn[df_nn["ensemble_size"] == 6]["loss_test"])
    assert len(mse_nn_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_6 has length {len(mse_nn_6)}"
    axis.plot(data_x, mse_nn_6, marker="o", label=f"Neural network - 6")

    # # NN - ensemble_size = 12
    # mse_nn_12 = np.array(df_nn[df_nn["ensemble_size"] == 12]["loss_test"])
    # assert len(mse_nn_12) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_12 has length {len(mse_nn_12)}"
    # axis.plot(data_x, mse_nn_12, marker="o", label=f"Neural network - 12")

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


def mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_6_fixed_epoch_num_400_fixed_ensemble_size_6(raw_df_arithmetic_mean: pd.DataFrame, raw_df_median: pd.DataFrame, raw_df_nn: pd.DataFrame, img_repository_path: str):
    assert raw_df_arithmetic_mean is not None and raw_df_median is not None and raw_df_nn is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_arithmetic_mean, raw_df_median, raw_df_nn, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_arithmetic_mean = {
        "hidden_size": [
            [6]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_median = {
        "hidden_size": [
            [6]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_nn = {
        "hidden_size": [
            [6]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(each base learner is [6] trained with 400 epochs)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"voter_static_ncl_mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_6_fixed_epoch_num_400_fixed_ensemble_size_6.png"
    # ====================================================================

    df_arithmetic_mean = data_helper.apply_dataframe_filter(
        df=raw_df_arithmetic_mean,
        filter=filter_arithmetic_mean
    )
    df_median = data_helper.apply_dataframe_filter(
        df=raw_df_median,
        filter=filter_median
    )
    df_nn = data_helper.apply_dataframe_filter(
        df=raw_df_nn,
        filter=filter_nn
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_arithmetic_mean["correlation_penalty_coefficient"].unique())

    # Plot lines
    # # Arithmetic mean - ensemble_size = 2
    # mse_arithmetic_mean_2 = np.array(
    #     df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 2]["loss_test"])
    # assert len(mse_arithmetic_mean_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_2 has length {len(mse_arithmetic_mean_2)}"
    # axis.plot(data_x, mse_arithmetic_mean_2,
    #           marker="o", label=f"Arithmetic mean - ensemble_size: 2")

    # Arithmetic mean - ensemble_size = 6
    mse_arithmetic_mean_6 = np.array(
        df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 6]["loss_test"])
    assert len(mse_arithmetic_mean_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_6 has length {len(mse_arithmetic_mean_6)}"
    axis.plot(data_x, mse_arithmetic_mean_6,
              marker="o", label=f"Arithmetic mean - ensemble_size: 6")

    # # Arithmetic mean - ensemble_size = 12
    # mse_arithmetic_mean_12 = np.array(
    #     df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 12]["loss_test"])
    # assert len(mse_arithmetic_mean_12) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_12 has length {len(mse_arithmetic_mean_12)}"
    # axis.plot(data_x, mse_arithmetic_mean_12,
    #           marker="o", label=f"Arithmetic mean - ensemble_size: 12")

    # # Median - ensemble_size = 2
    # mse_median_2 = np.array(
    #     df_median[df_median["ensemble_size"] == 2]["loss_test"])
    # assert len(mse_median_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_2 has length {len(mse_median_2)}"
    # axis.plot(data_x, mse_median_2, marker="o",
    #           label=f"Median - ensemble_size: 2")

    # Median - ensemble_size = 6
    mse_median_6 = np.array(
        df_median[df_median["ensemble_size"] == 6]["loss_test"])
    assert len(mse_median_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_6 has length {len(mse_median_6)}"
    axis.plot(data_x, mse_median_6, marker="o",
              label=f"Median - ensemble_size: 6")

    # # Median - ensemble_size = 12
    # mse_median_12 = np.array(
    #     df_median[df_median["ensemble_size"] == 12]["loss_test"])
    # assert len(mse_median_12) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_12 has length {len(mse_median_12)}"
    # axis.plot(data_x, mse_median_12, marker="o",
    #           label=f"Median - ensemble_size: 12")

    # # NN - ensemble_size = 2
    # mse_nn_2 = np.array(df_nn[df_nn["ensemble_size"] == 2]["loss_test"])
    # assert len(mse_nn_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_2 has length {len(mse_nn_2)}"
    # axis.plot(data_x, mse_nn_2, marker="o",
    #           label=f"Neural network - ensemble_size: 2")

    # NN - ensemble_size = 6
    mse_nn_6 = np.array(df_nn[df_nn["ensemble_size"] == 6]["loss_test"])
    assert len(mse_nn_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_6 has length {len(mse_nn_6)}"
    axis.plot(data_x, mse_nn_6, marker="o", label=f"Neural network - 6")

    # # NN - ensemble_size = 12
    # mse_nn_12 = np.array(df_nn[df_nn["ensemble_size"] == 12]["loss_test"])
    # assert len(mse_nn_12) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_12 has length {len(mse_nn_12)}"
    # axis.plot(data_x, mse_nn_12, marker="o", label=f"Neural network - 12")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend(loc=f"upper center", bbox_to_anchor=(
        0.5, -0.25), ncol=1, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_12_fixed_epoch_num_400_fixed_ensemble_size_6(raw_df_arithmetic_mean: pd.DataFrame, raw_df_median: pd.DataFrame, raw_df_nn: pd.DataFrame, img_repository_path: str):
    assert raw_df_arithmetic_mean is not None and raw_df_median is not None and raw_df_nn is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_arithmetic_mean, raw_df_median, raw_df_nn, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_arithmetic_mean = {
        "hidden_size": [
            [12]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_median = {
        "hidden_size": [
            [12]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_nn = {
        "hidden_size": [
            [12]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(each base learner is [12] trained with 400 epochs)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"voter_static_ncl_mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_12_fixed_epoch_num_400_fixed_ensemble_size_6.png"
    # ====================================================================

    df_arithmetic_mean = data_helper.apply_dataframe_filter(
        df=raw_df_arithmetic_mean,
        filter=filter_arithmetic_mean
    )
    df_median = data_helper.apply_dataframe_filter(
        df=raw_df_median,
        filter=filter_median
    )
    df_nn = data_helper.apply_dataframe_filter(
        df=raw_df_nn,
        filter=filter_nn
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_arithmetic_mean["correlation_penalty_coefficient"].unique())

    # Plot lines
    # # Arithmetic mean - ensemble_size = 2
    # mse_arithmetic_mean_2 = np.array(
    #     df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 2]["loss_test"])
    # assert len(mse_arithmetic_mean_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_2 has length {len(mse_arithmetic_mean_2)}"
    # axis.plot(data_x, mse_arithmetic_mean_2,
    #           marker="o", label=f"Arithmetic mean - ensemble_size: 2")

    # Arithmetic mean - ensemble_size = 6
    mse_arithmetic_mean_6 = np.array(
        df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 6]["loss_test"])
    assert len(mse_arithmetic_mean_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_6 has length {len(mse_arithmetic_mean_6)}"
    axis.plot(data_x, mse_arithmetic_mean_6,
              marker="o", label=f"Arithmetic mean - ensemble_size: 6")

    # # Arithmetic mean - ensemble_size = 12
    # mse_arithmetic_mean_12 = np.array(
    #     df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 12]["loss_test"])
    # assert len(mse_arithmetic_mean_12) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_12 has length {len(mse_arithmetic_mean_12)}"
    # axis.plot(data_x, mse_arithmetic_mean_12,
    #           marker="o", label=f"Arithmetic mean - ensemble_size: 12")

    # # Median - ensemble_size = 2
    # mse_median_2 = np.array(
    #     df_median[df_median["ensemble_size"] == 2]["loss_test"])
    # assert len(mse_median_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_2 has length {len(mse_median_2)}"
    # axis.plot(data_x, mse_median_2, marker="o",
    #           label=f"Median - ensemble_size: 2")

    # Median - ensemble_size = 6
    mse_median_6 = np.array(
        df_median[df_median["ensemble_size"] == 6]["loss_test"])
    assert len(mse_median_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_6 has length {len(mse_median_6)}"
    axis.plot(data_x, mse_median_6, marker="o",
              label=f"Median - ensemble_size: 6")

    # # Median - ensemble_size = 12
    # mse_median_12 = np.array(
    #     df_median[df_median["ensemble_size"] == 12]["loss_test"])
    # assert len(mse_median_12) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_12 has length {len(mse_median_12)}"
    # axis.plot(data_x, mse_median_12, marker="o",
    #           label=f"Median - ensemble_size: 12")

    # # NN - ensemble_size = 2
    # mse_nn_2 = np.array(df_nn[df_nn["ensemble_size"] == 2]["loss_test"])
    # assert len(mse_nn_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_2 has length {len(mse_nn_2)}"
    # axis.plot(data_x, mse_nn_2, marker="o",
    #           label=f"Neural network - ensemble_size: 2")

    # NN - ensemble_size = 6
    mse_nn_6 = np.array(df_nn[df_nn["ensemble_size"] == 6]["loss_test"])
    assert len(mse_nn_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_6 has length {len(mse_nn_6)}"
    axis.plot(data_x, mse_nn_6, marker="o", label=f"Neural network - 6")

    # # NN - ensemble_size = 12
    # mse_nn_12 = np.array(df_nn[df_nn["ensemble_size"] == 12]["loss_test"])
    # assert len(mse_nn_12) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_12 has length {len(mse_nn_12)}"
    # axis.plot(data_x, mse_nn_12, marker="o", label=f"Neural network - 12")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend(loc=f"upper center", bbox_to_anchor=(
        0.5, -0.25), ncol=1, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_fixed_epoch_num_400_fixed_ensemble_size_12(raw_df_arithmetic_mean: pd.DataFrame, raw_df_median: pd.DataFrame, raw_df_nn: pd.DataFrame, img_repository_path: str):
    assert raw_df_arithmetic_mean is not None and raw_df_median is not None and raw_df_nn is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_arithmetic_mean, raw_df_median, raw_df_nn, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_arithmetic_mean = {
        "hidden_size": [
            [2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_median = {
        "hidden_size": [
            [2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_nn = {
        "hidden_size": [
            [2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(each base learner is [2] trained with 400 epochs)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"voter_static_ncl_mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_fixed_epoch_num_400_fixed_ensemble_size_12.png"
    # ====================================================================

    df_arithmetic_mean = data_helper.apply_dataframe_filter(
        df=raw_df_arithmetic_mean,
        filter=filter_arithmetic_mean
    )
    df_median = data_helper.apply_dataframe_filter(
        df=raw_df_median,
        filter=filter_median
    )
    df_nn = data_helper.apply_dataframe_filter(
        df=raw_df_nn,
        filter=filter_nn
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_arithmetic_mean["correlation_penalty_coefficient"].unique())

    # Plot lines
    # # Arithmetic mean - ensemble_size = 2
    # mse_arithmetic_mean_2 = np.array(
    #     df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 2]["loss_test"])
    # assert len(mse_arithmetic_mean_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_2 has length {len(mse_arithmetic_mean_2)}"
    # axis.plot(data_x, mse_arithmetic_mean_2,
    #           marker="o", label=f"Arithmetic mean - ensemble_size: 2")

    # # Arithmetic mean - ensemble_size = 6
    # mse_arithmetic_mean_6 = np.array(
    #     df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 6]["loss_test"])
    # assert len(mse_arithmetic_mean_6) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_6 has length {len(mse_arithmetic_mean_6)}"
    # axis.plot(data_x, mse_arithmetic_mean_6,
    #           marker="o", label=f"Arithmetic mean - ensemble_size: 6")

    # Arithmetic mean - ensemble_size = 12
    mse_arithmetic_mean_12 = np.array(
        df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 12]["loss_test"])
    assert len(mse_arithmetic_mean_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_12 has length {len(mse_arithmetic_mean_12)}"
    axis.plot(data_x, mse_arithmetic_mean_12,
              marker="o", label=f"Arithmetic mean - ensemble_size: 12")

    # # Median - ensemble_size = 2
    # mse_median_2 = np.array(
    #     df_median[df_median["ensemble_size"] == 2]["loss_test"])
    # assert len(mse_median_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_2 has length {len(mse_median_2)}"
    # axis.plot(data_x, mse_median_2, marker="o",
    #           label=f"Median - ensemble_size: 2")

    # # Median - ensemble_size = 6
    # mse_median_6 = np.array(
    #     df_median[df_median["ensemble_size"] == 6]["loss_test"])
    # assert len(mse_median_6) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_6 has length {len(mse_median_6)}"
    # axis.plot(data_x, mse_median_6, marker="o",
    #           label=f"Median - ensemble_size: 6")

    # Median - ensemble_size = 12
    mse_median_12 = np.array(
        df_median[df_median["ensemble_size"] == 12]["loss_test"])
    assert len(mse_median_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_12 has length {len(mse_median_12)}"
    axis.plot(data_x, mse_median_12, marker="o",
              label=f"Median - ensemble_size: 12")

    # # NN - ensemble_size = 2
    # mse_nn_2 = np.array(df_nn[df_nn["ensemble_size"] == 2]["loss_test"])
    # assert len(mse_nn_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_2 has length {len(mse_nn_2)}"
    # axis.plot(data_x, mse_nn_2, marker="o",
    #           label=f"Neural network - ensemble_size: 2")

    # # NN - ensemble_size = 6
    # mse_nn_6 = np.array(df_nn[df_nn["ensemble_size"] == 6]["loss_test"])
    # assert len(mse_nn_6) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_6 has length {len(mse_nn_6)}"
    # axis.plot(data_x, mse_nn_6, marker="o", label=f"Neural network - 6")

    # NN - ensemble_size = 12
    mse_nn_12 = np.array(df_nn[df_nn["ensemble_size"] == 12]["loss_test"])
    assert len(mse_nn_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_12 has length {len(mse_nn_12)}"
    axis.plot(data_x, mse_nn_12, marker="o", label=f"Neural network - 12")

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


def mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_2_2_fixed_epoch_num_400_fixed_ensemble_size_12(raw_df_arithmetic_mean: pd.DataFrame, raw_df_median: pd.DataFrame, raw_df_nn: pd.DataFrame, img_repository_path: str):
    assert raw_df_arithmetic_mean is not None and raw_df_median is not None and raw_df_nn is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_arithmetic_mean, raw_df_median, raw_df_nn, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_arithmetic_mean = {
        "hidden_size": [
            [2, 2, 2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_median = {
        "hidden_size": [
            [2, 2, 2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_nn = {
        "hidden_size": [
            [2, 2, 2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(each base learner is [2, 2, 2] trained with 400 epochs)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"voter_static_ncl_mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_2_2_fixed_epoch_num_400_fixed_ensemble_size_12.png"
    # ====================================================================

    df_arithmetic_mean = data_helper.apply_dataframe_filter(
        df=raw_df_arithmetic_mean,
        filter=filter_arithmetic_mean
    )
    df_median = data_helper.apply_dataframe_filter(
        df=raw_df_median,
        filter=filter_median
    )
    df_nn = data_helper.apply_dataframe_filter(
        df=raw_df_nn,
        filter=filter_nn
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_arithmetic_mean["correlation_penalty_coefficient"].unique())

    # Plot lines
    # # Arithmetic mean - ensemble_size = 2
    # mse_arithmetic_mean_2 = np.array(
    #     df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 2]["loss_test"])
    # assert len(mse_arithmetic_mean_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_2 has length {len(mse_arithmetic_mean_2)}"
    # axis.plot(data_x, mse_arithmetic_mean_2,
    #           marker="o", label=f"Arithmetic mean - ensemble_size: 2")

    # # Arithmetic mean - ensemble_size = 6
    # mse_arithmetic_mean_6 = np.array(
    #     df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 6]["loss_test"])
    # assert len(mse_arithmetic_mean_6) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_6 has length {len(mse_arithmetic_mean_6)}"
    # axis.plot(data_x, mse_arithmetic_mean_6,
    #           marker="o", label=f"Arithmetic mean - ensemble_size: 6")

    # Arithmetic mean - ensemble_size = 12
    mse_arithmetic_mean_12 = np.array(
        df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 12]["loss_test"])
    assert len(mse_arithmetic_mean_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_12 has length {len(mse_arithmetic_mean_12)}"
    axis.plot(data_x, mse_arithmetic_mean_12,
              marker="o", label=f"Arithmetic mean - ensemble_size: 12")

    # # Median - ensemble_size = 2
    # mse_median_2 = np.array(
    #     df_median[df_median["ensemble_size"] == 2]["loss_test"])
    # assert len(mse_median_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_2 has length {len(mse_median_2)}"
    # axis.plot(data_x, mse_median_2, marker="o",
    #           label=f"Median - ensemble_size: 2")

    # # Median - ensemble_size = 6
    # mse_median_6 = np.array(
    #     df_median[df_median["ensemble_size"] == 6]["loss_test"])
    # assert len(mse_median_6) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_6 has length {len(mse_median_6)}"
    # axis.plot(data_x, mse_median_6, marker="o",
    #           label=f"Median - ensemble_size: 6")

    # Median - ensemble_size = 12
    mse_median_12 = np.array(
        df_median[df_median["ensemble_size"] == 12]["loss_test"])
    assert len(mse_median_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_12 has length {len(mse_median_12)}"
    axis.plot(data_x, mse_median_12, marker="o",
              label=f"Median - ensemble_size: 12")

    # # NN - ensemble_size = 2
    # mse_nn_2 = np.array(df_nn[df_nn["ensemble_size"] == 2]["loss_test"])
    # assert len(mse_nn_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_2 has length {len(mse_nn_2)}"
    # axis.plot(data_x, mse_nn_2, marker="o",
    #           label=f"Neural network - ensemble_size: 2")

    # # NN - ensemble_size = 6
    # mse_nn_6 = np.array(df_nn[df_nn["ensemble_size"] == 6]["loss_test"])
    # assert len(mse_nn_6) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_6 has length {len(mse_nn_6)}"
    # axis.plot(data_x, mse_nn_6, marker="o", label=f"Neural network - 6")

    # NN - ensemble_size = 12
    mse_nn_12 = np.array(df_nn[df_nn["ensemble_size"] == 12]["loss_test"])
    assert len(mse_nn_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_12 has length {len(mse_nn_12)}"
    axis.plot(data_x, mse_nn_12, marker="o", label=f"Neural network - 12")

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


def mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_2_2_2_2_2_fixed_epoch_num_400_fixed_ensemble_size_12(raw_df_arithmetic_mean: pd.DataFrame, raw_df_median: pd.DataFrame, raw_df_nn: pd.DataFrame, img_repository_path: str):
    assert raw_df_arithmetic_mean is not None and raw_df_median is not None and raw_df_nn is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_arithmetic_mean, raw_df_median, raw_df_nn, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_arithmetic_mean = {
        "hidden_size": [
            [2, 2, 2, 2, 2, 2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_median = {
        "hidden_size": [
            [2, 2, 2, 2, 2, 2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_nn = {
        "hidden_size": [
            [2, 2, 2, 2, 2, 2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(each base learner is [2, 2, 2, 2, 2, 2] trained with 400 epochs)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"voter_static_ncl_mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_2_2_2_2_2_fixed_epoch_num_400_fixed_ensemble_size_12.png"
    # ====================================================================

    df_arithmetic_mean = data_helper.apply_dataframe_filter(
        df=raw_df_arithmetic_mean,
        filter=filter_arithmetic_mean
    )
    df_median = data_helper.apply_dataframe_filter(
        df=raw_df_median,
        filter=filter_median
    )
    df_nn = data_helper.apply_dataframe_filter(
        df=raw_df_nn,
        filter=filter_nn
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_arithmetic_mean["correlation_penalty_coefficient"].unique())

    # Plot lines
    # # Arithmetic mean - ensemble_size = 2
    # mse_arithmetic_mean_2 = np.array(
    #     df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 2]["loss_test"])
    # assert len(mse_arithmetic_mean_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_2 has length {len(mse_arithmetic_mean_2)}"
    # axis.plot(data_x, mse_arithmetic_mean_2,
    #           marker="o", label=f"Arithmetic mean - ensemble_size: 2")

    # # Arithmetic mean - ensemble_size = 6
    # mse_arithmetic_mean_6 = np.array(
    #     df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 6]["loss_test"])
    # assert len(mse_arithmetic_mean_6) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_6 has length {len(mse_arithmetic_mean_6)}"
    # axis.plot(data_x, mse_arithmetic_mean_6,
    #           marker="o", label=f"Arithmetic mean - ensemble_size: 6")

    # Arithmetic mean - ensemble_size = 12
    mse_arithmetic_mean_12 = np.array(
        df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 12]["loss_test"])
    assert len(mse_arithmetic_mean_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_12 has length {len(mse_arithmetic_mean_12)}"
    axis.plot(data_x, mse_arithmetic_mean_12,
              marker="o", label=f"Arithmetic mean - ensemble_size: 12")

    # # Median - ensemble_size = 2
    # mse_median_2 = np.array(
    #     df_median[df_median["ensemble_size"] == 2]["loss_test"])
    # assert len(mse_median_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_2 has length {len(mse_median_2)}"
    # axis.plot(data_x, mse_median_2, marker="o",
    #           label=f"Median - ensemble_size: 2")

    # # Median - ensemble_size = 6
    # mse_median_6 = np.array(
    #     df_median[df_median["ensemble_size"] == 6]["loss_test"])
    # assert len(mse_median_6) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_6 has length {len(mse_median_6)}"
    # axis.plot(data_x, mse_median_6, marker="o",
    #           label=f"Median - ensemble_size: 6")

    # Median - ensemble_size = 12
    mse_median_12 = np.array(
        df_median[df_median["ensemble_size"] == 12]["loss_test"])
    assert len(mse_median_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_12 has length {len(mse_median_12)}"
    axis.plot(data_x, mse_median_12, marker="o",
              label=f"Median - ensemble_size: 12")

    # # NN - ensemble_size = 2
    # mse_nn_2 = np.array(df_nn[df_nn["ensemble_size"] == 2]["loss_test"])
    # assert len(mse_nn_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_2 has length {len(mse_nn_2)}"
    # axis.plot(data_x, mse_nn_2, marker="o",
    #           label=f"Neural network - ensemble_size: 2")

    # # NN - ensemble_size = 6
    # mse_nn_6 = np.array(df_nn[df_nn["ensemble_size"] == 6]["loss_test"])
    # assert len(mse_nn_6) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_6 has length {len(mse_nn_6)}"
    # axis.plot(data_x, mse_nn_6, marker="o", label=f"Neural network - 6")

    # NN - ensemble_size = 12
    mse_nn_12 = np.array(df_nn[df_nn["ensemble_size"] == 12]["loss_test"])
    assert len(mse_nn_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_12 has length {len(mse_nn_12)}"
    axis.plot(data_x, mse_nn_12, marker="o", label=f"Neural network - 12")

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


def mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_6_fixed_epoch_num_400_fixed_ensemble_size_12(raw_df_arithmetic_mean: pd.DataFrame, raw_df_median: pd.DataFrame, raw_df_nn: pd.DataFrame, img_repository_path: str):
    assert raw_df_arithmetic_mean is not None and raw_df_median is not None and raw_df_nn is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_arithmetic_mean, raw_df_median, raw_df_nn, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_arithmetic_mean = {
        "hidden_size": [
            [6]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_median = {
        "hidden_size": [
            [6]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_nn = {
        "hidden_size": [
            [6]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(each base learner is [6] trained with 400 epochs)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"voter_static_ncl_mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_6_fixed_epoch_num_400_fixed_ensemble_size_12.png"
    # ====================================================================

    df_arithmetic_mean = data_helper.apply_dataframe_filter(
        df=raw_df_arithmetic_mean,
        filter=filter_arithmetic_mean
    )
    df_median = data_helper.apply_dataframe_filter(
        df=raw_df_median,
        filter=filter_median
    )
    df_nn = data_helper.apply_dataframe_filter(
        df=raw_df_nn,
        filter=filter_nn
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_arithmetic_mean["correlation_penalty_coefficient"].unique())

    # Plot lines
    # # Arithmetic mean - ensemble_size = 2
    # mse_arithmetic_mean_2 = np.array(
    #     df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 2]["loss_test"])
    # assert len(mse_arithmetic_mean_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_2 has length {len(mse_arithmetic_mean_2)}"
    # axis.plot(data_x, mse_arithmetic_mean_2,
    #           marker="o", label=f"Arithmetic mean - ensemble_size: 2")

    # # Arithmetic mean - ensemble_size = 6
    # mse_arithmetic_mean_6 = np.array(
    #     df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 6]["loss_test"])
    # assert len(mse_arithmetic_mean_6) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_6 has length {len(mse_arithmetic_mean_6)}"
    # axis.plot(data_x, mse_arithmetic_mean_6,
    #           marker="o", label=f"Arithmetic mean - ensemble_size: 6")

    # Arithmetic mean - ensemble_size = 12
    mse_arithmetic_mean_12 = np.array(
        df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 12]["loss_test"])
    assert len(mse_arithmetic_mean_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_12 has length {len(mse_arithmetic_mean_12)}"
    axis.plot(data_x, mse_arithmetic_mean_12,
              marker="o", label=f"Arithmetic mean - ensemble_size: 12")

    # # Median - ensemble_size = 2
    # mse_median_2 = np.array(
    #     df_median[df_median["ensemble_size"] == 2]["loss_test"])
    # assert len(mse_median_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_2 has length {len(mse_median_2)}"
    # axis.plot(data_x, mse_median_2, marker="o",
    #           label=f"Median - ensemble_size: 2")

    # # Median - ensemble_size = 6
    # mse_median_6 = np.array(
    #     df_median[df_median["ensemble_size"] == 6]["loss_test"])
    # assert len(mse_median_6) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_6 has length {len(mse_median_6)}"
    # axis.plot(data_x, mse_median_6, marker="o",
    #           label=f"Median - ensemble_size: 6")

    # Median - ensemble_size = 12
    mse_median_12 = np.array(
        df_median[df_median["ensemble_size"] == 12]["loss_test"])
    assert len(mse_median_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_12 has length {len(mse_median_12)}"
    axis.plot(data_x, mse_median_12, marker="o",
              label=f"Median - ensemble_size: 12")

    # # NN - ensemble_size = 2
    # mse_nn_2 = np.array(df_nn[df_nn["ensemble_size"] == 2]["loss_test"])
    # assert len(mse_nn_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_2 has length {len(mse_nn_2)}"
    # axis.plot(data_x, mse_nn_2, marker="o",
    #           label=f"Neural network - ensemble_size: 2")

    # # NN - ensemble_size = 6
    # mse_nn_6 = np.array(df_nn[df_nn["ensemble_size"] == 6]["loss_test"])
    # assert len(mse_nn_6) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_6 has length {len(mse_nn_6)}"
    # axis.plot(data_x, mse_nn_6, marker="o", label=f"Neural network - 6")

    # NN - ensemble_size = 12
    mse_nn_12 = np.array(df_nn[df_nn["ensemble_size"] == 12]["loss_test"])
    assert len(mse_nn_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_12 has length {len(mse_nn_12)}"
    axis.plot(data_x, mse_nn_12, marker="o", label=f"Neural network - 12")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend(loc=f"upper center", bbox_to_anchor=(
        0.5, -0.25), ncol=1, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_12_fixed_epoch_num_400_fixed_ensemble_size_12(raw_df_arithmetic_mean: pd.DataFrame, raw_df_median: pd.DataFrame, raw_df_nn: pd.DataFrame, img_repository_path: str):
    assert raw_df_arithmetic_mean is not None and raw_df_median is not None and raw_df_nn is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_arithmetic_mean, raw_df_median, raw_df_nn, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_arithmetic_mean = {
        "hidden_size": [
            [12]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_median = {
        "hidden_size": [
            [12]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    filter_nn = {
        "hidden_size": [
            [12]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(each base learner is [12] trained with 400 epochs)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"voter_static_ncl_mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_12_fixed_epoch_num_400_fixed_ensemble_size_12.png"
    # ====================================================================

    df_arithmetic_mean = data_helper.apply_dataframe_filter(
        df=raw_df_arithmetic_mean,
        filter=filter_arithmetic_mean
    )
    df_median = data_helper.apply_dataframe_filter(
        df=raw_df_median,
        filter=filter_median
    )
    df_nn = data_helper.apply_dataframe_filter(
        df=raw_df_nn,
        filter=filter_nn
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_arithmetic_mean["correlation_penalty_coefficient"].unique())

    # Plot lines
    # # Arithmetic mean - ensemble_size = 2
    # mse_arithmetic_mean_2 = np.array(
    #     df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 2]["loss_test"])
    # assert len(mse_arithmetic_mean_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_2 has length {len(mse_arithmetic_mean_2)}"
    # axis.plot(data_x, mse_arithmetic_mean_2,
    #           marker="o", label=f"Arithmetic mean - ensemble_size: 2")

    # # Arithmetic mean - ensemble_size = 6
    # mse_arithmetic_mean_6 = np.array(
    #     df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 6]["loss_test"])
    # assert len(mse_arithmetic_mean_6) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_6 has length {len(mse_arithmetic_mean_6)}"
    # axis.plot(data_x, mse_arithmetic_mean_6,
    #           marker="o", label=f"Arithmetic mean - ensemble_size: 6")

    # Arithmetic mean - ensemble_size = 12
    mse_arithmetic_mean_12 = np.array(
        df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 12]["loss_test"])
    assert len(mse_arithmetic_mean_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean_12 has length {len(mse_arithmetic_mean_12)}"
    axis.plot(data_x, mse_arithmetic_mean_12,
              marker="o", label=f"Arithmetic mean - ensemble_size: 12")

    # # Median - ensemble_size = 2
    # mse_median_2 = np.array(
    #     df_median[df_median["ensemble_size"] == 2]["loss_test"])
    # assert len(mse_median_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_2 has length {len(mse_median_2)}"
    # axis.plot(data_x, mse_median_2, marker="o",
    #           label=f"Median - ensemble_size: 2")

    # # Median - ensemble_size = 6
    # mse_median_6 = np.array(
    #     df_median[df_median["ensemble_size"] == 6]["loss_test"])
    # assert len(mse_median_6) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_6 has length {len(mse_median_6)}"
    # axis.plot(data_x, mse_median_6, marker="o",
    #           label=f"Median - ensemble_size: 6")

    # Median - ensemble_size = 12
    mse_median_12 = np.array(
        df_median[df_median["ensemble_size"] == 12]["loss_test"])
    assert len(mse_median_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median_12 has length {len(mse_median_12)}"
    axis.plot(data_x, mse_median_12, marker="o",
              label=f"Median - ensemble_size: 12")

    # # NN - ensemble_size = 2
    # mse_nn_2 = np.array(df_nn[df_nn["ensemble_size"] == 2]["loss_test"])
    # assert len(mse_nn_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_2 has length {len(mse_nn_2)}"
    # axis.plot(data_x, mse_nn_2, marker="o",
    #           label=f"Neural network - ensemble_size: 2")

    # # NN - ensemble_size = 6
    # mse_nn_6 = np.array(df_nn[df_nn["ensemble_size"] == 6]["loss_test"])
    # assert len(mse_nn_6) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_6 has length {len(mse_nn_6)}"
    # axis.plot(data_x, mse_nn_6, marker="o", label=f"Neural network - 6")

    # NN - ensemble_size = 12
    mse_nn_12 = np.array(df_nn[df_nn["ensemble_size"] == 12]["loss_test"])
    assert len(mse_nn_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn_12 has length {len(mse_nn_12)}"
    axis.plot(data_x, mse_nn_12, marker="o", label=f"Neural network - 12")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend(loc=f"upper center", bbox_to_anchor=(
        0.5, -0.25), ncol=1, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


if __name__ == "__main__":
    df_arithmetic_mean = pd.read_csv(CSV_PATH_ARITHMETIC_MEAN)
    # NOTE: If we read from csv, df_arithmetic_mean["hidden_size"] will be of type class "str" instead of "list"
    df_arithmetic_mean["hidden_size"] = df_arithmetic_mean["hidden_size"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df_median = pd.read_csv(CSV_PATH_MEDIAN)
    # NOTE: If we read from csv, df_median["hidden_size"] will be of type class "str" instead of "list"
    df_median["hidden_size"] = df_median["hidden_size"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df_nn = pd.read_csv(CSV_PATH_NN)
    # NOTE: If we read from csv, df_nn["hidden_size"] will be of type class "str" instead of "list"
    df_nn["hidden_size"] = df_nn["hidden_size"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_fixed_epoch_num_400(
        raw_df_arithmetic_mean=df_arithmetic_mean,
        raw_df_median=df_median,
        raw_df_nn=df_nn,
        img_repository_path=IMG_REPOSITORY_PATH_VOTER_STATIC_NCL
    )

    mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_2_2_fixed_epoch_num_400(
        raw_df_arithmetic_mean=df_arithmetic_mean,
        raw_df_median=df_median,
        raw_df_nn=df_nn,
        img_repository_path=IMG_REPOSITORY_PATH_VOTER_STATIC_NCL
    )

    mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_2_2_2_2_2_fixed_epoch_num_400(
        raw_df_arithmetic_mean=df_arithmetic_mean,
        raw_df_median=df_median,
        raw_df_nn=df_nn,
        img_repository_path=IMG_REPOSITORY_PATH_VOTER_STATIC_NCL
    )

    mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_6_fixed_epoch_num_400(
        raw_df_arithmetic_mean=df_arithmetic_mean,
        raw_df_median=df_median,
        raw_df_nn=df_nn,
        img_repository_path=IMG_REPOSITORY_PATH_VOTER_STATIC_NCL
    )

    mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_12_fixed_epoch_num_400(
        raw_df_arithmetic_mean=df_arithmetic_mean,
        raw_df_median=df_median,
        raw_df_nn=df_nn,
        img_repository_path=IMG_REPOSITORY_PATH_VOTER_STATIC_NCL
    )

    mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_fixed_epoch_num_400_fixed_ensemble_size_2(
        raw_df_arithmetic_mean=df_arithmetic_mean,
        raw_df_median=df_median,
        raw_df_nn=df_nn,
        img_repository_path=IMG_REPOSITORY_PATH_VOTER_STATIC_NCL
    )

    mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_2_2_fixed_epoch_num_400_fixed_ensemble_size_2(
        raw_df_arithmetic_mean=df_arithmetic_mean,
        raw_df_median=df_median,
        raw_df_nn=df_nn,
        img_repository_path=IMG_REPOSITORY_PATH_VOTER_STATIC_NCL
    )

    mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_2_2_2_2_2_fixed_epoch_num_400_fixed_ensemble_size_2(
        raw_df_arithmetic_mean=df_arithmetic_mean,
        raw_df_median=df_median,
        raw_df_nn=df_nn,
        img_repository_path=IMG_REPOSITORY_PATH_VOTER_STATIC_NCL
    )

    mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_6_fixed_epoch_num_400_fixed_ensemble_size_2(
        raw_df_arithmetic_mean=df_arithmetic_mean,
        raw_df_median=df_median,
        raw_df_nn=df_nn,
        img_repository_path=IMG_REPOSITORY_PATH_VOTER_STATIC_NCL
    )

    mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_12_fixed_epoch_num_400_fixed_ensemble_size_2(
        raw_df_arithmetic_mean=df_arithmetic_mean,
        raw_df_median=df_median,
        raw_df_nn=df_nn,
        img_repository_path=IMG_REPOSITORY_PATH_VOTER_STATIC_NCL
    )

    mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_fixed_epoch_num_400_fixed_ensemble_size_6(
        raw_df_arithmetic_mean=df_arithmetic_mean,
        raw_df_median=df_median,
        raw_df_nn=df_nn,
        img_repository_path=IMG_REPOSITORY_PATH_VOTER_STATIC_NCL
    )

    mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_2_2_fixed_epoch_num_400_fixed_ensemble_size_6(
        raw_df_arithmetic_mean=df_arithmetic_mean,
        raw_df_median=df_median,
        raw_df_nn=df_nn,
        img_repository_path=IMG_REPOSITORY_PATH_VOTER_STATIC_NCL
    )

    mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_2_2_2_2_2_fixed_epoch_num_400_fixed_ensemble_size_6(
        raw_df_arithmetic_mean=df_arithmetic_mean,
        raw_df_median=df_median,
        raw_df_nn=df_nn,
        img_repository_path=IMG_REPOSITORY_PATH_VOTER_STATIC_NCL
    )

    mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_6_fixed_epoch_num_400_fixed_ensemble_size_6(
        raw_df_arithmetic_mean=df_arithmetic_mean,
        raw_df_median=df_median,
        raw_df_nn=df_nn,
        img_repository_path=IMG_REPOSITORY_PATH_VOTER_STATIC_NCL
    )

    mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_12_fixed_epoch_num_400_fixed_ensemble_size_6(
        raw_df_arithmetic_mean=df_arithmetic_mean,
        raw_df_median=df_median,
        raw_df_nn=df_nn,
        img_repository_path=IMG_REPOSITORY_PATH_VOTER_STATIC_NCL
    )

    mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_fixed_epoch_num_400_fixed_ensemble_size_12(
        raw_df_arithmetic_mean=df_arithmetic_mean,
        raw_df_median=df_median,
        raw_df_nn=df_nn,
        img_repository_path=IMG_REPOSITORY_PATH_VOTER_STATIC_NCL
    )

    mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_2_2_fixed_epoch_num_400_fixed_ensemble_size_12(
        raw_df_arithmetic_mean=df_arithmetic_mean,
        raw_df_median=df_median,
        raw_df_nn=df_nn,
        img_repository_path=IMG_REPOSITORY_PATH_VOTER_STATIC_NCL
    )

    mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_2_2_2_2_2_fixed_epoch_num_400_fixed_ensemble_size_12(
        raw_df_arithmetic_mean=df_arithmetic_mean,
        raw_df_median=df_median,
        raw_df_nn=df_nn,
        img_repository_path=IMG_REPOSITORY_PATH_VOTER_STATIC_NCL
    )

    mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_6_fixed_epoch_num_400_fixed_ensemble_size_12(
        raw_df_arithmetic_mean=df_arithmetic_mean,
        raw_df_median=df_median,
        raw_df_nn=df_nn,
        img_repository_path=IMG_REPOSITORY_PATH_VOTER_STATIC_NCL
    )

    mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_12_fixed_epoch_num_400_fixed_ensemble_size_12(
        raw_df_arithmetic_mean=df_arithmetic_mean,
        raw_df_median=df_median,
        raw_df_nn=df_nn,
        img_repository_path=IMG_REPOSITORY_PATH_VOTER_STATIC_NCL
    )

    logger.log(f"Graphs of voter static NCL experiment are saved ...")
