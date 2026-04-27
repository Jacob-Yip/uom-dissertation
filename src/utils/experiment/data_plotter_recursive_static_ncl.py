import ast
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from src.utils import data_helper, logger

"""
Run: python -m src.utils.experiment.data_plotter_recursive_static_ncl
"""


# Constants
# Path to project root
ROOT_PATH = os.path.join(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Path to each csv file in csv/
CSV_PATH_ORIGINAL_ARITHMETIC_MEAN = os.path.join(
    ROOT_PATH, f"data", f"experiment", f"static-ncl", f"csv", f"experiment_runner_static_ncl_arithmetic_mean.csv")
CSV_PATH_ORIGINAL_MEDIAN = os.path.join(
    ROOT_PATH, f"data", f"experiment", f"static-ncl", f"csv", f"experiment_runner_static_ncl_median.csv")
CSV_PATH_ORIGINAL_NN = os.path.join(
    ROOT_PATH, f"data", f"experiment", f"static-ncl", f"csv", f"experiment_runner_static_ncl_nn.csv")
CSV_PATH_RECURSIVE = os.path.join(
    ROOT_PATH, f"data", f"experiment", f"recursive", f"csv", f"experiment_runner_recursive_static_ncl.csv")

# Path to img/
IMG_REPOSITORY_PATH_RECURSIVE = os.path.join(
    ROOT_PATH, f"data", f"experiment", f"recursive", f"img")

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


def mse_against_correlation_penalty_coefficient_04_09_fixed_architecture_6_fixed_ensemble_size_8_fixed_aggregator_arithmetic_mean(raw_df_original_arithmetic_mean: pd.DataFrame, raw_df_original_median: pd.DataFrame, raw_df_original_nn: pd.DataFrame, raw_df_recursive: pd.DataFrame, img_repository_path: str):
    assert raw_df_original_arithmetic_mean is not None and raw_df_recursive is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_original_arithmetic_mean, raw_df_recursive, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_original_arithmetic_mean = {
        "hidden_size": [
            [6]
        ],
        "ensemble_size": [
            8
        ],
        "correlation_penalty_coefficient": [
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9
        ],
        "epoch_num": [
            400
        ]
    }
    filter_recursive = {
        "ensemble_size": [
            8
        ],
        "correlation_penalty_coefficient": [
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9
        ],
        "aggregator_type": [
            "arithmetic_mean"
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(each base learner is [6] trained with 400 epochs)\n(an arithmetic-mean-aggregator is used)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"recursive_static_ncl_mse_against_correlation_penalty_coefficient_04_09_fixed_architecture_6_fixed_ensemble_size_8_fixed_aggregator_arithmetic_mean.png"
    # ====================================================================

    df_original_arithmetic_mean = data_helper.apply_dataframe_filter(
        df=raw_df_original_arithmetic_mean,
        filter=filter_original_arithmetic_mean
    )
    df_recursive = data_helper.apply_dataframe_filter(
        df=raw_df_recursive,
        filter=filter_recursive
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_original_arithmetic_mean["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Recursive - architecture_index = 0 (perfect binary tree)
    mse_recursive_0 = np.array(
        df_recursive[df_recursive["architecture_index"] == 0]["loss_test"])
    assert len(mse_recursive_0) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_recursive_0 has length {len(mse_recursive_0)}"
    axis.plot(data_x, mse_recursive_0, marker="o",
              label=f"Recursive - perfect binary tree")

    # Recursive - architecture_index = 1 (flat tree)
    mse_recursive_1 = np.array(
        df_recursive[df_recursive["architecture_index"] == 1]["loss_test"])
    assert len(mse_recursive_1) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_recursive_1 has length {len(mse_recursive_1)}"
    axis.plot(data_x, mse_recursive_1, marker="o",
              label=f"Recursive - flat tree")

    # Original
    mse_original = np.array(df_original_arithmetic_mean["loss_test"])
    assert len(mse_original) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_original has length {len(mse_original)}"
    axis.plot(data_x, mse_original, marker="o",
              label=f"Non-recursive NCL")

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


def mse_against_correlation_penalty_coefficient_04_09_fixed_architecture_6_fixed_ensemble_size_16_fixed_aggregator_arithmetic_mean(raw_df_original_arithmetic_mean: pd.DataFrame, raw_df_original_median: pd.DataFrame, raw_df_original_nn: pd.DataFrame, raw_df_recursive: pd.DataFrame, img_repository_path: str):
    assert raw_df_original_arithmetic_mean is not None and raw_df_recursive is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_original_arithmetic_mean, raw_df_recursive, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_original_arithmetic_mean = {
        "hidden_size": [
            [6]
        ],
        "ensemble_size": [
            16
        ],
        "correlation_penalty_coefficient": [
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9
        ],
        "epoch_num": [
            400
        ]
    }
    filter_recursive = {
        "ensemble_size": [
            16
        ],
        "correlation_penalty_coefficient": [
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9
        ],
        "aggregator_type": [
            "arithmetic_mean"
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(each base learner is [6] trained with 400 epochs)\n(an arithmetic-mean-aggregator is used)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"recursive_static_ncl_mse_against_correlation_penalty_coefficient_04_09_fixed_architecture_6_fixed_ensemble_size_16_fixed_aggregator_arithmetic_mean.png"
    # ====================================================================

    df_original_arithmetic_mean = data_helper.apply_dataframe_filter(
        df=raw_df_original_arithmetic_mean,
        filter=filter_original_arithmetic_mean
    )
    df_recursive = data_helper.apply_dataframe_filter(
        df=raw_df_recursive,
        filter=filter_recursive
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_original_arithmetic_mean["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Recursive - architecture_index = 0 (perfect binary tree)
    mse_recursive_0 = np.array(
        df_recursive[df_recursive["architecture_index"] == 0]["loss_test"])
    assert len(mse_recursive_0) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_recursive_0 has length {len(mse_recursive_0)}"
    axis.plot(data_x, mse_recursive_0, marker="o",
              label=f"Recursive - perfect binary tree")

    # Recursive - architecture_index = 1 (flat tree)
    mse_recursive_1 = np.array(
        df_recursive[df_recursive["architecture_index"] == 1]["loss_test"])
    assert len(mse_recursive_1) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_recursive_1 has length {len(mse_recursive_1)}"
    axis.plot(data_x, mse_recursive_1, marker="o",
              label=f"Recursive - flat tree")

    # Original
    mse_original = np.array(df_original_arithmetic_mean["loss_test"])
    assert len(mse_original) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_original has length {len(mse_original)}"
    axis.plot(data_x, mse_original, marker="o",
              label=f"Non-recursive NCL")

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


def mse_against_correlation_penalty_coefficient_04_09_fixed_architecture_6_fixed_ensemble_size_8_fixed_aggregator_median(raw_df_original_arithmetic_mean: pd.DataFrame, raw_df_original_median: pd.DataFrame, raw_df_original_nn: pd.DataFrame, raw_df_recursive: pd.DataFrame, img_repository_path: str):
    assert raw_df_original_median is not None and raw_df_recursive is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_original_median, raw_df_recursive, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_original_median = {
        "hidden_size": [
            [6]
        ],
        "ensemble_size": [
            8
        ],
        "correlation_penalty_coefficient": [
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9
        ],
        "epoch_num": [
            400
        ]
    }
    filter_recursive = {
        "ensemble_size": [
            8
        ],
        "correlation_penalty_coefficient": [
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9
        ],
        "aggregator_type": [
            "median"
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(each base learner is [6] trained with 400 epochs)\n(a median-aggregator is used)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"recursive_static_ncl_mse_against_correlation_penalty_coefficient_04_09_fixed_architecture_6_fixed_ensemble_size_8_fixed_aggregator_median.png"
    # ====================================================================

    df_original_median = data_helper.apply_dataframe_filter(
        df=raw_df_original_median,
        filter=filter_original_median
    )
    df_recursive = data_helper.apply_dataframe_filter(
        df=raw_df_recursive,
        filter=filter_recursive
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_original_median["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Recursive - architecture_index = 0 (perfect binary tree)
    mse_recursive_0 = np.array(
        df_recursive[df_recursive["architecture_index"] == 0]["loss_test"])
    assert len(mse_recursive_0) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_recursive_0 has length {len(mse_recursive_0)}"
    axis.plot(data_x, mse_recursive_0, marker="o",
              label=f"Recursive - perfect binary tree")

    # Recursive - architecture_index = 1 (flat tree)
    mse_recursive_1 = np.array(
        df_recursive[df_recursive["architecture_index"] == 1]["loss_test"])
    assert len(mse_recursive_1) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_recursive_1 has length {len(mse_recursive_1)}"
    axis.plot(data_x, mse_recursive_1, marker="o",
              label=f"Recursive - flat tree")

    # Original
    mse_original = np.array(df_original_median["loss_test"])
    assert len(mse_original) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_original has length {len(mse_original)}"
    axis.plot(data_x, mse_original, marker="o",
              label=f"Non-recursive NCL")

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


def mse_against_correlation_penalty_coefficient_04_09_fixed_architecture_6_fixed_ensemble_size_16_fixed_aggregator_median(raw_df_original_arithmetic_mean: pd.DataFrame, raw_df_original_median: pd.DataFrame, raw_df_original_nn: pd.DataFrame, raw_df_recursive: pd.DataFrame, img_repository_path: str):
    assert raw_df_original_median is not None and raw_df_recursive is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_original_median, raw_df_recursive, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_original_median = {
        "hidden_size": [
            [6]
        ],
        "ensemble_size": [
            16
        ],
        "correlation_penalty_coefficient": [
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9
        ],
        "epoch_num": [
            400
        ]
    }
    filter_recursive = {
        "ensemble_size": [
            16
        ],
        "correlation_penalty_coefficient": [
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9
        ],
        "aggregator_type": [
            "median"
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(each base learner is [6] trained with 400 epochs)\n(a median-aggregator is used)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"recursive_static_ncl_mse_against_correlation_penalty_coefficient_04_09_fixed_architecture_6_fixed_ensemble_size_16_fixed_aggregator_median.png"
    # ====================================================================

    df_original_median = data_helper.apply_dataframe_filter(
        df=raw_df_original_median,
        filter=filter_original_median
    )
    df_recursive = data_helper.apply_dataframe_filter(
        df=raw_df_recursive,
        filter=filter_recursive
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_original_median["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Recursive - architecture_index = 0 (perfect binary tree)
    mse_recursive_0 = np.array(
        df_recursive[df_recursive["architecture_index"] == 0]["loss_test"])
    assert len(mse_recursive_0) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_recursive_0 has length {len(mse_recursive_0)}"
    axis.plot(data_x, mse_recursive_0, marker="o",
              label=f"Recursive - perfect binary tree")

    # Recursive - architecture_index = 1 (flat tree)
    mse_recursive_1 = np.array(
        df_recursive[df_recursive["architecture_index"] == 1]["loss_test"])
    assert len(mse_recursive_1) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_recursive_1 has length {len(mse_recursive_1)}"
    axis.plot(data_x, mse_recursive_1, marker="o",
              label=f"Recursive - flat tree")

    # Original
    mse_original = np.array(df_original_median["loss_test"])
    assert len(mse_original) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_original has length {len(mse_original)}"
    axis.plot(data_x, mse_original, marker="o",
              label=f"Non-recursive NCL")

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


def mse_against_correlation_penalty_coefficient_04_09_fixed_architecture_6_fixed_ensemble_size_8_fixed_aggregator_nn(raw_df_original_arithmetic_mean: pd.DataFrame, raw_df_original_median: pd.DataFrame, raw_df_original_nn: pd.DataFrame, raw_df_recursive: pd.DataFrame, img_repository_path: str):
    assert raw_df_original_nn is not None and raw_df_recursive is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_original_nn, raw_df_recursive, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_original_nn = {
        "hidden_size": [
            [6]
        ],
        "ensemble_size": [
            8
        ],
        "correlation_penalty_coefficient": [
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9
        ],
        "epoch_num": [
            400
        ]
    }
    filter_recursive = {
        "ensemble_size": [
            8
        ],
        "correlation_penalty_coefficient": [
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9
        ],
        "aggregator_type": [
            "nn"
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(each base learner is [6] trained with 400 epochs)\n(a neural-network-aggregator is used)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"recursive_static_ncl_mse_against_correlation_penalty_coefficient_04_09_fixed_architecture_6_fixed_ensemble_size_8_fixed_aggregator_nn.png"
    # ====================================================================

    df_original_nn = data_helper.apply_dataframe_filter(
        df=raw_df_original_nn,
        filter=filter_original_nn
    )
    df_recursive = data_helper.apply_dataframe_filter(
        df=raw_df_recursive,
        filter=filter_recursive
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_original_nn["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Recursive - architecture_index = 0 (perfect binary tree)
    mse_recursive_0 = np.array(
        df_recursive[df_recursive["architecture_index"] == 0]["loss_test"])
    assert len(mse_recursive_0) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_recursive_0 has length {len(mse_recursive_0)}"
    axis.plot(data_x, mse_recursive_0, marker="o",
              label=f"Recursive - perfect binary tree")

    # Recursive - architecture_index = 1 (flat tree)
    mse_recursive_1 = np.array(
        df_recursive[df_recursive["architecture_index"] == 1]["loss_test"])
    assert len(mse_recursive_1) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_recursive_1 has length {len(mse_recursive_1)}"
    axis.plot(data_x, mse_recursive_1, marker="o",
              label=f"Recursive - flat tree")

    # Original
    mse_original = np.array(df_original_nn["loss_test"])
    assert len(mse_original) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_original has length {len(mse_original)}"
    axis.plot(data_x, mse_original, marker="o",
              label=f"Non-recursive NCL")

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


def mse_against_correlation_penalty_coefficient_04_09_fixed_architecture_6_fixed_ensemble_size_16_fixed_aggregator_nn(raw_df_original_arithmetic_mean: pd.DataFrame, raw_df_original_median: pd.DataFrame, raw_df_original_nn: pd.DataFrame, raw_df_recursive: pd.DataFrame, img_repository_path: str):
    assert raw_df_original_nn is not None and raw_df_recursive is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_original_nn, raw_df_recursive, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_original_nn = {
        "hidden_size": [
            [6]
        ],
        "ensemble_size": [
            16
        ],
        "correlation_penalty_coefficient": [
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9
        ],
        "epoch_num": [
            400
        ]
    }
    filter_recursive = {
        "ensemble_size": [
            16
        ],
        "correlation_penalty_coefficient": [
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9
        ],
        "aggregator_type": [
            "nn"
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(each base learner is [6] trained with 400 epochs)\n(a neural-network-aggregator is used)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"recursive_static_ncl_mse_against_correlation_penalty_coefficient_04_09_fixed_architecture_6_fixed_ensemble_size_16_fixed_aggregator_nn.png"
    # ====================================================================

    df_original_nn = data_helper.apply_dataframe_filter(
        df=raw_df_original_nn,
        filter=filter_original_nn
    )
    df_recursive = data_helper.apply_dataframe_filter(
        df=raw_df_recursive,
        filter=filter_recursive
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_original_nn["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Recursive - architecture_index = 0 (perfect binary tree)
    mse_recursive_0 = np.array(
        df_recursive[df_recursive["architecture_index"] == 0]["loss_test"])
    assert len(mse_recursive_0) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_recursive_0 has length {len(mse_recursive_0)}"
    axis.plot(data_x, mse_recursive_0, marker="o",
              label=f"Recursive - perfect binary tree")

    # Recursive - architecture_index = 1 (flat tree)
    mse_recursive_1 = np.array(
        df_recursive[df_recursive["architecture_index"] == 1]["loss_test"])
    assert len(mse_recursive_1) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_recursive_1 has length {len(mse_recursive_1)}"
    axis.plot(data_x, mse_recursive_1, marker="o",
              label=f"Recursive - flat tree")

    # Original
    mse_original = np.array(df_original_nn["loss_test"])
    assert len(mse_original) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_original has length {len(mse_original)}"
    axis.plot(data_x, mse_original, marker="o",
              label=f"Non-recursive NCL")

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
    df_original_arithmetic_mean = pd.read_csv(
        CSV_PATH_ORIGINAL_ARITHMETIC_MEAN)
    # NOTE: If we read from csv, df_original_arithmetic_mean["hidden_size"] will be of type class "str" instead of "list"
    df_original_arithmetic_mean["hidden_size"] = df_original_arithmetic_mean["hidden_size"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df_original_median = pd.read_csv(CSV_PATH_ORIGINAL_MEDIAN)
    # NOTE: If we read from csv, df_original_median["hidden_size"] will be of type class "str" instead of "list"
    df_original_median["hidden_size"] = df_original_median["hidden_size"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df_original_nn = pd.read_csv(CSV_PATH_ORIGINAL_NN)
    # NOTE: If we read from csv, df_original_nn["hidden_size"] will be of type class "str" instead of "list"
    df_original_nn["hidden_size"] = df_original_nn["hidden_size"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df_recursive = pd.read_csv(CSV_PATH_RECURSIVE)

    mse_against_correlation_penalty_coefficient_04_09_fixed_architecture_6_fixed_ensemble_size_8_fixed_aggregator_arithmetic_mean(
        raw_df_original_arithmetic_mean=df_original_arithmetic_mean,
        raw_df_original_median=df_original_median,
        raw_df_original_nn=df_original_nn,
        raw_df_recursive=df_recursive,
        img_repository_path=IMG_REPOSITORY_PATH_RECURSIVE
    )

    mse_against_correlation_penalty_coefficient_04_09_fixed_architecture_6_fixed_ensemble_size_16_fixed_aggregator_arithmetic_mean(
        raw_df_original_arithmetic_mean=df_original_arithmetic_mean,
        raw_df_original_median=df_original_median,
        raw_df_original_nn=df_original_nn,
        raw_df_recursive=df_recursive,
        img_repository_path=IMG_REPOSITORY_PATH_RECURSIVE
    )

    mse_against_correlation_penalty_coefficient_04_09_fixed_architecture_6_fixed_ensemble_size_8_fixed_aggregator_median(
        raw_df_original_arithmetic_mean=df_original_arithmetic_mean,
        raw_df_original_median=df_original_median,
        raw_df_original_nn=df_original_nn,
        raw_df_recursive=df_recursive,
        img_repository_path=IMG_REPOSITORY_PATH_RECURSIVE
    )

    mse_against_correlation_penalty_coefficient_04_09_fixed_architecture_6_fixed_ensemble_size_16_fixed_aggregator_median(
        raw_df_original_arithmetic_mean=df_original_arithmetic_mean,
        raw_df_original_median=df_original_median,
        raw_df_original_nn=df_original_nn,
        raw_df_recursive=df_recursive,
        img_repository_path=IMG_REPOSITORY_PATH_RECURSIVE
    )

    mse_against_correlation_penalty_coefficient_04_09_fixed_architecture_6_fixed_ensemble_size_8_fixed_aggregator_nn(
        raw_df_original_arithmetic_mean=df_original_arithmetic_mean,
        raw_df_original_median=df_original_median,
        raw_df_original_nn=df_original_nn,
        raw_df_recursive=df_recursive,
        img_repository_path=IMG_REPOSITORY_PATH_RECURSIVE
    )

    mse_against_correlation_penalty_coefficient_04_09_fixed_architecture_6_fixed_ensemble_size_16_fixed_aggregator_nn(
        raw_df_original_arithmetic_mean=df_original_arithmetic_mean,
        raw_df_original_median=df_original_median,
        raw_df_original_nn=df_original_nn,
        raw_df_recursive=df_recursive,
        img_repository_path=IMG_REPOSITORY_PATH_RECURSIVE
    )

    logger.log(f"Graphs of recursive static NCL experiment are saved ...")
