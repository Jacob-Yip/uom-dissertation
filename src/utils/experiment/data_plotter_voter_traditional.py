import ast
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from src.utils import data_helper, logger

"""
Run: python -m src.utils.experiment.data_plotter_voter_traditional
"""


# Constants
# Path to project root
ROOT_PATH = os.path.join(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Path to each csv file in csv/
CSV_PATH_ARITHMETIC_MEAN = os.path.join(
    ROOT_PATH, f"data", f"experiment", f"traditional", f"csv", f"experiment_runner_traditional_arithmetic_mean.csv")
CSV_PATH_MEDIAN = os.path.join(
    ROOT_PATH, f"data", f"experiment", f"traditional", f"csv", f"experiment_runner_traditional_median.csv")
CSV_PATH_NN = os.path.join(
    ROOT_PATH, f"data", f"experiment", f"traditional", f"csv", f"experiment_runner_traditional_nn.csv")

# Path to img/
IMG_REPOSITORY_PATH_VOTER_TRADITIONAL = os.path.join(
    ROOT_PATH, f"data", f"experiment", f"voter", f"traditional", f"img")

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


def mse_against_ensemble_size_2_16_fixed_architecture_2_fixed_epoch_num_400(raw_df_arithmetic_mean: pd.DataFrame, raw_df_median: pd.DataFrame, raw_df_nn: pd.DataFrame, img_repository_path: str):
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
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16
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
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16
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
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"ensemble_size"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs ensemble_size\n(each base learner is [2] trained with 400 epochs)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"voter_traditional_mse_against_ensemble_size_2_16_fixed_architecture_2_fixed_epoch_num_400.png"
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
    data_x = np.array(df_arithmetic_mean["ensemble_size"].unique())

    # Plot lines
    # Arithmetic mean
    mse_arithmetic_mean = np.array(df_arithmetic_mean["loss_test"])
    assert len(mse_arithmetic_mean) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean has length {len(mse_arithmetic_mean)}"
    axis.plot(data_x, mse_arithmetic_mean, marker="o",
              label=f"Test - arithmetic mean")

    # Median
    mse_median = np.array(df_median["loss_test"])
    assert len(mse_median) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median has length {len(mse_median)}"
    axis.plot(data_x, mse_median, marker="o",
              label=f"Test - median")

    # NN
    mse_nn = np.array(df_nn["loss_test"])
    assert len(mse_nn) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn has length {len(mse_nn)}"
    axis.plot(data_x, mse_nn, marker="o",
              label=f"Test - neural network")

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


def mse_against_ensemble_size_2_16_fixed_architecture_2_2_2_fixed_epoch_num_400(raw_df_arithmetic_mean: pd.DataFrame, raw_df_median: pd.DataFrame, raw_df_nn: pd.DataFrame, img_repository_path: str):
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
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16
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
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16
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
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"ensemble_size"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs ensemble_size\n(each base learner is [2, 2, 2] trained with 400 epochs)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"voter_traditional_mse_against_ensemble_size_2_16_fixed_architecture_2_2_2_fixed_epoch_num_400.png"
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
    data_x = np.array(df_arithmetic_mean["ensemble_size"].unique())

    # Plot lines
    # Arithmetic mean
    mse_arithmetic_mean = np.array(df_arithmetic_mean["loss_test"])
    assert len(mse_arithmetic_mean) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean has length {len(mse_arithmetic_mean)}"
    axis.plot(data_x, mse_arithmetic_mean, marker="o",
              label=f"Test - arithmetic mean")

    # Median
    mse_median = np.array(df_median["loss_test"])
    assert len(mse_median) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median has length {len(mse_median)}"
    axis.plot(data_x, mse_median, marker="o",
              label=f"Test - median")

    # NN
    mse_nn = np.array(df_nn["loss_test"])
    assert len(mse_nn) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn has length {len(mse_nn)}"
    axis.plot(data_x, mse_nn, marker="o",
              label=f"Test - neural network")

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


def mse_against_ensemble_size_2_16_fixed_architecture_2_2_2_2_2_2_fixed_epoch_num_400(raw_df_arithmetic_mean: pd.DataFrame, raw_df_median: pd.DataFrame, raw_df_nn: pd.DataFrame, img_repository_path: str):
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
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16
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
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16
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
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"ensemble_size"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs ensemble_size\n(each base learner is [2, 2, 2, 2, 2, 2] trained with 400 epochs)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"voter_traditional_mse_against_ensemble_size_2_16_fixed_architecture_2_2_2_2_2_2_fixed_epoch_num_400.png"
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
    data_x = np.array(df_arithmetic_mean["ensemble_size"].unique())

    # Plot lines
    # Arithmetic mean
    mse_arithmetic_mean = np.array(df_arithmetic_mean["loss_test"])
    assert len(mse_arithmetic_mean) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean has length {len(mse_arithmetic_mean)}"
    axis.plot(data_x, mse_arithmetic_mean, marker="o",
              label=f"Test - arithmetic mean")

    # Median
    mse_median = np.array(df_median["loss_test"])
    assert len(mse_median) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median has length {len(mse_median)}"
    axis.plot(data_x, mse_median, marker="o",
              label=f"Test - median")

    # NN
    mse_nn = np.array(df_nn["loss_test"])
    assert len(mse_nn) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn has length {len(mse_nn)}"
    axis.plot(data_x, mse_nn, marker="o",
              label=f"Test - neural network")

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


def mse_against_ensemble_size_2_16_fixed_architecture_6_fixed_epoch_num_400(raw_df_arithmetic_mean: pd.DataFrame, raw_df_median: pd.DataFrame, raw_df_nn: pd.DataFrame, img_repository_path: str):
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
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16
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
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16
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
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"ensemble_size"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs ensemble_size\n(each base learner is [6] trained with 400 epochs)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"voter_traditional_mse_against_ensemble_size_2_16_fixed_architecture_6_fixed_epoch_num_400.png"
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
    data_x = np.array(df_arithmetic_mean["ensemble_size"].unique())

    # Plot lines
    # Arithmetic mean
    mse_arithmetic_mean = np.array(df_arithmetic_mean["loss_test"])
    assert len(mse_arithmetic_mean) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean has length {len(mse_arithmetic_mean)}"
    axis.plot(data_x, mse_arithmetic_mean, marker="o",
              label=f"Test - arithmetic mean")

    # Median
    mse_median = np.array(df_median["loss_test"])
    assert len(mse_median) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median has length {len(mse_median)}"
    axis.plot(data_x, mse_median, marker="o",
              label=f"Test - median")

    # NN
    mse_nn = np.array(df_nn["loss_test"])
    assert len(mse_nn) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn has length {len(mse_nn)}"
    axis.plot(data_x, mse_nn, marker="o",
              label=f"Test - neural network")

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


def mse_against_ensemble_size_2_16_fixed_architecture_12_fixed_epoch_num_400(raw_df_arithmetic_mean: pd.DataFrame, raw_df_median: pd.DataFrame, raw_df_nn: pd.DataFrame, img_repository_path: str):
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
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16
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
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16
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
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"ensemble_size"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs ensemble_size\n(each base learner is [12] trained with 400 epochs)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"voter_traditional_mse_against_ensemble_size_2_16_fixed_architecture_12_fixed_epoch_num_400.png"
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
    data_x = np.array(df_arithmetic_mean["ensemble_size"].unique())

    # Plot lines
    # Arithmetic mean
    mse_arithmetic_mean = np.array(df_arithmetic_mean["loss_test"])
    assert len(mse_arithmetic_mean) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean has length {len(mse_arithmetic_mean)}"
    axis.plot(data_x, mse_arithmetic_mean, marker="o",
              label=f"Test - arithmetic mean")

    # Median
    mse_median = np.array(df_median["loss_test"])
    assert len(mse_median) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median has length {len(mse_median)}"
    axis.plot(data_x, mse_median, marker="o",
              label=f"Test - median")

    # NN
    mse_nn = np.array(df_nn["loss_test"])
    assert len(mse_nn) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn has length {len(mse_nn)}"
    axis.plot(data_x, mse_nn, marker="o",
              label=f"Test - neural network")

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

    mse_against_ensemble_size_2_16_fixed_architecture_2_fixed_epoch_num_400(
        raw_df_arithmetic_mean=df_arithmetic_mean,
        raw_df_median=df_median,
        raw_df_nn=df_nn,
        img_repository_path=IMG_REPOSITORY_PATH_VOTER_TRADITIONAL
    )

    mse_against_ensemble_size_2_16_fixed_architecture_2_2_2_fixed_epoch_num_400(
        raw_df_arithmetic_mean=df_arithmetic_mean,
        raw_df_median=df_median,
        raw_df_nn=df_nn,
        img_repository_path=IMG_REPOSITORY_PATH_VOTER_TRADITIONAL
    )

    mse_against_ensemble_size_2_16_fixed_architecture_2_2_2_2_2_2_fixed_epoch_num_400(
        raw_df_arithmetic_mean=df_arithmetic_mean,
        raw_df_median=df_median,
        raw_df_nn=df_nn,
        img_repository_path=IMG_REPOSITORY_PATH_VOTER_TRADITIONAL
    )

    mse_against_ensemble_size_2_16_fixed_architecture_6_fixed_epoch_num_400(
        raw_df_arithmetic_mean=df_arithmetic_mean,
        raw_df_median=df_median,
        raw_df_nn=df_nn,
        img_repository_path=IMG_REPOSITORY_PATH_VOTER_TRADITIONAL
    )

    mse_against_ensemble_size_2_16_fixed_architecture_12_fixed_epoch_num_400(
        raw_df_arithmetic_mean=df_arithmetic_mean,
        raw_df_median=df_median,
        raw_df_nn=df_nn,
        img_repository_path=IMG_REPOSITORY_PATH_VOTER_TRADITIONAL
    )

    logger.log(f"Graphs of voter traditional experiment are saved ...")
