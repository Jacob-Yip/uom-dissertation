import ast
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from src.utils import data_helper, logger

"""
Run: python -m src.utils.experiment.data_plotter_static_ncl
"""


# Constants
# Path to project root
ROOT_PATH = os.path.join(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Path to each csv file in csv/
CSV_PATH_STATIC_NCL = os.path.join(
    ROOT_PATH, f"data", f"experiment", f"static-ncl", f"csv", f"experiment_runner_static_ncl_arithmetic_mean.csv")
CSV_PATH_TRADITIONAL = os.path.join(
    ROOT_PATH, f"data", f"experiment", f"traditional", f"csv", f"experiment_runner_traditional_arithmetic_mean.csv")
CSV_PATH_MLP = os.path.join(
    ROOT_PATH, f"data", f"experiment", f"mlp", f"csv", f"experiment_runner_mlp.csv")

# Path to img/
IMG_REPOSITORY_PATH_STATIC_NCL = os.path.join(
    ROOT_PATH, f"data", f"experiment", f"static-ncl", f"img")

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


def mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2(raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_static_ncl is not None and raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_static_ncl, raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_static_ncl = {
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
        ]
    }
    filter_traditional = {
        "hidden_size": [
            [2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "epoch_num": [
            400
        ]
    }
    filter_mlp = {
        "hidden_size": [
            [2]
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(each base learner is [2])"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"static_ncl_mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2.png"
    # ====================================================================

    df_static_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_static_ncl,
        filter=filter_static_ncl
    )
    df_traditional = data_helper.apply_dataframe_filter(
        df=raw_df_traditional,
        filter=filter_traditional
    )
    df_mlp = data_helper.apply_dataframe_filter(
        df=raw_df_mlp,
        filter=filter_mlp
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_static_ncl["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Static NCL - ensemble_size = 2
    mse_static_ncl_2 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 2]["loss_test"])
    assert len(mse_static_ncl_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_2 has length {len(mse_static_ncl_2)}"
    axis.plot(data_x, mse_static_ncl_2, marker="o", label=f"NCL - 2")

    # Static NCL - ensemble_size = 6
    mse_static_ncl_6 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 6]["loss_test"])
    assert len(mse_static_ncl_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_6 has length {len(mse_static_ncl_6)}"
    axis.plot(data_x, mse_static_ncl_6, marker="o", label=f"NCL - 6")

    # Static NCL - ensemble_size = 12
    mse_static_ncl_12 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 12]["loss_test"])
    assert len(mse_static_ncl_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_12 has length {len(mse_static_ncl_12)}"
    axis.plot(data_x, mse_static_ncl_12, marker="o", label=f"NCL - 12")

    # Arithmetic-mean-ensemble - ensemble_size = 2
    # Expect only 1 data point for traditional originally
    mse_traditional_2 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 2]["loss_test"]), len(data_x))
    assert len(mse_traditional_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_2 has length {len(mse_traditional_2)}"
    axis.plot(data_x, mse_traditional_2, marker="o",
              label=f"Arithmetic-mean-ensemble - 2")

    # Arithmetic-mean-ensemble - ensemble_size = 6
    # Expect only 1 data point for traditional originally
    mse_traditional_6 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 6]["loss_test"]), len(data_x))
    assert len(mse_traditional_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_6 has length {len(mse_traditional_6)}"
    axis.plot(data_x, mse_traditional_6, marker="o",
              label=f"Arithmetic-mean-ensemble - 6")

    # Arithmetic-mean-ensemble - ensemble_size = 12
    # Expect only 1 data point for traditional originally
    mse_traditional_12 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 12]["loss_test"]), len(data_x))
    assert len(mse_traditional_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_12 has length {len(mse_traditional_12)}"
    axis.plot(data_x, mse_traditional_12,
              marker="o", label=f"Arithmetic-mean-ensemble - 12")

    # MLP
    # Expect only 1 data point for MLP originally
    mse_mlp = np.repeat(np.array(df_mlp["loss_test"]), len(data_x))
    assert len(mse_mlp) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp has length {len(mse_mlp)}"
    axis.plot(data_x, mse_mlp, marker="o", label=f"MLP")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend(loc=f"upper center", bbox_to_anchor=(
        0.5, -0.25), ncol=2, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_correlation_penalty_coefficient_08_15_fixed_architecture_2(raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_static_ncl is not None and raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_static_ncl, raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_static_ncl = {
        "hidden_size": [
            [2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5
        ]
    }
    filter_traditional = {
        "hidden_size": [
            [2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "epoch_num": [
            400
        ]
    }
    filter_mlp = {
        "hidden_size": [
            [2]
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(each base learner is [2])"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"static_ncl_mse_against_correlation_penalty_coefficient_08_15_fixed_architecture_2.png"
    # ====================================================================

    df_static_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_static_ncl,
        filter=filter_static_ncl
    )
    df_traditional = data_helper.apply_dataframe_filter(
        df=raw_df_traditional,
        filter=filter_traditional
    )
    df_mlp = data_helper.apply_dataframe_filter(
        df=raw_df_mlp,
        filter=filter_mlp
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_static_ncl["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Static NCL - ensemble_size = 2
    mse_static_ncl_2 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 2]["loss_test"])
    assert len(mse_static_ncl_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_2 has length {len(mse_static_ncl_2)}"
    axis.plot(data_x, mse_static_ncl_2, marker="o", label=f"NCL - 2")

    # Static NCL - ensemble_size = 6
    mse_static_ncl_6 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 6]["loss_test"])
    assert len(mse_static_ncl_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_6 has length {len(mse_static_ncl_6)}"
    axis.plot(data_x, mse_static_ncl_6, marker="o", label=f"NCL - 6")

    # Static NCL - ensemble_size = 12
    mse_static_ncl_12 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 12]["loss_test"])
    assert len(mse_static_ncl_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_12 has length {len(mse_static_ncl_12)}"
    axis.plot(data_x, mse_static_ncl_12, marker="o", label=f"NCL - 12")

    # Arithmetic-mean-ensemble - ensemble_size = 2
    # Expect only 1 data point for traditional originally
    mse_traditional_2 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 2]["loss_test"]), len(data_x))
    assert len(mse_traditional_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_2 has length {len(mse_traditional_2)}"
    axis.plot(data_x, mse_traditional_2, marker="o",
              label=f"Arithmetic-mean-ensemble - 2")

    # Arithmetic-mean-ensemble - ensemble_size = 6
    # Expect only 1 data point for traditional originally
    mse_traditional_6 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 6]["loss_test"]), len(data_x))
    assert len(mse_traditional_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_6 has length {len(mse_traditional_6)}"
    axis.plot(data_x, mse_traditional_6, marker="o",
              label=f"Arithmetic-mean-ensemble - 6")

    # Arithmetic-mean-ensemble - ensemble_size = 12
    # Expect only 1 data point for traditional originally
    mse_traditional_12 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 12]["loss_test"]), len(data_x))
    assert len(mse_traditional_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_12 has length {len(mse_traditional_12)}"
    axis.plot(data_x, mse_traditional_12,
              marker="o", label=f"Arithmetic-mean-ensemble - 12")

    # MLP
    # Expect only 1 data point for MLP originally
    mse_mlp = np.repeat(np.array(df_mlp["loss_test"]), len(data_x))
    assert len(mse_mlp) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp has length {len(mse_mlp)}"
    axis.plot(data_x, mse_mlp, marker="o", label=f"MLP")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend(loc=f"upper center", bbox_to_anchor=(
        0.5, -0.25), ncol=2, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_2_2(raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_static_ncl is not None and raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_static_ncl, raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_static_ncl = {
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
        ]
    }
    filter_traditional = {
        "hidden_size": [
            [2, 2, 2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "epoch_num": [
            400
        ]
    }
    filter_mlp = {
        "hidden_size": [
            [2, 2, 2]
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(each base learner is [2, 2, 2])"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"static_ncl_mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_2_2.png"
    # ====================================================================

    df_static_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_static_ncl,
        filter=filter_static_ncl
    )
    df_traditional = data_helper.apply_dataframe_filter(
        df=raw_df_traditional,
        filter=filter_traditional
    )
    df_mlp = data_helper.apply_dataframe_filter(
        df=raw_df_mlp,
        filter=filter_mlp
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_static_ncl["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Static NCL - ensemble_size = 2
    mse_static_ncl_2 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 2]["loss_test"])
    assert len(mse_static_ncl_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_2 has length {len(mse_static_ncl_2)}"
    axis.plot(data_x, mse_static_ncl_2, marker="o", label=f"NCL - 2")

    # Static NCL - ensemble_size = 6
    mse_static_ncl_6 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 6]["loss_test"])
    assert len(mse_static_ncl_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_6 has length {len(mse_static_ncl_6)}"
    axis.plot(data_x, mse_static_ncl_6, marker="o", label=f"NCL - 6")

    # Static NCL - ensemble_size = 12
    mse_static_ncl_12 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 12]["loss_test"])
    assert len(mse_static_ncl_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_12 has length {len(mse_static_ncl_12)}"
    axis.plot(data_x, mse_static_ncl_12, marker="o", label=f"NCL - 12")

    # Arithmetic-mean-ensemble - ensemble_size = 2
    # Expect only 1 data point for traditional originally
    mse_traditional_2 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 2]["loss_test"]), len(data_x))
    assert len(mse_traditional_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_2 has length {len(mse_traditional_2)}"
    axis.plot(data_x, mse_traditional_2, marker="o",
              label=f"Arithmetic-mean-ensemble - 2")

    # Arithmetic-mean-ensemble - ensemble_size = 6
    # Expect only 1 data point for traditional originally
    mse_traditional_6 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 6]["loss_test"]), len(data_x))
    assert len(mse_traditional_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_6 has length {len(mse_traditional_6)}"
    axis.plot(data_x, mse_traditional_6, marker="o",
              label=f"Arithmetic-mean-ensemble - 6")

    # Arithmetic-mean-ensemble - ensemble_size = 12
    # Expect only 1 data point for traditional originally
    mse_traditional_12 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 12]["loss_test"]), len(data_x))
    assert len(mse_traditional_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_12 has length {len(mse_traditional_12)}"
    axis.plot(data_x, mse_traditional_12,
              marker="o", label=f"Arithmetic-mean-ensemble - 12")

    # MLP
    # Expect only 1 data point for MLP originally
    mse_mlp = np.repeat(np.array(df_mlp["loss_test"]), len(data_x))
    assert len(mse_mlp) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp has length {len(mse_mlp)}"
    axis.plot(data_x, mse_mlp, marker="o", label=f"MLP")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend(loc=f"upper center", bbox_to_anchor=(
        0.5, -0.25), ncol=2, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_correlation_penalty_coefficient_08_15_fixed_architecture_2_2_2(raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_static_ncl is not None and raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_static_ncl, raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_static_ncl = {
        "hidden_size": [
            [2, 2, 2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5
        ]
    }
    filter_traditional = {
        "hidden_size": [
            [2, 2, 2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "epoch_num": [
            400
        ]
    }
    filter_mlp = {
        "hidden_size": [
            [2, 2, 2]
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(each base learner is [2, 2, 2])"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"static_ncl_mse_against_correlation_penalty_coefficient_08_15_fixed_architecture_2_2_2.png"
    # ====================================================================

    df_static_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_static_ncl,
        filter=filter_static_ncl
    )
    df_traditional = data_helper.apply_dataframe_filter(
        df=raw_df_traditional,
        filter=filter_traditional
    )
    df_mlp = data_helper.apply_dataframe_filter(
        df=raw_df_mlp,
        filter=filter_mlp
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_static_ncl["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Static NCL - ensemble_size = 2
    mse_static_ncl_2 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 2]["loss_test"])
    assert len(mse_static_ncl_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_2 has length {len(mse_static_ncl_2)}"
    axis.plot(data_x, mse_static_ncl_2, marker="o", label=f"NCL - 2")

    # Static NCL - ensemble_size = 6
    mse_static_ncl_6 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 6]["loss_test"])
    assert len(mse_static_ncl_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_6 has length {len(mse_static_ncl_6)}"
    axis.plot(data_x, mse_static_ncl_6, marker="o", label=f"NCL - 6")

    # Static NCL - ensemble_size = 12
    mse_static_ncl_12 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 12]["loss_test"])
    assert len(mse_static_ncl_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_12 has length {len(mse_static_ncl_12)}"
    axis.plot(data_x, mse_static_ncl_12, marker="o", label=f"NCL - 12")

    # Arithmetic-mean-ensemble - ensemble_size = 2
    # Expect only 1 data point for traditional originally
    mse_traditional_2 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 2]["loss_test"]), len(data_x))
    assert len(mse_traditional_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_2 has length {len(mse_traditional_2)}"
    axis.plot(data_x, mse_traditional_2, marker="o",
              label=f"Arithmetic-mean-ensemble - 2")

    # Arithmetic-mean-ensemble - ensemble_size = 6
    # Expect only 1 data point for traditional originally
    mse_traditional_6 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 6]["loss_test"]), len(data_x))
    assert len(mse_traditional_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_6 has length {len(mse_traditional_6)}"
    axis.plot(data_x, mse_traditional_6, marker="o",
              label=f"Arithmetic-mean-ensemble - 6")

    # Arithmetic-mean-ensemble - ensemble_size = 12
    # Expect only 1 data point for traditional originally
    mse_traditional_12 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 12]["loss_test"]), len(data_x))
    assert len(mse_traditional_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_12 has length {len(mse_traditional_12)}"
    axis.plot(data_x, mse_traditional_12,
              marker="o", label=f"Arithmetic-mean-ensemble - 12")

    # MLP
    # Expect only 1 data point for MLP originally
    mse_mlp = np.repeat(np.array(df_mlp["loss_test"]), len(data_x))
    assert len(mse_mlp) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp has length {len(mse_mlp)}"
    axis.plot(data_x, mse_mlp, marker="o", label=f"MLP")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend(loc=f"upper center", bbox_to_anchor=(
        0.5, -0.25), ncol=2, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_2_2_2_2_2(raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_static_ncl is not None and raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_static_ncl, raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_static_ncl = {
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
        ]
    }
    filter_traditional = {
        "hidden_size": [
            [2, 2, 2, 2, 2, 2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "epoch_num": [
            400
        ]
    }
    filter_mlp = {
        "hidden_size": [
            [2, 2, 2, 2, 2, 2]
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(each base learner is [2, 2, 2, 2, 2, 2])"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"static_ncl_mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_2_2_2_2_2.png"
    # ====================================================================

    df_static_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_static_ncl,
        filter=filter_static_ncl
    )
    df_traditional = data_helper.apply_dataframe_filter(
        df=raw_df_traditional,
        filter=filter_traditional
    )
    df_mlp = data_helper.apply_dataframe_filter(
        df=raw_df_mlp,
        filter=filter_mlp
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_static_ncl["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Static NCL - ensemble_size = 2
    mse_static_ncl_2 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 2]["loss_test"])
    assert len(mse_static_ncl_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_2 has length {len(mse_static_ncl_2)}"
    axis.plot(data_x, mse_static_ncl_2, marker="o", label=f"NCL - 2")

    # Static NCL - ensemble_size = 6
    mse_static_ncl_6 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 6]["loss_test"])
    assert len(mse_static_ncl_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_6 has length {len(mse_static_ncl_6)}"
    axis.plot(data_x, mse_static_ncl_6, marker="o", label=f"NCL - 6")

    # Static NCL - ensemble_size = 12
    mse_static_ncl_12 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 12]["loss_test"])
    assert len(mse_static_ncl_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_12 has length {len(mse_static_ncl_12)}"
    axis.plot(data_x, mse_static_ncl_12, marker="o", label=f"NCL - 12")

    # Arithmetic-mean-ensemble - ensemble_size = 2
    # Expect only 1 data point for traditional originally
    mse_traditional_2 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 2]["loss_test"]), len(data_x))
    assert len(mse_traditional_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_2 has length {len(mse_traditional_2)}"
    axis.plot(data_x, mse_traditional_2, marker="o",
              label=f"Arithmetic-mean-ensemble - 2")

    # Arithmetic-mean-ensemble - ensemble_size = 6
    # Expect only 1 data point for traditional originally
    mse_traditional_6 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 6]["loss_test"]), len(data_x))
    assert len(mse_traditional_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_6 has length {len(mse_traditional_6)}"
    axis.plot(data_x, mse_traditional_6, marker="o",
              label=f"Arithmetic-mean-ensemble - 6")

    # Arithmetic-mean-ensemble - ensemble_size = 12
    # Expect only 1 data point for traditional originally
    mse_traditional_12 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 12]["loss_test"]), len(data_x))
    assert len(mse_traditional_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_12 has length {len(mse_traditional_12)}"
    axis.plot(data_x, mse_traditional_12,
              marker="o", label=f"Arithmetic-mean-ensemble - 12")

    # MLP
    # Expect only 1 data point for MLP originally
    mse_mlp = np.repeat(np.array(df_mlp["loss_test"]), len(data_x))
    assert len(mse_mlp) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp has length {len(mse_mlp)}"
    axis.plot(data_x, mse_mlp, marker="o", label=f"MLP")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend(loc=f"upper center", bbox_to_anchor=(
        0.5, -0.25), ncol=2, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_correlation_penalty_coefficient_08_15_fixed_architecture_2_2_2_2_2_2(raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_static_ncl is not None and raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_static_ncl, raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_static_ncl = {
        "hidden_size": [
            [2, 2, 2, 2, 2, 2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5
        ]
    }
    filter_traditional = {
        "hidden_size": [
            [2, 2, 2, 2, 2, 2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "epoch_num": [
            400
        ]
    }
    filter_mlp = {
        "hidden_size": [
            [2, 2, 2, 2, 2, 2]
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(each base learner is [2, 2, 2, 2, 2, 2])"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"static_ncl_mse_against_correlation_penalty_coefficient_08_15_fixed_architecture_2_2_2_2_2_2.png"
    # ====================================================================

    df_static_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_static_ncl,
        filter=filter_static_ncl
    )
    df_traditional = data_helper.apply_dataframe_filter(
        df=raw_df_traditional,
        filter=filter_traditional
    )
    df_mlp = data_helper.apply_dataframe_filter(
        df=raw_df_mlp,
        filter=filter_mlp
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_static_ncl["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Static NCL - ensemble_size = 2
    mse_static_ncl_2 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 2]["loss_test"])
    assert len(mse_static_ncl_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_2 has length {len(mse_static_ncl_2)}"
    axis.plot(data_x, mse_static_ncl_2, marker="o", label=f"NCL - 2")

    # Static NCL - ensemble_size = 6
    mse_static_ncl_6 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 6]["loss_test"])
    assert len(mse_static_ncl_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_6 has length {len(mse_static_ncl_6)}"
    axis.plot(data_x, mse_static_ncl_6, marker="o", label=f"NCL - 6")

    # Static NCL - ensemble_size = 12
    mse_static_ncl_12 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 12]["loss_test"])
    assert len(mse_static_ncl_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_12 has length {len(mse_static_ncl_12)}"
    axis.plot(data_x, mse_static_ncl_12, marker="o", label=f"NCL - 12")

    # Arithmetic-mean-ensemble - ensemble_size = 2
    # Expect only 1 data point for traditional originally
    mse_traditional_2 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 2]["loss_test"]), len(data_x))
    assert len(mse_traditional_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_2 has length {len(mse_traditional_2)}"
    axis.plot(data_x, mse_traditional_2, marker="o",
              label=f"Arithmetic-mean-ensemble - 2")

    # Arithmetic-mean-ensemble - ensemble_size = 6
    # Expect only 1 data point for traditional originally
    mse_traditional_6 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 6]["loss_test"]), len(data_x))
    assert len(mse_traditional_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_6 has length {len(mse_traditional_6)}"
    axis.plot(data_x, mse_traditional_6, marker="o",
              label=f"Arithmetic-mean-ensemble - 6")

    # Arithmetic-mean-ensemble - ensemble_size = 12
    # Expect only 1 data point for traditional originally
    mse_traditional_12 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 12]["loss_test"]), len(data_x))
    assert len(mse_traditional_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_12 has length {len(mse_traditional_12)}"
    axis.plot(data_x, mse_traditional_12,
              marker="o", label=f"Arithmetic-mean-ensemble - 12")

    # MLP
    # Expect only 1 data point for MLP originally
    mse_mlp = np.repeat(np.array(df_mlp["loss_test"]), len(data_x))
    assert len(mse_mlp) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp has length {len(mse_mlp)}"
    axis.plot(data_x, mse_mlp, marker="o", label=f"MLP")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend(loc=f"upper center", bbox_to_anchor=(
        0.5, -0.25), ncol=2, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_6(raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_static_ncl is not None and raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_static_ncl, raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_static_ncl = {
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
        ]
    }
    filter_traditional = {
        "hidden_size": [
            [6]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "epoch_num": [
            400
        ]
    }
    filter_mlp = {
        "hidden_size": [
            [6]
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(each base learner is [6])"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"static_ncl_mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_6.png"
    # ====================================================================

    df_static_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_static_ncl,
        filter=filter_static_ncl
    )
    df_traditional = data_helper.apply_dataframe_filter(
        df=raw_df_traditional,
        filter=filter_traditional
    )
    df_mlp = data_helper.apply_dataframe_filter(
        df=raw_df_mlp,
        filter=filter_mlp
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_static_ncl["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Static NCL - ensemble_size = 2
    mse_static_ncl_2 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 2]["loss_test"])
    assert len(mse_static_ncl_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_2 has length {len(mse_static_ncl_2)}"
    axis.plot(data_x, mse_static_ncl_2, marker="o", label=f"NCL - 2")

    # Static NCL - ensemble_size = 6
    mse_static_ncl_6 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 6]["loss_test"])
    assert len(mse_static_ncl_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_6 has length {len(mse_static_ncl_6)}"
    axis.plot(data_x, mse_static_ncl_6, marker="o", label=f"NCL - 6")

    # Static NCL - ensemble_size = 12
    mse_static_ncl_12 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 12]["loss_test"])
    assert len(mse_static_ncl_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_12 has length {len(mse_static_ncl_12)}"
    axis.plot(data_x, mse_static_ncl_12, marker="o", label=f"NCL - 12")

    # Arithmetic-mean-ensemble - ensemble_size = 2
    # Expect only 1 data point for traditional originally
    mse_traditional_2 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 2]["loss_test"]), len(data_x))
    assert len(mse_traditional_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_2 has length {len(mse_traditional_2)}"
    axis.plot(data_x, mse_traditional_2, marker="o",
              label=f"Arithmetic-mean-ensemble - 2")

    # Arithmetic-mean-ensemble - ensemble_size = 6
    # Expect only 1 data point for traditional originally
    mse_traditional_6 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 6]["loss_test"]), len(data_x))
    assert len(mse_traditional_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_6 has length {len(mse_traditional_6)}"
    axis.plot(data_x, mse_traditional_6, marker="o",
              label=f"Arithmetic-mean-ensemble - 6")

    # Arithmetic-mean-ensemble - ensemble_size = 12
    # Expect only 1 data point for traditional originally
    mse_traditional_12 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 12]["loss_test"]), len(data_x))
    assert len(mse_traditional_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_12 has length {len(mse_traditional_12)}"
    axis.plot(data_x, mse_traditional_12,
              marker="o", label=f"Arithmetic-mean-ensemble - 12")

    # MLP
    # Expect only 1 data point for MLP originally
    mse_mlp = np.repeat(np.array(df_mlp["loss_test"]), len(data_x))
    assert len(mse_mlp) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp has length {len(mse_mlp)}"
    axis.plot(data_x, mse_mlp, marker="o", label=f"MLP")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend(loc=f"upper center", bbox_to_anchor=(
        0.5, -0.25), ncol=2, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_correlation_penalty_coefficient_08_15_fixed_architecture_6(raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_static_ncl is not None and raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_static_ncl, raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_static_ncl = {
        "hidden_size": [
            [6]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5
        ]
    }
    filter_traditional = {
        "hidden_size": [
            [6]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "epoch_num": [
            400
        ]
    }
    filter_mlp = {
        "hidden_size": [
            [6]
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(each base learner is [6])"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"static_ncl_mse_against_correlation_penalty_coefficient_08_15_fixed_architecture_6.png"
    # ====================================================================

    df_static_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_static_ncl,
        filter=filter_static_ncl
    )
    df_traditional = data_helper.apply_dataframe_filter(
        df=raw_df_traditional,
        filter=filter_traditional
    )
    df_mlp = data_helper.apply_dataframe_filter(
        df=raw_df_mlp,
        filter=filter_mlp
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_static_ncl["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Static NCL - ensemble_size = 2
    mse_static_ncl_2 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 2]["loss_test"])
    assert len(mse_static_ncl_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_2 has length {len(mse_static_ncl_2)}"
    axis.plot(data_x, mse_static_ncl_2, marker="o", label=f"NCL - 2")

    # Static NCL - ensemble_size = 6
    mse_static_ncl_6 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 6]["loss_test"])
    assert len(mse_static_ncl_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_6 has length {len(mse_static_ncl_6)}"
    axis.plot(data_x, mse_static_ncl_6, marker="o", label=f"NCL - 6")

    # Static NCL - ensemble_size = 12
    mse_static_ncl_12 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 12]["loss_test"])
    assert len(mse_static_ncl_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_12 has length {len(mse_static_ncl_12)}"
    axis.plot(data_x, mse_static_ncl_12, marker="o", label=f"NCL - 12")

    # Arithmetic-mean-ensemble - ensemble_size = 2
    # Expect only 1 data point for traditional originally
    mse_traditional_2 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 2]["loss_test"]), len(data_x))
    assert len(mse_traditional_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_2 has length {len(mse_traditional_2)}"
    axis.plot(data_x, mse_traditional_2, marker="o",
              label=f"Arithmetic-mean-ensemble - 2")

    # Arithmetic-mean-ensemble - ensemble_size = 6
    # Expect only 1 data point for traditional originally
    mse_traditional_6 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 6]["loss_test"]), len(data_x))
    assert len(mse_traditional_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_6 has length {len(mse_traditional_6)}"
    axis.plot(data_x, mse_traditional_6, marker="o",
              label=f"Arithmetic-mean-ensemble - 6")

    # Arithmetic-mean-ensemble - ensemble_size = 12
    # Expect only 1 data point for traditional originally
    mse_traditional_12 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 12]["loss_test"]), len(data_x))
    assert len(mse_traditional_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_12 has length {len(mse_traditional_12)}"
    axis.plot(data_x, mse_traditional_12,
              marker="o", label=f"Arithmetic-mean-ensemble - 12")

    # MLP
    # Expect only 1 data point for MLP originally
    mse_mlp = np.repeat(np.array(df_mlp["loss_test"]), len(data_x))
    assert len(mse_mlp) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp has length {len(mse_mlp)}"
    axis.plot(data_x, mse_mlp, marker="o", label=f"MLP")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend(loc=f"upper center", bbox_to_anchor=(
        0.5, -0.25), ncol=2, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_12(raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_static_ncl is not None and raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_static_ncl, raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_static_ncl = {
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
        ]
    }
    filter_traditional = {
        "hidden_size": [
            [12]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "epoch_num": [
            400
        ]
    }
    filter_mlp = {
        "hidden_size": [
            [12]
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(each base learner is [12])"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"static_ncl_mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_12.png"
    # ====================================================================

    df_static_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_static_ncl,
        filter=filter_static_ncl
    )
    df_traditional = data_helper.apply_dataframe_filter(
        df=raw_df_traditional,
        filter=filter_traditional
    )
    df_mlp = data_helper.apply_dataframe_filter(
        df=raw_df_mlp,
        filter=filter_mlp
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_static_ncl["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Static NCL - ensemble_size = 2
    mse_static_ncl_2 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 2]["loss_test"])
    assert len(mse_static_ncl_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_2 has length {len(mse_static_ncl_2)}"
    axis.plot(data_x, mse_static_ncl_2, marker="o", label=f"NCL - 2")

    # Static NCL - ensemble_size = 6
    mse_static_ncl_6 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 6]["loss_test"])
    assert len(mse_static_ncl_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_6 has length {len(mse_static_ncl_6)}"
    axis.plot(data_x, mse_static_ncl_6, marker="o", label=f"NCL - 6")

    # Static NCL - ensemble_size = 12
    mse_static_ncl_12 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 12]["loss_test"])
    assert len(mse_static_ncl_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_12 has length {len(mse_static_ncl_12)}"
    axis.plot(data_x, mse_static_ncl_12, marker="o", label=f"NCL - 12")

    # Arithmetic-mean-ensemble - ensemble_size = 2
    # Expect only 1 data point for traditional originally
    mse_traditional_2 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 2]["loss_test"]), len(data_x))
    assert len(mse_traditional_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_2 has length {len(mse_traditional_2)}"
    axis.plot(data_x, mse_traditional_2, marker="o",
              label=f"Arithmetic-mean-ensemble - 2")

    # Arithmetic-mean-ensemble - ensemble_size = 6
    # Expect only 1 data point for traditional originally
    mse_traditional_6 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 6]["loss_test"]), len(data_x))
    assert len(mse_traditional_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_6 has length {len(mse_traditional_6)}"
    axis.plot(data_x, mse_traditional_6, marker="o",
              label=f"Arithmetic-mean-ensemble - 6")

    # Arithmetic-mean-ensemble - ensemble_size = 12
    # Expect only 1 data point for traditional originally
    mse_traditional_12 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 12]["loss_test"]), len(data_x))
    assert len(mse_traditional_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_12 has length {len(mse_traditional_12)}"
    axis.plot(data_x, mse_traditional_12,
              marker="o", label=f"Arithmetic-mean-ensemble - 12")

    # MLP
    # Expect only 1 data point for MLP originally
    mse_mlp = np.repeat(np.array(df_mlp["loss_test"]), len(data_x))
    assert len(mse_mlp) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp has length {len(mse_mlp)}"
    axis.plot(data_x, mse_mlp, marker="o", label=f"MLP")

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


def mse_against_correlation_penalty_coefficient_08_15_fixed_architecture_12(raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_static_ncl is not None and raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_static_ncl, raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_static_ncl = {
        "hidden_size": [
            [12]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5
        ]
    }
    filter_traditional = {
        "hidden_size": [
            [12]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "epoch_num": [
            400
        ]
    }
    filter_mlp = {
        "hidden_size": [
            [12]
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(each base learner is [12])"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"static_ncl_mse_against_correlation_penalty_coefficient_08_15_fixed_architecture_12.png"
    # ====================================================================

    df_static_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_static_ncl,
        filter=filter_static_ncl
    )
    df_traditional = data_helper.apply_dataframe_filter(
        df=raw_df_traditional,
        filter=filter_traditional
    )
    df_mlp = data_helper.apply_dataframe_filter(
        df=raw_df_mlp,
        filter=filter_mlp
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_static_ncl["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Static NCL - ensemble_size = 2
    mse_static_ncl_2 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 2]["loss_test"])
    assert len(mse_static_ncl_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_2 has length {len(mse_static_ncl_2)}"
    axis.plot(data_x, mse_static_ncl_2, marker="o", label=f"NCL - 2")

    # Static NCL - ensemble_size = 6
    mse_static_ncl_6 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 6]["loss_test"])
    assert len(mse_static_ncl_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_6 has length {len(mse_static_ncl_6)}"
    axis.plot(data_x, mse_static_ncl_6, marker="o", label=f"NCL - 6")

    # Static NCL - ensemble_size = 12
    mse_static_ncl_12 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 12]["loss_test"])
    assert len(mse_static_ncl_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_12 has length {len(mse_static_ncl_12)}"
    axis.plot(data_x, mse_static_ncl_12, marker="o", label=f"NCL - 12")

    # Arithmetic-mean-ensemble - ensemble_size = 2
    # Expect only 1 data point for traditional originally
    mse_traditional_2 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 2]["loss_test"]), len(data_x))
    assert len(mse_traditional_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_2 has length {len(mse_traditional_2)}"
    axis.plot(data_x, mse_traditional_2, marker="o",
              label=f"Arithmetic-mean-ensemble - 2")

    # Arithmetic-mean-ensemble - ensemble_size = 6
    # Expect only 1 data point for traditional originally
    mse_traditional_6 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 6]["loss_test"]), len(data_x))
    assert len(mse_traditional_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_6 has length {len(mse_traditional_6)}"
    axis.plot(data_x, mse_traditional_6, marker="o",
              label=f"Arithmetic-mean-ensemble - 6")

    # Arithmetic-mean-ensemble - ensemble_size = 12
    # Expect only 1 data point for traditional originally
    mse_traditional_12 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 12]["loss_test"]), len(data_x))
    assert len(mse_traditional_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_12 has length {len(mse_traditional_12)}"
    axis.plot(data_x, mse_traditional_12,
              marker="o", label=f"Arithmetic-mean-ensemble - 12")

    # MLP
    # Expect only 1 data point for MLP originally
    mse_mlp = np.repeat(np.array(df_mlp["loss_test"]), len(data_x))
    assert len(mse_mlp) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp has length {len(mse_mlp)}"
    axis.plot(data_x, mse_mlp, marker="o", label=f"MLP")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend(loc=f"upper center", bbox_to_anchor=(
        0.5, -0.25), ncol=2, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_correlation_penalty_coefficient_0_1_fixed_ensemble_size_6_architecture_2_2_2_comparator_traditional(raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_static_ncl is not None and raw_df_traditional is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_static_ncl, raw_df_traditional, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_static_ncl = {
        "hidden_size": [
            [2],
            [2, 2, 2],
            [2, 2, 2, 2, 2, 2]
        ],
        "ensemble_size": [
            6
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
        ]
    }
    filter_traditional = {
        "hidden_size": [
            [2],
            [2, 2, 2],
            [2, 2, 2, 2, 2, 2]
        ],
        "ensemble_size": [
            6
        ],
        "epoch_num": [
            400
        ]
    }
    filter_mlp = {
        "hidden_size": [
            [2],
            [2, 2, 2],
            [2, 2, 2, 2, 2, 2]
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(6 base learners per ensemble model)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"static_ncl_mse_against_correlation_penalty_coefficient_0_1_fixed_ensemble_size_6_architecture_2_2_2_comparator_traditional.png"
    # ====================================================================

    df_static_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_static_ncl,
        filter=filter_static_ncl
    )
    df_traditional = data_helper.apply_dataframe_filter(
        df=raw_df_traditional,
        filter=filter_traditional
    )
    df_mlp = data_helper.apply_dataframe_filter(
        df=raw_df_mlp,
        filter=filter_mlp
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_static_ncl["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Static NCL - hidden_size = [2]
    mse_static_ncl_2 = np.array(
        df_static_ncl[df_static_ncl["hidden_size"].apply(lambda x: x == [2])]["loss_test"])
    assert len(mse_static_ncl_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_2 has length {len(mse_static_ncl_2)}"
    axis.plot(data_x, mse_static_ncl_2, marker="o", label=f"NCL - [2]")

    # Static NCL - hidden_size = [2, 2, 2]
    mse_static_ncl_2_2_2 = np.array(df_static_ncl[df_static_ncl["hidden_size"].apply(
        lambda x: x == [2, 2, 2])]["loss_test"])
    assert len(mse_static_ncl_2_2_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_2_2_2 has length {len(mse_static_ncl_2_2_2)}"
    axis.plot(data_x, mse_static_ncl_2_2_2,
              marker="o", label=f"NCL - [2, 2, 2]")

    # Static NCL - hidden_size = [2, 2, 2, 2, 2, 2]
    mse_static_ncl_2_2_2_2_2_2 = np.array(df_static_ncl[df_static_ncl["hidden_size"].apply(
        lambda x: x == [2, 2, 2, 2, 2, 2])]["loss_test"])
    assert len(mse_static_ncl_2_2_2_2_2_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_2_2_2_2_2_2 has length {len(mse_static_ncl_2_2_2_2_2_2)}"
    axis.plot(data_x, mse_static_ncl_2_2_2_2_2_2,
              marker="o", label=f"NCL - [2, 2, 2, 2, 2, 2]")

    # Arithmetic-mean-ensemble - hidden_size = [2]
    mse_traditional_2 = np.repeat(np.array(df_traditional[df_traditional["hidden_size"].apply(
        lambda x: x == [2])]["loss_test"]), len(data_x))
    assert len(mse_traditional_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_2 has length {len(mse_traditional_2)}"
    axis.plot(data_x, mse_traditional_2, marker="o",
              label=f"Arithmetic-mean-ensemble - [2]")

    # Arithmetic-mean-ensemble - hidden_size = [2, 2, 2]
    mse_traditional_2_2_2 = np.repeat(np.array(df_traditional[df_traditional["hidden_size"].apply(
        lambda x: x == [2, 2, 2])]["loss_test"]), len(data_x))
    assert len(mse_traditional_2_2_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_2_2_2 has length {len(mse_traditional_2_2_2)}"
    axis.plot(data_x, mse_traditional_2_2_2, marker="o",
              label=f"Arithmetic-mean-ensemble - [2, 2, 2]")

    # Arithmetic-mean-ensemble - hidden_size = [2, 2, 2, 2, 2, 2]
    mse_traditional_2_2_2_2_2_2 = np.repeat(np.array(df_traditional[df_traditional["hidden_size"].apply(
        lambda x: x == [2, 2, 2, 2, 2, 2])]["loss_test"]), len(data_x))
    assert len(mse_traditional_2_2_2_2_2_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_2_2_2_2_2_2 has length {len(mse_traditional_2_2_2_2_2_2)}"
    axis.plot(data_x, mse_traditional_2_2_2_2_2_2, marker="o",
              label=f"Arithmetic-mean-ensemble - [2, 2, 2, 2, 2, 2]")

    # # MLP - hidden_size = [2]
    # # Expect only 1 data point for MLP originally
    # mse_mlp_2 = np.repeat(np.array(df_mlp[df_mlp["hidden_size"].apply(
    #     lambda x: x == [2])]["loss_test"]), len(data_x))
    # assert len(mse_mlp_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp_2 has length {len(mse_mlp_2)}"
    # axis.plot(data_x, mse_mlp_2, marker="o", label=f"MLP - [2]")

    # # MLP - hidden_size = [2, 2, 2]
    # # Expect only 1 data point for MLP originally
    # mse_mlp_2_2_2 = np.repeat(np.array(df_mlp[df_mlp["hidden_size"].apply(
    #     lambda x: x == [2, 2, 2])]["loss_test"]), len(data_x))
    # assert len(mse_mlp_2_2_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp_2_2_2 has length {len(mse_mlp_2_2_2)}"
    # axis.plot(data_x, mse_mlp_2_2_2, marker="o", label=f"MLP - [2, 2, 2]")

    # # MLP - hidden_size = [2, 2, 2, 2, 2, 2]
    # # Expect only 1 data point for MLP originally
    # mse_mlp_2_2_2_2_2_2 = np.repeat(np.array(df_mlp[df_mlp["hidden_size"].apply(
    #     lambda x: x == [2, 2, 2, 2, 2, 2])]["loss_test"]), len(data_x))
    # assert len(mse_mlp_2_2_2_2_2_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp_2_2_2_2_2_2 has length {len(mse_mlp_2_2_2_2_2_2)}"
    # axis.plot(data_x, mse_mlp_2_2_2_2_2_2, marker="o",
    #           label=f"MLP - [2, 2, 2, 2, 2, 2]")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend(loc=f"upper center", bbox_to_anchor=(
        0.5, -0.25), ncol=2, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_correlation_penalty_coefficient_08_15_fixed_ensemble_size_6_architecture_2_2_2_comparator_traditional(raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_static_ncl is not None and raw_df_traditional is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_static_ncl, raw_df_traditional, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_static_ncl = {
        "hidden_size": [
            [2],
            [2, 2, 2],
            [2, 2, 2, 2, 2, 2]
        ],
        "ensemble_size": [
            6
        ],
        "correlation_penalty_coefficient": [
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5
        ]
    }
    filter_traditional = {
        "hidden_size": [
            [2],
            [2, 2, 2],
            [2, 2, 2, 2, 2, 2]
        ],
        "ensemble_size": [
            6
        ],
        "epoch_num": [
            400
        ]
    }
    filter_mlp = {
        "hidden_size": [
            [2],
            [2, 2, 2],
            [2, 2, 2, 2, 2, 2]
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(6 base learners per ensemble model)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"static_ncl_mse_against_correlation_penalty_coefficient_08_15_fixed_ensemble_size_6_architecture_2_2_2_comparator_traditional.png"
    # ====================================================================

    df_static_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_static_ncl,
        filter=filter_static_ncl
    )
    df_traditional = data_helper.apply_dataframe_filter(
        df=raw_df_traditional,
        filter=filter_traditional
    )
    df_mlp = data_helper.apply_dataframe_filter(
        df=raw_df_mlp,
        filter=filter_mlp
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_static_ncl["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Static NCL - hidden_size = [2]
    mse_static_ncl_2 = np.array(
        df_static_ncl[df_static_ncl["hidden_size"].apply(lambda x: x == [2])]["loss_test"])
    assert len(mse_static_ncl_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_2 has length {len(mse_static_ncl_2)}"
    axis.plot(data_x, mse_static_ncl_2, marker="o", label=f"NCL - [2]")

    # Static NCL - hidden_size = [2, 2, 2]
    mse_static_ncl_2_2_2 = np.array(df_static_ncl[df_static_ncl["hidden_size"].apply(
        lambda x: x == [2, 2, 2])]["loss_test"])
    assert len(mse_static_ncl_2_2_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_2_2_2 has length {len(mse_static_ncl_2_2_2)}"
    axis.plot(data_x, mse_static_ncl_2_2_2,
              marker="o", label=f"NCL - [2, 2, 2]")

    # Static NCL - hidden_size = [2, 2, 2, 2, 2, 2]
    mse_static_ncl_2_2_2_2_2_2 = np.array(df_static_ncl[df_static_ncl["hidden_size"].apply(
        lambda x: x == [2, 2, 2, 2, 2, 2])]["loss_test"])
    assert len(mse_static_ncl_2_2_2_2_2_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_2_2_2_2_2_2 has length {len(mse_static_ncl_2_2_2_2_2_2)}"
    axis.plot(data_x, mse_static_ncl_2_2_2_2_2_2,
              marker="o", label=f"NCL - [2, 2, 2, 2, 2, 2]")

    # Arithmetic-mean-ensemble - hidden_size = [2]
    mse_traditional_2 = np.repeat(np.array(df_traditional[df_traditional["hidden_size"].apply(
        lambda x: x == [2])]["loss_test"]), len(data_x))
    assert len(mse_traditional_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_2 has length {len(mse_traditional_2)}"
    axis.plot(data_x, mse_traditional_2, marker="o",
              label=f"Arithmetic-mean-ensemble - [2]")

    # Arithmetic-mean-ensemble - hidden_size = [2, 2, 2]
    mse_traditional_2_2_2 = np.repeat(np.array(df_traditional[df_traditional["hidden_size"].apply(
        lambda x: x == [2, 2, 2])]["loss_test"]), len(data_x))
    assert len(mse_traditional_2_2_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_2_2_2 has length {len(mse_traditional_2_2_2)}"
    axis.plot(data_x, mse_traditional_2_2_2, marker="o",
              label=f"Arithmetic-mean-ensemble - [2, 2, 2]")

    # Arithmetic-mean-ensemble - hidden_size = [2, 2, 2, 2, 2, 2]
    mse_traditional_2_2_2_2_2_2 = np.repeat(np.array(df_traditional[df_traditional["hidden_size"].apply(
        lambda x: x == [2, 2, 2, 2, 2, 2])]["loss_test"]), len(data_x))
    assert len(mse_traditional_2_2_2_2_2_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_2_2_2_2_2_2 has length {len(mse_traditional_2_2_2_2_2_2)}"
    axis.plot(data_x, mse_traditional_2_2_2_2_2_2, marker="o",
              label=f"Arithmetic-mean-ensemble - [2, 2, 2, 2, 2, 2]")

    # # MLP - hidden_size = [2]
    # # Expect only 1 data point for MLP originally
    # mse_mlp_2 = np.repeat(np.array(df_mlp[df_mlp["hidden_size"].apply(
    #     lambda x: x == [2])]["loss_test"]), len(data_x))
    # assert len(mse_mlp_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp_2 has length {len(mse_mlp_2)}"
    # axis.plot(data_x, mse_mlp_2, marker="o", label=f"MLP - [2]")

    # # MLP - hidden_size = [2, 2, 2]
    # # Expect only 1 data point for MLP originally
    # mse_mlp_2_2_2 = np.repeat(np.array(df_mlp[df_mlp["hidden_size"].apply(
    #     lambda x: x == [2, 2, 2])]["loss_test"]), len(data_x))
    # assert len(mse_mlp_2_2_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp_2_2_2 has length {len(mse_mlp_2_2_2)}"
    # axis.plot(data_x, mse_mlp_2_2_2, marker="o", label=f"MLP - [2, 2, 2]")

    # # MLP - hidden_size = [2, 2, 2, 2, 2, 2]
    # # Expect only 1 data point for MLP originally
    # mse_mlp_2_2_2_2_2_2 = np.repeat(np.array(df_mlp[df_mlp["hidden_size"].apply(
    #     lambda x: x == [2, 2, 2, 2, 2, 2])]["loss_test"]), len(data_x))
    # assert len(mse_mlp_2_2_2_2_2_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp_2_2_2_2_2_2 has length {len(mse_mlp_2_2_2_2_2_2)}"
    # axis.plot(data_x, mse_mlp_2_2_2_2_2_2, marker="o",
    #           label=f"MLP - [2, 2, 2, 2, 2, 2]")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend(loc=f"upper center", bbox_to_anchor=(
        0.5, -0.25), ncol=2, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_correlation_penalty_coefficient_0_1_fixed_ensemble_size_6_architecture_6_comparator_traditional(raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_static_ncl is not None and raw_df_traditional is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_static_ncl, raw_df_traditional, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_static_ncl = {
        "hidden_size": [
            [2],
            [6],
            [12]
        ],
        "ensemble_size": [
            6
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
        ]
    }
    filter_traditional = {
        "hidden_size": [
            [2],
            [6],
            [12]
        ],
        "ensemble_size": [
            6
        ],
        "epoch_num": [
            400
        ]
    }
    filter_mlp = {
        "hidden_size": [
            [2],
            [6],
            [12]
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(6 base learners per ensemble model)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"static_ncl_mse_against_correlation_penalty_coefficient_0_1_fixed_ensemble_size_6_architecture_6_comparator_traditional.png"
    # ====================================================================

    df_static_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_static_ncl,
        filter=filter_static_ncl
    )
    df_traditional = data_helper.apply_dataframe_filter(
        df=raw_df_traditional,
        filter=filter_traditional
    )
    df_mlp = data_helper.apply_dataframe_filter(
        df=raw_df_mlp,
        filter=filter_mlp
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_static_ncl["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Static NCL - hidden_size = [2]
    mse_static_ncl_2 = np.array(
        df_static_ncl[df_static_ncl["hidden_size"].apply(lambda x: x == [2])]["loss_test"])
    assert len(mse_static_ncl_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_2 has length {len(mse_static_ncl_2)}"
    axis.plot(data_x, mse_static_ncl_2, marker="o", label=f"NCL - [2]")

    # Static NCL - hidden_size = [2, 2, 2]
    mse_static_ncl_6 = np.array(
        df_static_ncl[df_static_ncl["hidden_size"].apply(lambda x: x == [6])]["loss_test"])
    assert len(mse_static_ncl_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_6 has length {len(mse_static_ncl_6)}"
    axis.plot(data_x, mse_static_ncl_6, marker="o", label=f"NCL - [6]")

    # Static NCL - hidden_size = [12]
    mse_static_ncl_12 = np.array(
        df_static_ncl[df_static_ncl["hidden_size"].apply(lambda x: x == [12])]["loss_test"])
    assert len(mse_static_ncl_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_12 has length {len(mse_static_ncl_12)}"
    axis.plot(data_x, mse_static_ncl_12, marker="o", label=f"NCL - [12]")

    # Arithmetic-mean-ensemble - hidden_size = [2]
    mse_traditional_2 = np.repeat(np.array(df_traditional[df_traditional["hidden_size"].apply(
        lambda x: x == [2])]["loss_test"]), len(data_x))
    assert len(mse_traditional_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_2 has length {len(mse_traditional_2)}"
    axis.plot(data_x, mse_traditional_2, marker="o",
              label=f"Arithmetic-mean-ensemble - [2]")

    # Arithmetic-mean-ensemble - hidden_size = [6]
    mse_traditional_6 = np.repeat(np.array(df_traditional[df_traditional["hidden_size"].apply(
        lambda x: x == [6])]["loss_test"]), len(data_x))
    assert len(mse_traditional_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_6 has length {len(mse_traditional_6)}"
    axis.plot(data_x, mse_traditional_6, marker="o",
              label=f"Arithmetic-mean-ensemble - [6]")

    # Arithmetic-mean-ensemble - hidden_size = [12]
    mse_traditional_12 = np.repeat(np.array(df_traditional[df_traditional["hidden_size"].apply(
        lambda x: x == [12])]["loss_test"]), len(data_x))
    assert len(mse_traditional_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_12 has length {len(mse_traditional_12)}"
    axis.plot(data_x, mse_traditional_12, marker="o",
              label=f"Arithmetic-mean-ensemble - [12]")

    # # MLP - hidden_size = [2]
    # # Expect only 1 data point for MLP originally
    # mse_mlp_2 = np.repeat(np.array(df_mlp[df_mlp["hidden_size"].apply(
    #     lambda x: x == [2])]["loss_test"]), len(data_x))
    # assert len(mse_mlp_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp_2 has length {len(mse_mlp_2)}"
    # axis.plot(data_x, mse_mlp_2, marker="o", label=f"MLP - [2]")

    # # MLP - hidden_size = [6]
    # # Expect only 1 data point for MLP originally
    # mse_mlp_6 = np.repeat(np.array(df_mlp[df_mlp["hidden_size"].apply(
    #     lambda x: x == [6])]["loss_test"]), len(data_x))
    # assert len(mse_mlp_6) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp_6 has length {len(mse_mlp_6)}"
    # axis.plot(data_x, mse_mlp_6, marker="o", label=f"MLP - [6]")

    # # MLP - hidden_size = [12]
    # # Expect only 1 data point for MLP originally
    # mse_mlp_12 = np.repeat(np.array(df_mlp[df_mlp["hidden_size"].apply(
    #     lambda x: x == [12])]["loss_test"]), len(data_x))
    # assert len(mse_mlp_12) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp_12 has length {len(mse_mlp_12)}"
    # axis.plot(data_x, mse_mlp_12, marker="o", label=f"MLP - [12]")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend(loc=f"upper center", bbox_to_anchor=(
        0.5, -0.25), ncol=2, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_correlation_penalty_coefficient_08_15_fixed_ensemble_size_6_architecture_6_comparator_traditional(raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_static_ncl is not None and raw_df_traditional is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_static_ncl, raw_df_traditional, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_static_ncl = {
        "hidden_size": [
            [2],
            [6],
            [12]
        ],
        "ensemble_size": [
            6
        ],
        "correlation_penalty_coefficient": [
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5
        ]
    }
    filter_traditional = {
        "hidden_size": [
            [2],
            [6],
            [12]
        ],
        "ensemble_size": [
            6
        ],
        "epoch_num": [
            400
        ]
    }
    filter_mlp = {
        "hidden_size": [
            [2],
            [6],
            [12]
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(6 base learners per ensemble model)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"static_ncl_mse_against_correlation_penalty_coefficient_08_15_fixed_ensemble_size_6_architecture_6_comparator_traditional.png"
    # ====================================================================

    df_static_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_static_ncl,
        filter=filter_static_ncl
    )
    df_traditional = data_helper.apply_dataframe_filter(
        df=raw_df_traditional,
        filter=filter_traditional
    )
    df_mlp = data_helper.apply_dataframe_filter(
        df=raw_df_mlp,
        filter=filter_mlp
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_static_ncl["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Static NCL - hidden_size = [2]
    mse_static_ncl_2 = np.array(
        df_static_ncl[df_static_ncl["hidden_size"].apply(lambda x: x == [2])]["loss_test"])
    assert len(mse_static_ncl_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_2 has length {len(mse_static_ncl_2)}"
    axis.plot(data_x, mse_static_ncl_2, marker="o", label=f"NCL - [2]")

    # Static NCL - hidden_size = [2, 2, 2]
    mse_static_ncl_6 = np.array(
        df_static_ncl[df_static_ncl["hidden_size"].apply(lambda x: x == [6])]["loss_test"])
    assert len(mse_static_ncl_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_6 has length {len(mse_static_ncl_6)}"
    axis.plot(data_x, mse_static_ncl_6, marker="o", label=f"NCL - [6]")

    # Static NCL - hidden_size = [12]
    mse_static_ncl_12 = np.array(
        df_static_ncl[df_static_ncl["hidden_size"].apply(lambda x: x == [12])]["loss_test"])
    assert len(mse_static_ncl_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_12 has length {len(mse_static_ncl_12)}"
    axis.plot(data_x, mse_static_ncl_12, marker="o", label=f"NCL - [12]")

    # Arithmetic-mean-ensemble - hidden_size = [2]
    mse_traditional_2 = np.repeat(np.array(df_traditional[df_traditional["hidden_size"].apply(
        lambda x: x == [2])]["loss_test"]), len(data_x))
    assert len(mse_traditional_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_2 has length {len(mse_traditional_2)}"
    axis.plot(data_x, mse_traditional_2, marker="o",
              label=f"Arithmetic-mean-ensemble - [2]")

    # Arithmetic-mean-ensemble - hidden_size = [6]
    mse_traditional_6 = np.repeat(np.array(df_traditional[df_traditional["hidden_size"].apply(
        lambda x: x == [6])]["loss_test"]), len(data_x))
    assert len(mse_traditional_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_6 has length {len(mse_traditional_6)}"
    axis.plot(data_x, mse_traditional_6, marker="o",
              label=f"Arithmetic-mean-ensemble - [6]")

    # Arithmetic-mean-ensemble - hidden_size = [12]
    mse_traditional_12 = np.repeat(np.array(df_traditional[df_traditional["hidden_size"].apply(
        lambda x: x == [12])]["loss_test"]), len(data_x))
    assert len(mse_traditional_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_12 has length {len(mse_traditional_12)}"
    axis.plot(data_x, mse_traditional_12, marker="o",
              label=f"Arithmetic-mean-ensemble - [12]")

    # # MLP - hidden_size = [2]
    # # Expect only 1 data point for MLP originally
    # mse_mlp_2 = np.repeat(np.array(df_mlp[df_mlp["hidden_size"].apply(
    #     lambda x: x == [2])]["loss_test"]), len(data_x))
    # assert len(mse_mlp_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp_2 has length {len(mse_mlp_2)}"
    # axis.plot(data_x, mse_mlp_2, marker="o", label=f"MLP - [2]")

    # # MLP - hidden_size = [6]
    # # Expect only 1 data point for MLP originally
    # mse_mlp_6 = np.repeat(np.array(df_mlp[df_mlp["hidden_size"].apply(
    #     lambda x: x == [6])]["loss_test"]), len(data_x))
    # assert len(mse_mlp_6) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp_6 has length {len(mse_mlp_6)}"
    # axis.plot(data_x, mse_mlp_6, marker="o", label=f"MLP - [6]")

    # # MLP - hidden_size = [12]
    # # Expect only 1 data point for MLP originally
    # mse_mlp_12 = np.repeat(np.array(df_mlp[df_mlp["hidden_size"].apply(
    #     lambda x: x == [12])]["loss_test"]), len(data_x))
    # assert len(mse_mlp_12) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp_12 has length {len(mse_mlp_12)}"
    # axis.plot(data_x, mse_mlp_12, marker="o", label=f"MLP - [12]")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend(loc=f"upper center", bbox_to_anchor=(
        0.5, -0.25), ncol=2, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_correlation_penalty_coefficient_0_1_fixed_ensemble_size_6_architecture_2_2_2_comparator_mlp(raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_static_ncl is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_static_ncl, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_static_ncl = {
        "hidden_size": [
            [2],
            [2, 2, 2],
            [2, 2, 2, 2, 2, 2]
        ],
        "ensemble_size": [
            6
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
        ]
    }
    filter_traditional = {
        "hidden_size": [
            [2],
            [2, 2, 2],
            [2, 2, 2, 2, 2, 2]
        ],
        "ensemble_size": [
            6
        ],
        "epoch_num": [
            400
        ]
    }
    filter_mlp = {
        "hidden_size": [
            [2],
            [2, 2, 2],
            [2, 2, 2, 2, 2, 2]
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(6 base learners per ensemble model)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"static_ncl_mse_against_correlation_penalty_coefficient_0_1_fixed_ensemble_size_6_architecture_2_2_2_comparator_mlp.png"
    # ====================================================================

    df_static_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_static_ncl,
        filter=filter_static_ncl
    )
    df_traditional = data_helper.apply_dataframe_filter(
        df=raw_df_traditional,
        filter=filter_traditional
    )
    df_mlp = data_helper.apply_dataframe_filter(
        df=raw_df_mlp,
        filter=filter_mlp
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_static_ncl["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Static NCL - hidden_size = [2]
    mse_static_ncl_2 = np.array(
        df_static_ncl[df_static_ncl["hidden_size"].apply(lambda x: x == [2])]["loss_test"])
    assert len(mse_static_ncl_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_2 has length {len(mse_static_ncl_2)}"
    axis.plot(data_x, mse_static_ncl_2, marker="o", label=f"NCL - [2]")

    # Static NCL - hidden_size = [2, 2, 2]
    mse_static_ncl_2_2_2 = np.array(df_static_ncl[df_static_ncl["hidden_size"].apply(
        lambda x: x == [2, 2, 2])]["loss_test"])
    assert len(mse_static_ncl_2_2_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_2_2_2 has length {len(mse_static_ncl_2_2_2)}"
    axis.plot(data_x, mse_static_ncl_2_2_2,
              marker="o", label=f"NCL - [2, 2, 2]")

    # Static NCL - hidden_size = [2, 2, 2, 2, 2, 2]
    mse_static_ncl_2_2_2_2_2_2 = np.array(df_static_ncl[df_static_ncl["hidden_size"].apply(
        lambda x: x == [2, 2, 2, 2, 2, 2])]["loss_test"])
    assert len(mse_static_ncl_2_2_2_2_2_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_2_2_2_2_2_2 has length {len(mse_static_ncl_2_2_2_2_2_2)}"
    axis.plot(data_x, mse_static_ncl_2_2_2_2_2_2,
              marker="o", label=f"NCL - [2, 2, 2, 2, 2, 2]")

    # # Arithmetic-mean-ensemble - hidden_size = [2]
    # mse_traditional_2 = np.repeat(np.array(df_traditional[df_traditional["hidden_size"].apply(
    #     lambda x: x == [2])]["loss_test"]), len(data_x))
    # assert len(mse_traditional_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_2 has length {len(mse_traditional_2)}"
    # axis.plot(data_x, mse_traditional_2, marker="o",
    #           label=f"Arithmetic-mean-ensemble - [2]")

    # # Arithmetic-mean-ensemble - hidden_size = [2, 2, 2]
    # mse_traditional_2_2_2 = np.repeat(np.array(df_traditional[df_traditional["hidden_size"].apply(
    #     lambda x: x == [2, 2, 2])]["loss_test"]), len(data_x))
    # assert len(mse_traditional_2_2_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_2_2_2 has length {len(mse_traditional_2_2_2)}"
    # axis.plot(data_x, mse_traditional_2_2_2, marker="o",
    #           label=f"Arithmetic-mean-ensemble - [2, 2, 2]")

    # # Arithmetic-mean-ensemble - hidden_size = [2, 2, 2, 2, 2, 2]
    # mse_traditional_2_2_2_2_2_2 = np.repeat(np.array(df_traditional[df_traditional["hidden_size"].apply(
    #     lambda x: x == [2, 2, 2, 2, 2, 2])]["loss_test"]), len(data_x))
    # assert len(mse_traditional_2_2_2_2_2_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_2_2_2_2_2_2 has length {len(mse_traditional_2_2_2_2_2_2)}"
    # axis.plot(data_x, mse_traditional_2_2_2_2_2_2, marker="o",
    #           label=f"Arithmetic-mean-ensemble - [2, 2, 2, 2, 2, 2]")

    # MLP - hidden_size = [2]
    # Expect only 1 data point for MLP originally
    mse_mlp_2 = np.repeat(np.array(df_mlp[df_mlp["hidden_size"].apply(
        lambda x: x == [2])]["loss_test"]), len(data_x))
    assert len(mse_mlp_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp_2 has length {len(mse_mlp_2)}"
    axis.plot(data_x, mse_mlp_2, marker="o", label=f"MLP - [2]")

    # MLP - hidden_size = [2, 2, 2]
    # Expect only 1 data point for MLP originally
    mse_mlp_2_2_2 = np.repeat(np.array(df_mlp[df_mlp["hidden_size"].apply(
        lambda x: x == [2, 2, 2])]["loss_test"]), len(data_x))
    assert len(mse_mlp_2_2_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp_2_2_2 has length {len(mse_mlp_2_2_2)}"
    axis.plot(data_x, mse_mlp_2_2_2, marker="o", label=f"MLP - [2, 2, 2]")

    # MLP - hidden_size = [2, 2, 2, 2, 2, 2]
    # Expect only 1 data point for MLP originally
    mse_mlp_2_2_2_2_2_2 = np.repeat(np.array(df_mlp[df_mlp["hidden_size"].apply(
        lambda x: x == [2, 2, 2, 2, 2, 2])]["loss_test"]), len(data_x))
    assert len(mse_mlp_2_2_2_2_2_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp_2_2_2_2_2_2 has length {len(mse_mlp_2_2_2_2_2_2)}"
    axis.plot(data_x, mse_mlp_2_2_2_2_2_2, marker="o",
              label=f"MLP - [2, 2, 2, 2, 2, 2]")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend(loc=f"upper center", bbox_to_anchor=(
        0.5, -0.25), ncol=2, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_correlation_penalty_coefficient_08_15_fixed_ensemble_size_6_architecture_2_2_2_comparator_mlp(raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_static_ncl is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_static_ncl, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_static_ncl = {
        "hidden_size": [
            [2],
            [2, 2, 2],
            [2, 2, 2, 2, 2, 2]
        ],
        "ensemble_size": [
            6
        ],
        "correlation_penalty_coefficient": [
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5
        ]
    }
    filter_traditional = {
        "hidden_size": [
            [2],
            [2, 2, 2],
            [2, 2, 2, 2, 2, 2]
        ],
        "ensemble_size": [
            6
        ],
        "epoch_num": [
            400
        ]
    }
    filter_mlp = {
        "hidden_size": [
            [2],
            [2, 2, 2],
            [2, 2, 2, 2, 2, 2]
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(6 base learners per ensemble model)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"static_ncl_mse_against_correlation_penalty_coefficient_08_15_fixed_ensemble_size_6_architecture_2_2_2_comparator_mlp.png"
    # ====================================================================

    df_static_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_static_ncl,
        filter=filter_static_ncl
    )
    df_traditional = data_helper.apply_dataframe_filter(
        df=raw_df_traditional,
        filter=filter_traditional
    )
    df_mlp = data_helper.apply_dataframe_filter(
        df=raw_df_mlp,
        filter=filter_mlp
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_static_ncl["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Static NCL - hidden_size = [2]
    mse_static_ncl_2 = np.array(
        df_static_ncl[df_static_ncl["hidden_size"].apply(lambda x: x == [2])]["loss_test"])
    assert len(mse_static_ncl_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_2 has length {len(mse_static_ncl_2)}"
    axis.plot(data_x, mse_static_ncl_2, marker="o", label=f"NCL - [2]")

    # Static NCL - hidden_size = [2, 2, 2]
    mse_static_ncl_2_2_2 = np.array(df_static_ncl[df_static_ncl["hidden_size"].apply(
        lambda x: x == [2, 2, 2])]["loss_test"])
    assert len(mse_static_ncl_2_2_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_2_2_2 has length {len(mse_static_ncl_2_2_2)}"
    axis.plot(data_x, mse_static_ncl_2_2_2,
              marker="o", label=f"NCL - [2, 2, 2]")

    # Static NCL - hidden_size = [2, 2, 2, 2, 2, 2]
    mse_static_ncl_2_2_2_2_2_2 = np.array(df_static_ncl[df_static_ncl["hidden_size"].apply(
        lambda x: x == [2, 2, 2, 2, 2, 2])]["loss_test"])
    assert len(mse_static_ncl_2_2_2_2_2_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_2_2_2_2_2_2 has length {len(mse_static_ncl_2_2_2_2_2_2)}"
    axis.plot(data_x, mse_static_ncl_2_2_2_2_2_2,
              marker="o", label=f"NCL - [2, 2, 2, 2, 2, 2]")

    # # Arithmetic-mean-ensemble - hidden_size = [2]
    # mse_traditional_2 = np.repeat(np.array(df_traditional[df_traditional["hidden_size"].apply(
    #     lambda x: x == [2])]["loss_test"]), len(data_x))
    # assert len(mse_traditional_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_2 has length {len(mse_traditional_2)}"
    # axis.plot(data_x, mse_traditional_2, marker="o",
    #           label=f"Arithmetic-mean-ensemble - [2]")

    # # Arithmetic-mean-ensemble - hidden_size = [2, 2, 2]
    # mse_traditional_2_2_2 = np.repeat(np.array(df_traditional[df_traditional["hidden_size"].apply(
    #     lambda x: x == [2, 2, 2])]["loss_test"]), len(data_x))
    # assert len(mse_traditional_2_2_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_2_2_2 has length {len(mse_traditional_2_2_2)}"
    # axis.plot(data_x, mse_traditional_2_2_2, marker="o",
    #           label=f"Arithmetic-mean-ensemble - [2, 2, 2]")

    # # Arithmetic-mean-ensemble - hidden_size = [2, 2, 2, 2, 2, 2]
    # mse_traditional_2_2_2_2_2_2 = np.repeat(np.array(df_traditional[df_traditional["hidden_size"].apply(
    #     lambda x: x == [2, 2, 2, 2, 2, 2])]["loss_test"]), len(data_x))
    # assert len(mse_traditional_2_2_2_2_2_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_2_2_2_2_2_2 has length {len(mse_traditional_2_2_2_2_2_2)}"
    # axis.plot(data_x, mse_traditional_2_2_2_2_2_2, marker="o",
    #           label=f"Arithmetic-mean-ensemble - [2, 2, 2, 2, 2, 2]")

    # MLP - hidden_size = [2]
    # Expect only 1 data point for MLP originally
    mse_mlp_2 = np.repeat(np.array(df_mlp[df_mlp["hidden_size"].apply(
        lambda x: x == [2])]["loss_test"]), len(data_x))
    assert len(mse_mlp_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp_2 has length {len(mse_mlp_2)}"
    axis.plot(data_x, mse_mlp_2, marker="o", label=f"MLP - [2]")

    # MLP - hidden_size = [2, 2, 2]
    # Expect only 1 data point for MLP originally
    mse_mlp_2_2_2 = np.repeat(np.array(df_mlp[df_mlp["hidden_size"].apply(
        lambda x: x == [2, 2, 2])]["loss_test"]), len(data_x))
    assert len(mse_mlp_2_2_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp_2_2_2 has length {len(mse_mlp_2_2_2)}"
    axis.plot(data_x, mse_mlp_2_2_2, marker="o", label=f"MLP - [2, 2, 2]")

    # MLP - hidden_size = [2, 2, 2, 2, 2, 2]
    # Expect only 1 data point for MLP originally
    mse_mlp_2_2_2_2_2_2 = np.repeat(np.array(df_mlp[df_mlp["hidden_size"].apply(
        lambda x: x == [2, 2, 2, 2, 2, 2])]["loss_test"]), len(data_x))
    assert len(mse_mlp_2_2_2_2_2_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp_2_2_2_2_2_2 has length {len(mse_mlp_2_2_2_2_2_2)}"
    axis.plot(data_x, mse_mlp_2_2_2_2_2_2, marker="o",
              label=f"MLP - [2, 2, 2, 2, 2, 2]")

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


def mse_against_correlation_penalty_coefficient_0_1_fixed_ensemble_size_6_architecture_6_comparator_mlp(raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_static_ncl is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_static_ncl, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_static_ncl = {
        "hidden_size": [
            [2],
            [6],
            [12]
        ],
        "ensemble_size": [
            6
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
        ]
    }
    filter_traditional = {
        "hidden_size": [
            [2],
            [6],
            [12]
        ],
        "ensemble_size": [
            6
        ],
        "epoch_num": [
            400
        ]
    }
    filter_mlp = {
        "hidden_size": [
            [2],
            [6],
            [12]
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(6 base learners per ensemble model)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"static_ncl_mse_against_correlation_penalty_coefficient_0_1_fixed_ensemble_size_6_architecture_6_comparator_mlp.png"
    # ====================================================================

    df_static_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_static_ncl,
        filter=filter_static_ncl
    )
    df_traditional = data_helper.apply_dataframe_filter(
        df=raw_df_traditional,
        filter=filter_traditional
    )
    df_mlp = data_helper.apply_dataframe_filter(
        df=raw_df_mlp,
        filter=filter_mlp
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_static_ncl["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Static NCL - hidden_size = [2]
    mse_static_ncl_2 = np.array(
        df_static_ncl[df_static_ncl["hidden_size"].apply(lambda x: x == [2])]["loss_test"])
    assert len(mse_static_ncl_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_2 has length {len(mse_static_ncl_2)}"
    axis.plot(data_x, mse_static_ncl_2, marker="o", label=f"NCL - [2]")

    # Static NCL - hidden_size = [2, 2, 2]
    mse_static_ncl_6 = np.array(
        df_static_ncl[df_static_ncl["hidden_size"].apply(lambda x: x == [6])]["loss_test"])
    assert len(mse_static_ncl_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_6 has length {len(mse_static_ncl_6)}"
    axis.plot(data_x, mse_static_ncl_6, marker="o", label=f"NCL - [6]")

    # Static NCL - hidden_size = [12]
    mse_static_ncl_12 = np.array(
        df_static_ncl[df_static_ncl["hidden_size"].apply(lambda x: x == [12])]["loss_test"])
    assert len(mse_static_ncl_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_12 has length {len(mse_static_ncl_12)}"
    axis.plot(data_x, mse_static_ncl_12, marker="o", label=f"NCL - [12]")

    # # Arithmetic-mean-ensemble - hidden_size = [2]
    # mse_traditional_2 = np.repeat(np.array(df_traditional[df_traditional["hidden_size"].apply(
    #     lambda x: x == [2])]["loss_test"]), len(data_x))
    # assert len(mse_traditional_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_2 has length {len(mse_traditional_2)}"
    # axis.plot(data_x, mse_traditional_2, marker="o",
    #           label=f"Arithmetic-mean-ensemble - [2]")

    # # Arithmetic-mean-ensemble - hidden_size = [6]
    # mse_traditional_6 = np.repeat(np.array(df_traditional[df_traditional["hidden_size"].apply(
    #     lambda x: x == [6])]["loss_test"]), len(data_x))
    # assert len(mse_traditional_6) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_6 has length {len(mse_traditional_6)}"
    # axis.plot(data_x, mse_traditional_6, marker="o",
    #           label=f"Arithmetic-mean-ensemble - [6]")

    # # Arithmetic-mean-ensemble - hidden_size = [12]
    # mse_traditional_12 = np.repeat(np.array(df_traditional[df_traditional["hidden_size"].apply(
    #     lambda x: x == [12])]["loss_test"]), len(data_x))
    # assert len(mse_traditional_12) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_12 has length {len(mse_traditional_12)}"
    # axis.plot(data_x, mse_traditional_12, marker="o",
    #           label=f"Arithmetic-mean-ensemble - [12]")

    # MLP - hidden_size = [2]
    # Expect only 1 data point for MLP originally
    mse_mlp_2 = np.repeat(np.array(df_mlp[df_mlp["hidden_size"].apply(
        lambda x: x == [2])]["loss_test"]), len(data_x))
    assert len(mse_mlp_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp_2 has length {len(mse_mlp_2)}"
    axis.plot(data_x, mse_mlp_2, marker="o", label=f"MLP - [2]")

    # MLP - hidden_size = [6]
    # Expect only 1 data point for MLP originally
    mse_mlp_6 = np.repeat(np.array(df_mlp[df_mlp["hidden_size"].apply(
        lambda x: x == [6])]["loss_test"]), len(data_x))
    assert len(mse_mlp_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp_6 has length {len(mse_mlp_6)}"
    axis.plot(data_x, mse_mlp_6, marker="o", label=f"MLP - [6]")

    # MLP - hidden_size = [12]
    # Expect only 1 data point for MLP originally
    mse_mlp_12 = np.repeat(np.array(df_mlp[df_mlp["hidden_size"].apply(
        lambda x: x == [12])]["loss_test"]), len(data_x))
    assert len(mse_mlp_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp_12 has length {len(mse_mlp_12)}"
    axis.plot(data_x, mse_mlp_12, marker="o", label=f"MLP - [12]")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend(loc=f"upper center", bbox_to_anchor=(
        0.5, -0.25), ncol=2, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_correlation_penalty_coefficient_08_15_fixed_ensemble_size_6_architecture_6_comparator_mlp(raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_static_ncl is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_static_ncl, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_static_ncl = {
        "hidden_size": [
            [2],
            [6],
            [12]
        ],
        "ensemble_size": [
            6
        ],
        "correlation_penalty_coefficient": [
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5
        ]
    }
    filter_traditional = {
        "hidden_size": [
            [2],
            [6],
            [12]
        ],
        "ensemble_size": [
            6
        ],
        "epoch_num": [
            400
        ]
    }
    filter_mlp = {
        "hidden_size": [
            [2],
            [6],
            [12]
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\lambda$\n(6 base learners per ensemble model)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"static_ncl_mse_against_correlation_penalty_coefficient_08_15_fixed_ensemble_size_6_architecture_6_comparator_mlp.png"
    # ====================================================================

    df_static_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_static_ncl,
        filter=filter_static_ncl
    )
    df_traditional = data_helper.apply_dataframe_filter(
        df=raw_df_traditional,
        filter=filter_traditional
    )
    df_mlp = data_helper.apply_dataframe_filter(
        df=raw_df_mlp,
        filter=filter_mlp
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_static_ncl["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Static NCL - hidden_size = [2]
    mse_static_ncl_2 = np.array(
        df_static_ncl[df_static_ncl["hidden_size"].apply(lambda x: x == [2])]["loss_test"])
    assert len(mse_static_ncl_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_2 has length {len(mse_static_ncl_2)}"
    axis.plot(data_x, mse_static_ncl_2, marker="o", label=f"NCL - [2]")

    # Static NCL - hidden_size = [2, 2, 2]
    mse_static_ncl_6 = np.array(
        df_static_ncl[df_static_ncl["hidden_size"].apply(lambda x: x == [6])]["loss_test"])
    assert len(mse_static_ncl_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_6 has length {len(mse_static_ncl_6)}"
    axis.plot(data_x, mse_static_ncl_6, marker="o", label=f"NCL - [6]")

    # Static NCL - hidden_size = [12]
    mse_static_ncl_12 = np.array(
        df_static_ncl[df_static_ncl["hidden_size"].apply(lambda x: x == [12])]["loss_test"])
    assert len(mse_static_ncl_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_12 has length {len(mse_static_ncl_12)}"
    axis.plot(data_x, mse_static_ncl_12, marker="o", label=f"NCL - [12]")

    # # Arithmetic-mean-ensemble - hidden_size = [2]
    # mse_traditional_2 = np.repeat(np.array(df_traditional[df_traditional["hidden_size"].apply(
    #     lambda x: x == [2])]["loss_test"]), len(data_x))
    # assert len(mse_traditional_2) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_2 has length {len(mse_traditional_2)}"
    # axis.plot(data_x, mse_traditional_2, marker="o",
    #           label=f"Arithmetic-mean-ensemble - [2]")

    # # Arithmetic-mean-ensemble - hidden_size = [6]
    # mse_traditional_6 = np.repeat(np.array(df_traditional[df_traditional["hidden_size"].apply(
    #     lambda x: x == [6])]["loss_test"]), len(data_x))
    # assert len(mse_traditional_6) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_6 has length {len(mse_traditional_6)}"
    # axis.plot(data_x, mse_traditional_6, marker="o",
    #           label=f"Arithmetic-mean-ensemble - [6]")

    # # Arithmetic-mean-ensemble - hidden_size = [12]
    # mse_traditional_12 = np.repeat(np.array(df_traditional[df_traditional["hidden_size"].apply(
    #     lambda x: x == [12])]["loss_test"]), len(data_x))
    # assert len(mse_traditional_12) == len(
    #     data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_12 has length {len(mse_traditional_12)}"
    # axis.plot(data_x, mse_traditional_12, marker="o",
    #           label=f"Arithmetic-mean-ensemble - [12]")

    # MLP - hidden_size = [2]
    # Expect only 1 data point for MLP originally
    mse_mlp_2 = np.repeat(np.array(df_mlp[df_mlp["hidden_size"].apply(
        lambda x: x == [2])]["loss_test"]), len(data_x))
    assert len(mse_mlp_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp_2 has length {len(mse_mlp_2)}"
    axis.plot(data_x, mse_mlp_2, marker="o", label=f"MLP - [2]")

    # MLP - hidden_size = [6]
    # Expect only 1 data point for MLP originally
    mse_mlp_6 = np.repeat(np.array(df_mlp[df_mlp["hidden_size"].apply(
        lambda x: x == [6])]["loss_test"]), len(data_x))
    assert len(mse_mlp_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp_6 has length {len(mse_mlp_6)}"
    axis.plot(data_x, mse_mlp_6, marker="o", label=f"MLP - [6]")

    # MLP - hidden_size = [12]
    # Expect only 1 data point for MLP originally
    mse_mlp_12 = np.repeat(np.array(df_mlp[df_mlp["hidden_size"].apply(
        lambda x: x == [12])]["loss_test"]), len(data_x))
    assert len(mse_mlp_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_mlp_12 has length {len(mse_mlp_12)}"
    axis.plot(data_x, mse_mlp_12, marker="o", label=f"MLP - [12]")

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


def diversity_coefficient_against_correlation_penalty_coefficient_0_1_fixed_architecture_2(raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_static_ncl is not None and raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_static_ncl, raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_static_ncl = {
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
        ]
    }
    filter_traditional = {
        "hidden_size": [
            [2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"$\\rho$"
    FIGURE_TITLE = f"$\\rho$ vs $\\lambda$\n(each base learner is [2])"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"static_ncl_diversity_coefficient_against_correlation_penalty_coefficient_0_1_fixed_architecture_2.png"
    # ====================================================================

    df_static_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_static_ncl,
        filter=filter_static_ncl
    )
    df_traditional = data_helper.apply_dataframe_filter(
        df=raw_df_traditional,
        filter=filter_traditional
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_static_ncl["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Static NCL - ensemble_size = 2
    diversity_coefficient_static_ncl_2 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 2]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_2 has length {len(diversity_coefficient_static_ncl_2)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_2,
              marker="o", label=f"NCL - 2")

    # Static NCL - ensemble_size = 6
    diversity_coefficient_static_ncl_6 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 6]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_6 has length {len(diversity_coefficient_static_ncl_6)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_6,
              marker="o", label=f"NCL - 6")

    # Static NCL - ensemble_size = 12
    diversity_coefficient_static_ncl_12 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 12]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_12 has length {len(diversity_coefficient_static_ncl_12)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_12,
              marker="o", label=f"NCL - 12")

    # Arithmetic-mean-ensemble - ensemble_size = 2
    # Expect only 1 data point for traditional originally
    diversity_coefficient_traditional_2 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 2]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_2 has length {len(diversity_coefficient_traditional_2)}"
    axis.plot(data_x, diversity_coefficient_traditional_2,
              marker="o", label=f"Arithmetic-mean-ensemble - 2")

    # Arithmetic-mean-ensemble - ensemble_size = 6
    # Expect only 1 data point for traditional originally
    diversity_coefficient_traditional_6 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 6]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_6 has length {len(diversity_coefficient_traditional_6)}"
    axis.plot(data_x, diversity_coefficient_traditional_6,
              marker="o", label=f"Arithmetic-mean-ensemble - 6")

    # Arithmetic-mean-ensemble - ensemble_size = 12
    # Expect only 1 data point for traditional originally
    diversity_coefficient_traditional_12 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 12]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_12 has length {len(diversity_coefficient_traditional_12)}"
    axis.plot(data_x, diversity_coefficient_traditional_12,
              marker="o", label=f"Arithmetic-mean-ensemble - 12")

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


def diversity_coefficient_against_correlation_penalty_coefficient_08_15_fixed_architecture_2(raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_static_ncl is not None and raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_static_ncl, raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_static_ncl = {
        "hidden_size": [
            [2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5
        ]
    }
    filter_traditional = {
        "hidden_size": [
            [2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"$\\rho$"
    FIGURE_TITLE = f"$\\rho$ vs $\\lambda$\n(each base learner is [2])"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"static_ncl_diversity_coefficient_against_correlation_penalty_coefficient_08_15_fixed_architecture_2.png"
    # ====================================================================

    df_static_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_static_ncl,
        filter=filter_static_ncl
    )
    df_traditional = data_helper.apply_dataframe_filter(
        df=raw_df_traditional,
        filter=filter_traditional
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_static_ncl["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Static NCL - ensemble_size = 2
    diversity_coefficient_static_ncl_2 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 2]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_2 has length {len(diversity_coefficient_static_ncl_2)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_2,
              marker="o", label=f"NCL - 2")

    # Static NCL - ensemble_size = 6
    diversity_coefficient_static_ncl_6 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 6]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_6 has length {len(diversity_coefficient_static_ncl_6)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_6,
              marker="o", label=f"NCL - 6")

    # Static NCL - ensemble_size = 12
    diversity_coefficient_static_ncl_12 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 12]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_12 has length {len(diversity_coefficient_static_ncl_12)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_12,
              marker="o", label=f"NCL - 12")

    # Arithmetic-mean-ensemble - ensemble_size = 2
    # Expect only 1 data point for traditional originally
    diversity_coefficient_traditional_2 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 2]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_2 has length {len(diversity_coefficient_traditional_2)}"
    axis.plot(data_x, diversity_coefficient_traditional_2,
              marker="o", label=f"Arithmetic-mean-ensemble - 2")

    # Arithmetic-mean-ensemble - ensemble_size = 6
    # Expect only 1 data point for traditional originally
    diversity_coefficient_traditional_6 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 6]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_6 has length {len(diversity_coefficient_traditional_6)}"
    axis.plot(data_x, diversity_coefficient_traditional_6,
              marker="o", label=f"Arithmetic-mean-ensemble - 6")

    # Arithmetic-mean-ensemble - ensemble_size = 12
    # Expect only 1 data point for traditional originally
    diversity_coefficient_traditional_12 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 12]["loss_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_12 has length {len(diversity_coefficient_traditional_12)}"
    axis.plot(data_x, diversity_coefficient_traditional_12,
              marker="o", label=f"Arithmetic-mean-ensemble - 12")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend(loc=f"upper center", bbox_to_anchor=(
        0.5, -0.25), ncol=2, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def diversity_coefficient_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_2_2(raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_static_ncl is not None and raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_static_ncl, raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_static_ncl = {
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
        ]
    }
    filter_traditional = {
        "hidden_size": [
            [2, 2, 2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"$\\rho$"
    FIGURE_TITLE = f"$\\rho$ vs $\\lambda$\n(each base learner is [2, 2, 2])"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"static_ncl_diversity_coefficient_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_2_2.png"
    # ====================================================================

    df_static_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_static_ncl,
        filter=filter_static_ncl
    )
    df_traditional = data_helper.apply_dataframe_filter(
        df=raw_df_traditional,
        filter=filter_traditional
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_static_ncl["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Static NCL - ensemble_size = 2
    diversity_coefficient_static_ncl_2 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 2]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_2 has length {len(diversity_coefficient_static_ncl_2)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_2,
              marker="o", label=f"NCL - 2")

    # Static NCL - ensemble_size = 6
    diversity_coefficient_static_ncl_6 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 6]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_6 has length {len(diversity_coefficient_static_ncl_6)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_6,
              marker="o", label=f"NCL - 6")

    # Static NCL - ensemble_size = 12
    diversity_coefficient_static_ncl_12 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 12]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_12 has length {len(diversity_coefficient_static_ncl_12)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_12,
              marker="o", label=f"NCL - 12")

    # Arithmetic-mean-ensemble - ensemble_size = 2
    # Expect only 1 data point for traditional originally
    diversity_coefficient_traditional_2 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 2]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_2 has length {len(diversity_coefficient_traditional_2)}"
    axis.plot(data_x, diversity_coefficient_traditional_2,
              marker="o", label=f"Arithmetic-mean-ensemble - 2")

    # Arithmetic-mean-ensemble - ensemble_size = 6
    # Expect only 1 data point for traditional originally
    diversity_coefficient_traditional_6 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 6]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_6 has length {len(diversity_coefficient_traditional_6)}"
    axis.plot(data_x, diversity_coefficient_traditional_6,
              marker="o", label=f"Arithmetic-mean-ensemble - 6")

    # Arithmetic-mean-ensemble - ensemble_size = 12
    # Expect only 1 data point for traditional originally
    diversity_coefficient_traditional_12 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 12]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_12 has length {len(diversity_coefficient_traditional_12)}"
    axis.plot(data_x, diversity_coefficient_traditional_12,
              marker="o", label=f"Arithmetic-mean-ensemble - 12")

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


def diversity_coefficient_against_correlation_penalty_coefficient_08_15_fixed_architecture_2_2_2(raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_static_ncl is not None and raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_static_ncl, raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_static_ncl = {
        "hidden_size": [
            [2, 2, 2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5
        ]
    }
    filter_traditional = {
        "hidden_size": [
            [2, 2, 2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"$\\rho$"
    FIGURE_TITLE = f"$\\rho$ vs $\\lambda$\n(each base learner is [2, 2, 2])"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"static_ncl_diversity_coefficient_against_correlation_penalty_coefficient_08_15_fixed_architecture_2_2_2.png"
    # ====================================================================

    df_static_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_static_ncl,
        filter=filter_static_ncl
    )
    df_traditional = data_helper.apply_dataframe_filter(
        df=raw_df_traditional,
        filter=filter_traditional
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_static_ncl["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Static NCL - ensemble_size = 2
    diversity_coefficient_static_ncl_2 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 2]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_2 has length {len(diversity_coefficient_static_ncl_2)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_2,
              marker="o", label=f"NCL - 2")

    # Static NCL - ensemble_size = 6
    diversity_coefficient_static_ncl_6 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 6]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_6 has length {len(diversity_coefficient_static_ncl_6)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_6,
              marker="o", label=f"NCL - 6")

    # Static NCL - ensemble_size = 12
    diversity_coefficient_static_ncl_12 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 12]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_12 has length {len(diversity_coefficient_static_ncl_12)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_12,
              marker="o", label=f"NCL - 12")

    # Arithmetic-mean-ensemble - ensemble_size = 2
    # Expect only 1 data point for traditional originally
    diversity_coefficient_traditional_2 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 2]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_2 has length {len(diversity_coefficient_traditional_2)}"
    axis.plot(data_x, diversity_coefficient_traditional_2,
              marker="o", label=f"Arithmetic-mean-ensemble - 2")

    # Arithmetic-mean-ensemble - ensemble_size = 6
    # Expect only 1 data point for traditional originally
    diversity_coefficient_traditional_6 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 6]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_6 has length {len(diversity_coefficient_traditional_6)}"
    axis.plot(data_x, diversity_coefficient_traditional_6,
              marker="o", label=f"Arithmetic-mean-ensemble - 6")

    # Arithmetic-mean-ensemble - ensemble_size = 12
    # Expect only 1 data point for traditional originally
    diversity_coefficient_traditional_12 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 12]["loss_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_12 has length {len(diversity_coefficient_traditional_12)}"
    axis.plot(data_x, diversity_coefficient_traditional_12,
              marker="o", label=f"Arithmetic-mean-ensemble - 12")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend(loc=f"upper center", bbox_to_anchor=(
        0.5, -0.25), ncol=2, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def diversity_coefficient_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_2_2_2_2_2(raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_static_ncl is not None and raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_static_ncl, raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_static_ncl = {
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
        ]
    }
    filter_traditional = {
        "hidden_size": [
            [2, 2, 2, 2, 2, 2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"$\\rho$"
    FIGURE_TITLE = f"$\\rho$ vs $\\lambda$\n(each base learner is [2, 2, 2, 2, 2, 2])"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"static_ncl_diversity_coefficient_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_2_2_2_2_2.png"
    # ====================================================================

    df_static_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_static_ncl,
        filter=filter_static_ncl
    )
    df_traditional = data_helper.apply_dataframe_filter(
        df=raw_df_traditional,
        filter=filter_traditional
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_static_ncl["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Static NCL - ensemble_size = 2
    diversity_coefficient_static_ncl_2 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 2]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_2 has length {len(diversity_coefficient_static_ncl_2)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_2,
              marker="o", label=f"NCL - 2")

    # Static NCL - ensemble_size = 6
    diversity_coefficient_static_ncl_6 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 6]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_6 has length {len(diversity_coefficient_static_ncl_6)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_6,
              marker="o", label=f"NCL - 6")

    # Static NCL - ensemble_size = 12
    diversity_coefficient_static_ncl_12 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 12]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_12 has length {len(diversity_coefficient_static_ncl_12)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_12,
              marker="o", label=f"NCL - 12")

    # Arithmetic-mean-ensemble - ensemble_size = 2
    # Expect only 1 data point for traditional originally
    diversity_coefficient_traditional_2 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 2]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_2 has length {len(diversity_coefficient_traditional_2)}"
    axis.plot(data_x, diversity_coefficient_traditional_2,
              marker="o", label=f"Arithmetic-mean-ensemble - 2")

    # Arithmetic-mean-ensemble - ensemble_size = 6
    # Expect only 1 data point for traditional originally
    diversity_coefficient_traditional_6 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 6]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_6 has length {len(diversity_coefficient_traditional_6)}"
    axis.plot(data_x, diversity_coefficient_traditional_6,
              marker="o", label=f"Arithmetic-mean-ensemble - 6")

    # Arithmetic-mean-ensemble - ensemble_size = 12
    # Expect only 1 data point for traditional originally
    diversity_coefficient_traditional_12 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 12]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_12 has length {len(diversity_coefficient_traditional_12)}"
    axis.plot(data_x, diversity_coefficient_traditional_12,
              marker="o", label=f"Arithmetic-mean-ensemble - 12")

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


def diversity_coefficient_against_correlation_penalty_coefficient_08_15_fixed_architecture_2_2_2_2_2_2(raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_static_ncl is not None and raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_static_ncl, raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_static_ncl = {
        "hidden_size": [
            [2, 2, 2, 2, 2, 2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5
        ]
    }
    filter_traditional = {
        "hidden_size": [
            [2, 2, 2, 2, 2, 2]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"$\\rho$"
    FIGURE_TITLE = f"$\\rho$ vs $\\lambda$\n(each base learner is [2, 2, 2, 2, 2, 2])"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"static_ncl_diversity_coefficient_against_correlation_penalty_coefficient_08_15_fixed_architecture_2_2_2_2_2_2.png"
    # ====================================================================

    df_static_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_static_ncl,
        filter=filter_static_ncl
    )
    df_traditional = data_helper.apply_dataframe_filter(
        df=raw_df_traditional,
        filter=filter_traditional
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_static_ncl["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Static NCL - ensemble_size = 2
    diversity_coefficient_static_ncl_2 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 2]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_2 has length {len(diversity_coefficient_static_ncl_2)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_2,
              marker="o", label=f"NCL - 2")

    # Static NCL - ensemble_size = 6
    diversity_coefficient_static_ncl_6 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 6]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_6 has length {len(diversity_coefficient_static_ncl_6)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_6,
              marker="o", label=f"NCL - 6")

    # Static NCL - ensemble_size = 12
    diversity_coefficient_static_ncl_12 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 12]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_12 has length {len(diversity_coefficient_static_ncl_12)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_12,
              marker="o", label=f"NCL - 12")

    # Arithmetic-mean-ensemble - ensemble_size = 2
    # Expect only 1 data point for traditional originally
    diversity_coefficient_traditional_2 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 2]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_2 has length {len(diversity_coefficient_traditional_2)}"
    axis.plot(data_x, diversity_coefficient_traditional_2,
              marker="o", label=f"Arithmetic-mean-ensemble - 2")

    # Arithmetic-mean-ensemble - ensemble_size = 6
    # Expect only 1 data point for traditional originally
    diversity_coefficient_traditional_6 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 6]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_6 has length {len(diversity_coefficient_traditional_6)}"
    axis.plot(data_x, diversity_coefficient_traditional_6,
              marker="o", label=f"Arithmetic-mean-ensemble - 6")

    # Arithmetic-mean-ensemble - ensemble_size = 12
    # Expect only 1 data point for traditional originally
    diversity_coefficient_traditional_12 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 12]["loss_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_12 has length {len(diversity_coefficient_traditional_12)}"
    axis.plot(data_x, diversity_coefficient_traditional_12,
              marker="o", label=f"Arithmetic-mean-ensemble - 12")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend(loc=f"upper center", bbox_to_anchor=(
        0.5, -0.25), ncol=2, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def diversity_coefficient_against_correlation_penalty_coefficient_0_1_fixed_architecture_6(raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_static_ncl is not None and raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_static_ncl, raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_static_ncl = {
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
        ]
    }
    filter_traditional = {
        "hidden_size": [
            [6]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"$\\rho$"
    FIGURE_TITLE = f"$\\rho$ vs $\\lambda$\n(each base learner is [6])"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"static_ncl_diversity_coefficient_against_correlation_penalty_coefficient_0_1_fixed_architecture_6.png"
    # ====================================================================

    df_static_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_static_ncl,
        filter=filter_static_ncl
    )
    df_traditional = data_helper.apply_dataframe_filter(
        df=raw_df_traditional,
        filter=filter_traditional
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_static_ncl["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Static NCL - ensemble_size = 2
    diversity_coefficient_static_ncl_2 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 2]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_2 has length {len(diversity_coefficient_static_ncl_2)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_2,
              marker="o", label=f"NCL - 2")

    # Static NCL - ensemble_size = 6
    diversity_coefficient_static_ncl_6 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 6]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_6 has length {len(diversity_coefficient_static_ncl_6)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_6,
              marker="o", label=f"NCL - 6")

    # Static NCL - ensemble_size = 12
    diversity_coefficient_static_ncl_12 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 12]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_12 has length {len(diversity_coefficient_static_ncl_12)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_12,
              marker="o", label=f"NCL - 12")

    # Arithmetic-mean-ensemble - ensemble_size = 2
    # Expect only 1 data point for traditional originally
    diversity_coefficient_traditional_2 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 2]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_2 has length {len(diversity_coefficient_traditional_2)}"
    axis.plot(data_x, diversity_coefficient_traditional_2,
              marker="o", label=f"Arithmetic-mean-ensemble - 2")

    # Arithmetic-mean-ensemble - ensemble_size = 6
    # Expect only 1 data point for traditional originally
    diversity_coefficient_traditional_6 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 6]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_6 has length {len(diversity_coefficient_traditional_6)}"
    axis.plot(data_x, diversity_coefficient_traditional_6,
              marker="o", label=f"Arithmetic-mean-ensemble - 6")

    # Arithmetic-mean-ensemble - ensemble_size = 12
    # Expect only 1 data point for traditional originally
    diversity_coefficient_traditional_12 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 12]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_12 has length {len(diversity_coefficient_traditional_12)}"
    axis.plot(data_x, diversity_coefficient_traditional_12,
              marker="o", label=f"Arithmetic-mean-ensemble - 12")

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


def diversity_coefficient_against_correlation_penalty_coefficient_08_15_fixed_architecture_6(raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_static_ncl is not None and raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_static_ncl, raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_static_ncl = {
        "hidden_size": [
            [6]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5
        ]
    }
    filter_traditional = {
        "hidden_size": [
            [6]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"$\\rho$"
    FIGURE_TITLE = f"$\\rho$ vs $\\lambda$\n(each base learner is [6])"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"static_ncl_diversity_coefficient_against_correlation_penalty_coefficient_08_15_fixed_architecture_6.png"
    # ====================================================================

    df_static_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_static_ncl,
        filter=filter_static_ncl
    )
    df_traditional = data_helper.apply_dataframe_filter(
        df=raw_df_traditional,
        filter=filter_traditional
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_static_ncl["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Static NCL - ensemble_size = 2
    diversity_coefficient_static_ncl_2 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 2]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_2 has length {len(diversity_coefficient_static_ncl_2)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_2,
              marker="o", label=f"NCL - 2")

    # Static NCL - ensemble_size = 6
    diversity_coefficient_static_ncl_6 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 6]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_6 has length {len(diversity_coefficient_static_ncl_6)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_6,
              marker="o", label=f"NCL - 6")

    # Static NCL - ensemble_size = 12
    diversity_coefficient_static_ncl_12 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 12]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_12 has length {len(diversity_coefficient_static_ncl_12)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_12,
              marker="o", label=f"NCL - 12")

    # Arithmetic-mean-ensemble - ensemble_size = 2
    # Expect only 1 data point for traditional originally
    diversity_coefficient_traditional_2 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 2]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_2 has length {len(diversity_coefficient_traditional_2)}"
    axis.plot(data_x, diversity_coefficient_traditional_2,
              marker="o", label=f"Arithmetic-mean-ensemble - 2")

    # Arithmetic-mean-ensemble - ensemble_size = 6
    # Expect only 1 data point for traditional originally
    diversity_coefficient_traditional_6 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 6]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_6 has length {len(diversity_coefficient_traditional_6)}"
    axis.plot(data_x, diversity_coefficient_traditional_6,
              marker="o", label=f"Arithmetic-mean-ensemble - 6")

    # Arithmetic-mean-ensemble - ensemble_size = 12
    # Expect only 1 data point for traditional originally
    diversity_coefficient_traditional_12 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 12]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_12 has length {len(diversity_coefficient_traditional_12)}"
    axis.plot(data_x, diversity_coefficient_traditional_12,
              marker="o", label=f"Arithmetic-mean-ensemble - 12")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend(loc=f"upper center", bbox_to_anchor=(
        0.5, -0.25), ncol=2, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def diversity_coefficient_against_correlation_penalty_coefficient_0_1_fixed_architecture_12(raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_static_ncl is not None and raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_static_ncl, raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_static_ncl = {
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
        ]
    }
    filter_traditional = {
        "hidden_size": [
            [12]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"$\\rho$"
    FIGURE_TITLE = f"$\\rho$ vs $\\lambda$\n(each base learner is [12])"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"static_ncl_diversity_coefficient_against_correlation_penalty_coefficient_0_1_fixed_architecture_12.png"
    # ====================================================================

    df_static_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_static_ncl,
        filter=filter_static_ncl
    )
    df_traditional = data_helper.apply_dataframe_filter(
        df=raw_df_traditional,
        filter=filter_traditional
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_static_ncl["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Static NCL - ensemble_size = 2
    diversity_coefficient_static_ncl_2 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 2]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_2 has length {len(diversity_coefficient_static_ncl_2)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_2,
              marker="o", label=f"NCL - 2")

    # Static NCL - ensemble_size = 6
    diversity_coefficient_static_ncl_6 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 6]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_6 has length {len(diversity_coefficient_static_ncl_6)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_6,
              marker="o", label=f"NCL - 6")

    # Static NCL - ensemble_size = 12
    diversity_coefficient_static_ncl_12 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 12]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_12 has length {len(diversity_coefficient_static_ncl_12)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_12,
              marker="o", label=f"NCL - 12")

    # Arithmetic-mean-ensemble - ensemble_size = 2
    # Expect only 1 data point for traditional originally
    diversity_coefficient_traditional_2 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 2]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_2 has length {len(diversity_coefficient_traditional_2)}"
    axis.plot(data_x, diversity_coefficient_traditional_2,
              marker="o", label=f"Arithmetic-mean-ensemble - 2")

    # Arithmetic-mean-ensemble - ensemble_size = 6
    # Expect only 1 data point for traditional originally
    diversity_coefficient_traditional_6 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 6]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_6 has length {len(diversity_coefficient_traditional_6)}"
    axis.plot(data_x, diversity_coefficient_traditional_6,
              marker="o", label=f"Arithmetic-mean-ensemble - 6")

    # Arithmetic-mean-ensemble - ensemble_size = 12
    # Expect only 1 data point for traditional originally
    diversity_coefficient_traditional_12 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 12]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_12 has length {len(diversity_coefficient_traditional_12)}"
    axis.plot(data_x, diversity_coefficient_traditional_12,
              marker="o", label=f"Arithmetic-mean-ensemble - 12")

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


def diversity_coefficient_against_correlation_penalty_coefficient_08_15_fixed_architecture_12(raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_static_ncl is not None and raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_static_ncl, raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_static_ncl = {
        "hidden_size": [
            [12]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "correlation_penalty_coefficient": [
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5
        ]
    }
    filter_traditional = {
        "hidden_size": [
            [12]
        ],
        "ensemble_size": [
            2,
            6,
            12
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"$\\rho$"
    FIGURE_TITLE = f"$\\rho$ vs $\\lambda$\n(each base learner is [12])"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"static_ncl_diversity_coefficient_against_correlation_penalty_coefficient_08_15_fixed_architecture_12.png"
    # ====================================================================

    df_static_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_static_ncl,
        filter=filter_static_ncl
    )
    df_traditional = data_helper.apply_dataframe_filter(
        df=raw_df_traditional,
        filter=filter_traditional
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_static_ncl["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Static NCL - ensemble_size = 2
    diversity_coefficient_static_ncl_2 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 2]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_2 has length {len(diversity_coefficient_static_ncl_2)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_2,
              marker="o", label=f"NCL - 2")

    # Static NCL - ensemble_size = 6
    diversity_coefficient_static_ncl_6 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 6]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_6 has length {len(diversity_coefficient_static_ncl_6)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_6,
              marker="o", label=f"NCL - 6")

    # Static NCL - ensemble_size = 12
    diversity_coefficient_static_ncl_12 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 12]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_12 has length {len(diversity_coefficient_static_ncl_12)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_12,
              marker="o", label=f"NCL - 12")

    # Arithmetic-mean-ensemble - ensemble_size = 2
    # Expect only 1 data point for traditional originally
    diversity_coefficient_traditional_2 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 2]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_2 has length {len(diversity_coefficient_traditional_2)}"
    axis.plot(data_x, diversity_coefficient_traditional_2,
              marker="o", label=f"Arithmetic-mean-ensemble - 2")

    # Arithmetic-mean-ensemble - ensemble_size = 6
    # Expect only 1 data point for traditional originally
    diversity_coefficient_traditional_6 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 6]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_6 has length {len(diversity_coefficient_traditional_6)}"
    axis.plot(data_x, diversity_coefficient_traditional_6,
              marker="o", label=f"Arithmetic-mean-ensemble - 6")

    # Arithmetic-mean-ensemble - ensemble_size = 12
    # Expect only 1 data point for traditional originally
    diversity_coefficient_traditional_12 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 12]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_12 has length {len(diversity_coefficient_traditional_12)}"
    axis.plot(data_x, diversity_coefficient_traditional_12,
              marker="o", label=f"Arithmetic-mean-ensemble - 12")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend(loc=f"upper center", bbox_to_anchor=(
        0.5, -0.25), ncol=2, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def diversity_coefficient_against_correlation_penalty_coefficient_0_1_fixed_ensemble_size_6_architecture_2_2_2(raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_static_ncl is not None and raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_static_ncl, raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_static_ncl = {
        "hidden_size": [
            [2],
            [2, 2, 2],
            [2, 2, 2, 2, 2, 2]
        ],
        "ensemble_size": [
            6
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
        ]
    }
    filter_traditional = {
        "hidden_size": [
            [2],
            [2, 2, 2],
            [2, 2, 2, 2, 2, 2]
        ],
        "ensemble_size": [
            6
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"$\\rho$"
    FIGURE_TITLE = f"$\\rho$ vs $\\lambda$\n(6 base learners per ensemble model)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"static_ncl_diversity_coefficient_against_correlation_penalty_coefficient_0_1_fixed_ensemble_size_6_architecture_2_2_2.png"
    # ====================================================================

    df_static_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_static_ncl,
        filter=filter_static_ncl
    )
    df_traditional = data_helper.apply_dataframe_filter(
        df=raw_df_traditional,
        filter=filter_traditional
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_static_ncl["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Static NCL - hidden_size = [2]
    diversity_coefficient_static_ncl_2 = np.array(
        df_static_ncl[df_static_ncl["hidden_size"].apply(lambda x: x == [2])]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_2 has length {len(diversity_coefficient_static_ncl_2)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_2,
              marker="o", label=f"NCL - [2]")

    # Static NCL - hidden_size = [2, 2, 2]
    diversity_coefficient_static_ncl_2_2_2 = np.array(df_static_ncl[df_static_ncl["hidden_size"].apply(
        lambda x: x == [2, 2, 2])]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_2_2_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_2_2_2 has length {len(diversity_coefficient_static_ncl_2_2_2)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_2_2_2,
              marker="o", label=f"NCL - [2, 2, 2]")

    # Static NCL - hidden_size = [2, 2, 2, 2, 2, 2]
    diversity_coefficient_static_ncl_2_2_2_2_2_2 = np.array(df_static_ncl[df_static_ncl["hidden_size"].apply(
        lambda x: x == [2, 2, 2, 2, 2, 2])]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_2_2_2_2_2_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_2_2_2_2_2_2 has length {len(diversity_coefficient_static_ncl_2_2_2_2_2_2)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_2_2_2_2_2_2,
              marker="o", label=f"NCL - [2, 2, 2, 2, 2, 2]")

    # Arithmetic-mean-ensemble - hidden_size = [2]
    diversity_coefficient_traditional_2 = np.repeat(np.array(df_traditional[df_traditional["hidden_size"].apply(
        lambda x: x == [2])]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_2 has length {len(diversity_coefficient_traditional_2)}"
    axis.plot(data_x, diversity_coefficient_traditional_2,
              marker="o", label=f"Arithmetic-mean-ensemble - [2]")

    # Arithmetic-mean-ensemble - hidden_size = [2, 2, 2]
    diversity_coefficient_traditional_2_2_2 = np.repeat(np.array(df_traditional[df_traditional["hidden_size"].apply(
        lambda x: x == [2, 2, 2])]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_2_2_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_2_2_2 has length {len(diversity_coefficient_traditional_2_2_2)}"
    axis.plot(data_x, diversity_coefficient_traditional_2_2_2,
              marker="o", label=f"Arithmetic-mean-ensemble - [2, 2, 2]")

    # Arithmetic-mean-ensemble - hidden_size = [2, 2, 2, 2, 2, 2]
    diversity_coefficient_traditional_2_2_2_2_2_2 = np.repeat(np.array(df_traditional[df_traditional["hidden_size"].apply(
        lambda x: x == [2, 2, 2, 2, 2, 2])]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_2_2_2_2_2_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_2_2_2_2_2_2 has length {len(diversity_coefficient_traditional_2_2_2_2_2_2)}"
    axis.plot(data_x, diversity_coefficient_traditional_2_2_2_2_2_2,
              marker="o", label=f"Arithmetic-mean-ensemble - [2, 2, 2, 2, 2, 2]")

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


def diversity_coefficient_against_correlation_penalty_coefficient_08_15_fixed_ensemble_size_6_architecture_2_2_2(raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_static_ncl is not None and raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_static_ncl, raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_static_ncl = {
        "hidden_size": [
            [2],
            [2, 2, 2],
            [2, 2, 2, 2, 2, 2]
        ],
        "ensemble_size": [
            6
        ],
        "correlation_penalty_coefficient": [
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5
        ]
    }
    filter_traditional = {
        "hidden_size": [
            [2],
            [2, 2, 2],
            [2, 2, 2, 2, 2, 2]
        ],
        "ensemble_size": [
            6
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"$\\rho$"
    FIGURE_TITLE = f"$\\rho$ vs $\\lambda$\n(6 base learners per ensemble model)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"static_ncl_diversity_coefficient_against_correlation_penalty_coefficient_08_15_fixed_ensemble_size_6_architecture_2_2_2.png"
    # ====================================================================

    df_static_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_static_ncl,
        filter=filter_static_ncl
    )
    df_traditional = data_helper.apply_dataframe_filter(
        df=raw_df_traditional,
        filter=filter_traditional
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_static_ncl["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Static NCL - hidden_size = [2]
    diversity_coefficient_static_ncl_2 = np.array(
        df_static_ncl[df_static_ncl["hidden_size"].apply(lambda x: x == [2])]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_2 has length {len(diversity_coefficient_static_ncl_2)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_2,
              marker="o", label=f"NCL - [2]")

    # Static NCL - hidden_size = [2, 2, 2]
    diversity_coefficient_static_ncl_2_2_2 = np.array(df_static_ncl[df_static_ncl["hidden_size"].apply(
        lambda x: x == [2, 2, 2])]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_2_2_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_2_2_2 has length {len(diversity_coefficient_static_ncl_2_2_2)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_2_2_2,
              marker="o", label=f"NCL - [2, 2, 2]")

    # Static NCL - hidden_size = [2, 2, 2, 2, 2, 2]
    diversity_coefficient_static_ncl_2_2_2_2_2_2 = np.array(df_static_ncl[df_static_ncl["hidden_size"].apply(
        lambda x: x == [2, 2, 2, 2, 2, 2])]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_2_2_2_2_2_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_2_2_2_2_2_2 has length {len(diversity_coefficient_static_ncl_2_2_2_2_2_2)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_2_2_2_2_2_2,
              marker="o", label=f"NCL - [2, 2, 2, 2, 2, 2]")

    # Arithmetic-mean-ensemble - hidden_size = [2]
    diversity_coefficient_traditional_2 = np.repeat(np.array(df_traditional[df_traditional["hidden_size"].apply(
        lambda x: x == [2])]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_2 has length {len(diversity_coefficient_traditional_2)}"
    axis.plot(data_x, diversity_coefficient_traditional_2,
              marker="o", label=f"Arithmetic-mean-ensemble - [2]")

    # Arithmetic-mean-ensemble - hidden_size = [2, 2, 2]
    diversity_coefficient_traditional_2_2_2 = np.repeat(np.array(df_traditional[df_traditional["hidden_size"].apply(
        lambda x: x == [2, 2, 2])]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_2_2_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_2_2_2 has length {len(diversity_coefficient_traditional_2_2_2)}"
    axis.plot(data_x, diversity_coefficient_traditional_2_2_2,
              marker="o", label=f"Arithmetic-mean-ensemble - [2, 2, 2]")

    # Arithmetic-mean-ensemble - hidden_size = [2, 2, 2, 2, 2, 2]
    diversity_coefficient_traditional_2_2_2_2_2_2 = np.repeat(np.array(df_traditional[df_traditional["hidden_size"].apply(
        lambda x: x == [2, 2, 2, 2, 2, 2])]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_2_2_2_2_2_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_2_2_2_2_2_2 has length {len(diversity_coefficient_traditional_2_2_2_2_2_2)}"
    axis.plot(data_x, diversity_coefficient_traditional_2_2_2_2_2_2,
              marker="o", label=f"Arithmetic-mean-ensemble - [2, 2, 2, 2, 2, 2]")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend(loc=f"upper center", bbox_to_anchor=(
        0.5, -0.25), ncol=2, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def diversity_coefficient_against_correlation_penalty_coefficient_0_1_fixed_ensemble_size_6_architecture_6(raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_static_ncl is not None and raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_static_ncl, raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_static_ncl = {
        "hidden_size": [
            [2],
            [6],
            [12]
        ],
        "ensemble_size": [
            6
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
        ]
    }
    filter_traditional = {
        "hidden_size": [
            [2],
            [6],
            [12]
        ],
        "ensemble_size": [
            6
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"$\\rho$"
    FIGURE_TITLE = f"$\\rho$ vs $\\lambda$\n(6 base learners per ensemble model)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"static_ncl_diversity_coefficient_against_correlation_penalty_coefficient_0_1_fixed_ensemble_size_6_architecture_6.png"
    # ====================================================================

    df_static_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_static_ncl,
        filter=filter_static_ncl
    )
    df_traditional = data_helper.apply_dataframe_filter(
        df=raw_df_traditional,
        filter=filter_traditional
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_static_ncl["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Static NCL - hidden_size = [2]
    diversity_coefficient_static_ncl_2 = np.array(
        df_static_ncl[df_static_ncl["hidden_size"].apply(lambda x: x == [2])]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_2 has length {len(diversity_coefficient_static_ncl_2)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_2,
              marker="o", label=f"NCL - [2]")

    # Static NCL - hidden_size = [2, 2, 2]
    diversity_coefficient_static_ncl_6 = np.array(
        df_static_ncl[df_static_ncl["hidden_size"].apply(lambda x: x == [6])]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_6 has length {len(diversity_coefficient_static_ncl_6)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_6,
              marker="o", label=f"NCL - [6]")

    # Static NCL - hidden_size = [12]
    diversity_coefficient_static_ncl_12 = np.array(
        df_static_ncl[df_static_ncl["hidden_size"].apply(lambda x: x == [12])]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_12 has length {len(diversity_coefficient_static_ncl_12)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_12,
              marker="o", label=f"NCL - [12]")

    # Arithmetic-mean-ensemble - hidden_size = [2]
    diversity_coefficient_traditional_2 = np.repeat(np.array(df_traditional[df_traditional["hidden_size"].apply(
        lambda x: x == [2])]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_2 has length {len(diversity_coefficient_traditional_2)}"
    axis.plot(data_x, diversity_coefficient_traditional_2,
              marker="o", label=f"Arithmetic-mean-ensemble - [2]")

    # Arithmetic-mean-ensemble - hidden_size = [6]
    diversity_coefficient_traditional_6 = np.repeat(np.array(df_traditional[df_traditional["hidden_size"].apply(
        lambda x: x == [6])]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_6 has length {len(diversity_coefficient_traditional_6)}"
    axis.plot(data_x, diversity_coefficient_traditional_6,
              marker="o", label=f"Arithmetic-mean-ensemble - [6]")

    # Arithmetic-mean-ensemble - hidden_size = [12]
    diversity_coefficient_traditional_12 = np.repeat(np.array(df_traditional[df_traditional["hidden_size"].apply(
        lambda x: x == [12])]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_12 has length {len(diversity_coefficient_traditional_12)}"
    axis.plot(data_x, diversity_coefficient_traditional_12,
              marker="o", label=f"Arithmetic-mean-ensemble - [12]")

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


def diversity_coefficient_against_correlation_penalty_coefficient_08_15_fixed_ensemble_size_6_architecture_6(raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_static_ncl is not None and raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_static_ncl, raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_static_ncl = {
        "hidden_size": [
            [2],
            [6],
            [12]
        ],
        "ensemble_size": [
            6
        ],
        "correlation_penalty_coefficient": [
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5
        ]
    }
    filter_traditional = {
        "hidden_size": [
            [2],
            [6],
            [12]
        ],
        "ensemble_size": [
            6
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\lambda$"
    Y_AXIS = f"$\\rho$"
    FIGURE_TITLE = f"$\\rho$ vs $\\lambda$\n(6 base learners per ensemble model)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"static_ncl_diversity_coefficient_against_correlation_penalty_coefficient_08_15_fixed_ensemble_size_6_architecture_6.png"
    # ====================================================================

    df_static_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_static_ncl,
        filter=filter_static_ncl
    )
    df_traditional = data_helper.apply_dataframe_filter(
        df=raw_df_traditional,
        filter=filter_traditional
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(
        df_static_ncl["correlation_penalty_coefficient"].unique())

    # Plot lines
    # Static NCL - hidden_size = [2]
    diversity_coefficient_static_ncl_2 = np.array(
        df_static_ncl[df_static_ncl["hidden_size"].apply(lambda x: x == [2])]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_2 has length {len(diversity_coefficient_static_ncl_2)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_2,
              marker="o", label=f"NCL - [2]")

    # Static NCL - hidden_size = [2, 2, 2]
    diversity_coefficient_static_ncl_6 = np.array(
        df_static_ncl[df_static_ncl["hidden_size"].apply(lambda x: x == [6])]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_6 has length {len(diversity_coefficient_static_ncl_6)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_6,
              marker="o", label=f"NCL - [6]")

    # Static NCL - hidden_size = [12]
    diversity_coefficient_static_ncl_12 = np.array(
        df_static_ncl[df_static_ncl["hidden_size"].apply(lambda x: x == [12])]["diversity_coefficient_test"])
    assert len(diversity_coefficient_static_ncl_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_12 has length {len(diversity_coefficient_static_ncl_12)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_12,
              marker="o", label=f"NCL - [12]")

    # Arithmetic-mean-ensemble - hidden_size = [2]
    diversity_coefficient_traditional_2 = np.repeat(np.array(df_traditional[df_traditional["hidden_size"].apply(
        lambda x: x == [2])]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_2 has length {len(diversity_coefficient_traditional_2)}"
    axis.plot(data_x, diversity_coefficient_traditional_2,
              marker="o", label=f"Arithmetic-mean-ensemble - [2]")

    # Arithmetic-mean-ensemble - hidden_size = [6]
    diversity_coefficient_traditional_6 = np.repeat(np.array(df_traditional[df_traditional["hidden_size"].apply(
        lambda x: x == [6])]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_6 has length {len(diversity_coefficient_traditional_6)}"
    axis.plot(data_x, diversity_coefficient_traditional_6,
              marker="o", label=f"Arithmetic-mean-ensemble - [6]")

    # Arithmetic-mean-ensemble - hidden_size = [12]
    diversity_coefficient_traditional_12 = np.repeat(np.array(df_traditional[df_traditional["hidden_size"].apply(
        lambda x: x == [12])]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_12 has length {len(diversity_coefficient_traditional_12)}"
    axis.plot(data_x, diversity_coefficient_traditional_12,
              marker="o", label=f"Arithmetic-mean-ensemble - [12]")

    # Set up other graph-related attributes
    axis.set_xlabel(X_AXIS)
    axis.set_ylabel(Y_AXIS)
    axis.set_title(FIGURE_TITLE)
    axis.grid(True)
    # Make line labels visible
    axis.legend(loc=f"upper center", bbox_to_anchor=(
        0.5, -0.25), ncol=2, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


if __name__ == "__main__":
    df_static_ncl = pd.read_csv(CSV_PATH_STATIC_NCL)
    # NOTE: If we read from csv, df_static_ncl["hidden_size"] will be of type class "str" instead of "list"
    df_static_ncl["hidden_size"] = df_static_ncl["hidden_size"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df_traditional = pd.read_csv(CSV_PATH_TRADITIONAL)
    # NOTE: If we read from csv, df_traditional["hidden_size"] will be of type class "str" instead of "list"
    df_traditional["hidden_size"] = df_traditional["hidden_size"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df_mlp = pd.read_csv(CSV_PATH_MLP)
    # NOTE: If we read from csv, df_mlp["hidden_size"] will be of type class "str" instead of "list"
    df_mlp["hidden_size"] = df_mlp["hidden_size"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2(
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_STATIC_NCL
    )

    mse_against_correlation_penalty_coefficient_08_15_fixed_architecture_2(
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_STATIC_NCL
    )

    mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_2_2(
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_STATIC_NCL
    )

    mse_against_correlation_penalty_coefficient_08_15_fixed_architecture_2_2_2(
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_STATIC_NCL
    )

    mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_2_2_2_2_2(
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_STATIC_NCL
    )

    mse_against_correlation_penalty_coefficient_08_15_fixed_architecture_2_2_2_2_2_2(
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_STATIC_NCL
    )

    mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_6(
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_STATIC_NCL
    )

    mse_against_correlation_penalty_coefficient_08_15_fixed_architecture_6(
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_STATIC_NCL
    )

    mse_against_correlation_penalty_coefficient_0_1_fixed_architecture_12(
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_STATIC_NCL
    )

    mse_against_correlation_penalty_coefficient_08_15_fixed_architecture_12(
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_STATIC_NCL
    )

    mse_against_correlation_penalty_coefficient_0_1_fixed_ensemble_size_6_architecture_2_2_2_comparator_traditional(
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_STATIC_NCL
    )

    mse_against_correlation_penalty_coefficient_08_15_fixed_ensemble_size_6_architecture_2_2_2_comparator_traditional(
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_STATIC_NCL
    )

    mse_against_correlation_penalty_coefficient_0_1_fixed_ensemble_size_6_architecture_6_comparator_traditional(
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_STATIC_NCL
    )

    mse_against_correlation_penalty_coefficient_08_15_fixed_ensemble_size_6_architecture_6_comparator_traditional(
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_STATIC_NCL
    )

    mse_against_correlation_penalty_coefficient_0_1_fixed_ensemble_size_6_architecture_2_2_2_comparator_mlp(
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_STATIC_NCL
    )

    mse_against_correlation_penalty_coefficient_08_15_fixed_ensemble_size_6_architecture_2_2_2_comparator_mlp(
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_STATIC_NCL
    )

    mse_against_correlation_penalty_coefficient_0_1_fixed_ensemble_size_6_architecture_6_comparator_mlp(
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_STATIC_NCL
    )

    mse_against_correlation_penalty_coefficient_08_15_fixed_ensemble_size_6_architecture_6_comparator_mlp(
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_STATIC_NCL
    )

    diversity_coefficient_against_correlation_penalty_coefficient_0_1_fixed_architecture_2(
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_STATIC_NCL
    )

    diversity_coefficient_against_correlation_penalty_coefficient_08_15_fixed_architecture_2(
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_STATIC_NCL
    )

    diversity_coefficient_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_2_2(
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_STATIC_NCL
    )

    diversity_coefficient_against_correlation_penalty_coefficient_08_15_fixed_architecture_2_2_2(
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_STATIC_NCL
    )

    diversity_coefficient_against_correlation_penalty_coefficient_0_1_fixed_architecture_2_2_2_2_2_2(
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_STATIC_NCL
    )

    diversity_coefficient_against_correlation_penalty_coefficient_08_15_fixed_architecture_2_2_2_2_2_2(
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_STATIC_NCL
    )

    diversity_coefficient_against_correlation_penalty_coefficient_0_1_fixed_architecture_6(
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_STATIC_NCL
    )

    diversity_coefficient_against_correlation_penalty_coefficient_08_15_fixed_architecture_6(
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_STATIC_NCL
    )

    diversity_coefficient_against_correlation_penalty_coefficient_0_1_fixed_architecture_12(
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_STATIC_NCL
    )

    diversity_coefficient_against_correlation_penalty_coefficient_08_15_fixed_architecture_12(
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_STATIC_NCL
    )

    diversity_coefficient_against_correlation_penalty_coefficient_0_1_fixed_ensemble_size_6_architecture_2_2_2(
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_STATIC_NCL
    )

    diversity_coefficient_against_correlation_penalty_coefficient_08_15_fixed_ensemble_size_6_architecture_2_2_2(
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_STATIC_NCL
    )

    diversity_coefficient_against_correlation_penalty_coefficient_0_1_fixed_ensemble_size_6_architecture_6(
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_STATIC_NCL
    )

    diversity_coefficient_against_correlation_penalty_coefficient_08_15_fixed_ensemble_size_6_architecture_6(
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_STATIC_NCL
    )

    logger.log(f"Graphs of static NCL experiment are saved ...")
