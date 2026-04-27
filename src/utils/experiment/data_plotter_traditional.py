import ast
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from src.utils import data_helper, logger

"""
Run: python -m src.utils.experiment.data_plotter_traditional
"""


# Constants
# Path to project root
ROOT_PATH = os.path.join(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Path to each csv file in csv/
CSV_PATH_TRADITIONAL = os.path.join(
    ROOT_PATH, f"data", f"experiment", f"traditional", f"csv", f"experiment_runner_traditional_arithmetic_mean.csv")
CSV_PATH_MLP = os.path.join(
    ROOT_PATH, f"data", f"experiment", f"mlp", f"csv", f"experiment_runner_mlp.csv")

# Path to img/
IMG_REPOSITORY_PATH_TRADITIONAL = os.path.join(
    ROOT_PATH, f"data", f"experiment", f"traditional", f"img")

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


def mse_against_ensemble_size_2_16_fixed_architecture_2_fixed_epoch_num_200(raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_traditional = {
        "hidden_size": [
            [2]
        ],
        "epoch_num": [
            200
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
        ]
    }
    filter_mlp = {
        "hidden_size": [
            [2]
        ],
        "epoch_num": [
            200
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"ensemble_size"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs ensemble_size\n(each base learner is [2] trained with 200 epochs)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"traditional_mse_against_ensemble_size_2_16_fixed_architecture_2_fixed_epoch_num_200.png"
    # ====================================================================

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
    data_x = np.array(df_traditional["ensemble_size"].unique())

    # Plot lines
    # Traditional ensemble loss
    mse_traditional_ensemble = np.array(df_traditional["loss_test"])
    assert len(mse_traditional_ensemble) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_ensemble has length {len(mse_traditional_ensemble)}"
    axis.plot(data_x, mse_traditional_ensemble, marker="o",
              label=f"Test - arithmetic-mean-ensemble")

    # Traditional average base learner loss
    mse_traditional_base_learner = np.array(
        df_traditional["loss_test_base_learner"])
    assert len(mse_traditional_base_learner) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_base_learner has length {len(mse_traditional_base_learner)}"
    axis.plot(data_x, mse_traditional_base_learner, marker="o",
              label=f"Test - base learner (average)")

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
        0.5, -0.25), ncol=1, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_ensemble_size_2_16_fixed_architecture_2_2_2_fixed_epoch_num_200(raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_traditional = {
        "hidden_size": [
            [2, 2, 2]
        ],
        "epoch_num": [
            200
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
        ]
    }
    filter_mlp = {
        "hidden_size": [
            [2, 2, 2]
        ],
        "epoch_num": [
            200
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"ensemble_size"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs ensemble_size\n(each base learner is [2, 2, 2] trained with 200 epochs)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"traditional_mse_against_ensemble_size_2_16_fixed_architecture_2_2_2_fixed_epoch_num_200.png"
    # ====================================================================

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
    data_x = np.array(df_traditional["ensemble_size"].unique())

    # Plot lines
    # Traditional ensemble loss
    mse_traditional_ensemble = np.array(df_traditional["loss_test"])
    assert len(mse_traditional_ensemble) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_ensemble has length {len(mse_traditional_ensemble)}"
    axis.plot(data_x, mse_traditional_ensemble, marker="o",
              label=f"Test - arithmetic-mean-ensemble")

    # Traditional average base learner loss
    mse_traditional_base_learner = np.array(
        df_traditional["loss_test_base_learner"])
    assert len(mse_traditional_base_learner) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_base_learner has length {len(mse_traditional_base_learner)}"
    axis.plot(data_x, mse_traditional_base_learner, marker="o",
              label=f"Test - base learner (average)")

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
        0.5, -0.25), ncol=1, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_ensemble_size_2_16_fixed_architecture_2_2_2_2_2_2_fixed_epoch_num_200(raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_traditional = {
        "hidden_size": [
            [2, 2, 2, 2, 2, 2]
        ],
        "epoch_num": [
            200
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
        ]
    }
    filter_mlp = {
        "hidden_size": [
            [2, 2, 2, 2, 2, 2]
        ],
        "epoch_num": [
            200
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"ensemble_size"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs ensemble_size\n(each base learner is [2, 2, 2, 2, 2, 2] trained with 200 epochs)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"traditional_mse_against_ensemble_size_2_16_fixed_architecture_2_2_2_2_2_2_fixed_epoch_num_200.png"
    # ====================================================================

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
    data_x = np.array(df_traditional["ensemble_size"].unique())

    # Plot lines
    # Traditional ensemble loss
    mse_traditional_ensemble = np.array(df_traditional["loss_test"])
    assert len(mse_traditional_ensemble) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_ensemble has length {len(mse_traditional_ensemble)}"
    axis.plot(data_x, mse_traditional_ensemble, marker="o",
              label=f"Test - arithmetic-mean-ensemble")

    # Traditional average base learner loss
    mse_traditional_base_learner = np.array(
        df_traditional["loss_test_base_learner"])
    assert len(mse_traditional_base_learner) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_base_learner has length {len(mse_traditional_base_learner)}"
    axis.plot(data_x, mse_traditional_base_learner, marker="o",
              label=f"Test - base learner (average)")

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
        0.5, -0.25), ncol=1, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_ensemble_size_2_16_fixed_architecture_6_fixed_epoch_num_200(raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_traditional = {
        "hidden_size": [
            [6]
        ],
        "epoch_num": [
            200
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
        ]
    }
    filter_mlp = {
        "hidden_size": [
            [6]
        ],
        "epoch_num": [
            200
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"ensemble_size"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs ensemble_size\n(each base learner is [6] trained with 200 epochs)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"traditional_mse_against_ensemble_size_2_16_fixed_architecture_6_fixed_epoch_num_200.png"
    # ====================================================================

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
    data_x = np.array(df_traditional["ensemble_size"].unique())

    # Plot lines
    # Traditional ensemble loss
    mse_traditional_ensemble = np.array(df_traditional["loss_test"])
    assert len(mse_traditional_ensemble) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_ensemble has length {len(mse_traditional_ensemble)}"
    axis.plot(data_x, mse_traditional_ensemble, marker="o",
              label=f"Test - arithmetic-mean-ensemble")

    # Traditional average base learner loss
    mse_traditional_base_learner = np.array(
        df_traditional["loss_test_base_learner"])
    assert len(mse_traditional_base_learner) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_base_learner has length {len(mse_traditional_base_learner)}"
    axis.plot(data_x, mse_traditional_base_learner, marker="o",
              label=f"Test - base learner (average)")

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
        0.5, -0.25), ncol=1, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_ensemble_size_2_16_fixed_architecture_12_fixed_epoch_num_200(raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_traditional = {
        "hidden_size": [
            [12]
        ],
        "epoch_num": [
            200
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
        ]
    }
    filter_mlp = {
        "hidden_size": [
            [12]
        ],
        "epoch_num": [
            200
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"ensemble_size"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs ensemble_size\n(each base learner is [12] trained with 200 epochs)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"traditional_mse_against_ensemble_size_2_16_fixed_architecture_12_fixed_epoch_num_200.png"
    # ====================================================================

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
    data_x = np.array(df_traditional["ensemble_size"].unique())

    # Plot lines
    # Traditional ensemble loss
    mse_traditional_ensemble = np.array(df_traditional["loss_test"])
    assert len(mse_traditional_ensemble) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_ensemble has length {len(mse_traditional_ensemble)}"
    axis.plot(data_x, mse_traditional_ensemble, marker="o",
              label=f"Test - arithmetic-mean-ensemble")

    # Traditional average base learner loss
    mse_traditional_base_learner = np.array(
        df_traditional["loss_test_base_learner"])
    assert len(mse_traditional_base_learner) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_base_learner has length {len(mse_traditional_base_learner)}"
    axis.plot(data_x, mse_traditional_base_learner, marker="o",
              label=f"Test - base learner (average)")

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
        0.5, -0.25), ncol=1, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_ensemble_size_2_16_fixed_architecture_2_fixed_epoch_num_400(raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_traditional = {
        "hidden_size": [
            [2]
        ],
        "epoch_num": [
            400
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
    X_AXIS = f"ensemble_size"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs ensemble_size\n(each base learner is [2] trained with 400 epochs)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"traditional_mse_against_ensemble_size_2_16_fixed_architecture_2_fixed_epoch_num_400.png"
    # ====================================================================

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
    data_x = np.array(df_traditional["ensemble_size"].unique())

    # Plot lines
    # Traditional ensemble loss
    mse_traditional_ensemble = np.array(df_traditional["loss_test"])
    assert len(mse_traditional_ensemble) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_ensemble has length {len(mse_traditional_ensemble)}"
    axis.plot(data_x, mse_traditional_ensemble, marker="o",
              label=f"Test - arithmetic-mean-ensemble")

    # Traditional average base learner loss
    mse_traditional_base_learner = np.array(
        df_traditional["loss_test_base_learner"])
    assert len(mse_traditional_base_learner) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_base_learner has length {len(mse_traditional_base_learner)}"
    axis.plot(data_x, mse_traditional_base_learner, marker="o",
              label=f"Test - base learner (average)")

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
        0.5, -0.25), ncol=1, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_ensemble_size_2_16_fixed_architecture_2_2_2_fixed_epoch_num_400(raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_traditional = {
        "hidden_size": [
            [2, 2, 2]
        ],
        "epoch_num": [
            400
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
    X_AXIS = f"ensemble_size"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs ensemble_size\n(each base learner is [2, 2, 2] trained with 400 epochs)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"traditional_mse_against_ensemble_size_2_16_fixed_architecture_2_2_2_fixed_epoch_num_400.png"
    # ====================================================================

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
    data_x = np.array(df_traditional["ensemble_size"].unique())

    # Plot lines
    # Traditional ensemble loss
    mse_traditional_ensemble = np.array(df_traditional["loss_test"])
    assert len(mse_traditional_ensemble) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_ensemble has length {len(mse_traditional_ensemble)}"
    axis.plot(data_x, mse_traditional_ensemble, marker="o",
              label=f"Test - arithmetic-mean-ensemble")

    # Traditional average base learner loss
    mse_traditional_base_learner = np.array(
        df_traditional["loss_test_base_learner"])
    assert len(mse_traditional_base_learner) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_base_learner has length {len(mse_traditional_base_learner)}"
    axis.plot(data_x, mse_traditional_base_learner, marker="o",
              label=f"Test - base learner (average)")

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
        0.5, -0.25), ncol=1, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_ensemble_size_2_16_fixed_architecture_2_2_2_2_2_2_fixed_epoch_num_400(raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_traditional = {
        "hidden_size": [
            [2, 2, 2, 2, 2, 2]
        ],
        "epoch_num": [
            400
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
    X_AXIS = f"ensemble_size"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs ensemble_size\n(each base learner is [2, 2, 2, 2, 2, 2] trained with 400 epochs)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"traditional_mse_against_ensemble_size_2_16_fixed_architecture_2_2_2_2_2_2_fixed_epoch_num_400.png"
    # ====================================================================

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
    data_x = np.array(df_traditional["ensemble_size"].unique())

    # Plot lines
    # Traditional ensemble loss
    mse_traditional_ensemble = np.array(df_traditional["loss_test"])
    assert len(mse_traditional_ensemble) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_ensemble has length {len(mse_traditional_ensemble)}"
    axis.plot(data_x, mse_traditional_ensemble, marker="o",
              label=f"Test - arithmetic-mean-ensemble")

    # Traditional average base learner loss
    mse_traditional_base_learner = np.array(
        df_traditional["loss_test_base_learner"])
    assert len(mse_traditional_base_learner) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_base_learner has length {len(mse_traditional_base_learner)}"
    axis.plot(data_x, mse_traditional_base_learner, marker="o",
              label=f"Test - base learner (average)")

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


def mse_against_ensemble_size_2_16_fixed_architecture_6_fixed_epoch_num_400(raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_traditional = {
        "hidden_size": [
            [6]
        ],
        "epoch_num": [
            400
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
    X_AXIS = f"ensemble_size"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs ensemble_size\n(each base learner is [6] trained with 400 epochs)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"traditional_mse_against_ensemble_size_2_16_fixed_architecture_6_fixed_epoch_num_400.png"
    # ====================================================================

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
    data_x = np.array(df_traditional["ensemble_size"].unique())

    # Plot lines
    # Traditional ensemble loss
    mse_traditional_ensemble = np.array(df_traditional["loss_test"])
    assert len(mse_traditional_ensemble) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_ensemble has length {len(mse_traditional_ensemble)}"
    axis.plot(data_x, mse_traditional_ensemble, marker="o",
              label=f"Test - arithmetic-mean-ensemble")

    # Traditional average base learner loss
    mse_traditional_base_learner = np.array(
        df_traditional["loss_test_base_learner"])
    assert len(mse_traditional_base_learner) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_base_learner has length {len(mse_traditional_base_learner)}"
    axis.plot(data_x, mse_traditional_base_learner, marker="o",
              label=f"Test - base learner (average)")

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
        0.5, -0.25), ncol=1, borderaxespad=0.)

    # Save image
    figure.savefig(os.path.join(img_repository_path, IMG_NAME),
                   dpi=IMG_RESOLUTION, bbox_inches="tight")

    plt.close(figure)


def mse_against_ensemble_size_2_16_fixed_architecture_12_fixed_epoch_num_400(raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_traditional = {
        "hidden_size": [
            [12]
        ],
        "epoch_num": [
            400
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
    X_AXIS = f"ensemble_size"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs ensemble_size\n(each base learner is [12] trained with 400 epochs)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"traditional_mse_against_ensemble_size_2_16_fixed_architecture_12_fixed_epoch_num_400.png"
    # ====================================================================

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
    data_x = np.array(df_traditional["ensemble_size"].unique())

    # Plot lines
    # Traditional ensemble loss
    mse_traditional_ensemble = np.array(df_traditional["loss_test"])
    assert len(mse_traditional_ensemble) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_ensemble has length {len(mse_traditional_ensemble)}"
    axis.plot(data_x, mse_traditional_ensemble, marker="o",
              label=f"Test - arithmetic-mean-ensemble")

    # Traditional average base learner loss
    mse_traditional_base_learner = np.array(
        df_traditional["loss_test_base_learner"])
    assert len(mse_traditional_base_learner) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_base_learner has length {len(mse_traditional_base_learner)}"
    axis.plot(data_x, mse_traditional_base_learner, marker="o",
              label=f"Test - base learner (average)")

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


def diversity_coefficient_against_ensemble_size_2_16_fixed_epoch_num_400(raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_traditional is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_traditional, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_traditional = {
        "hidden_size": [
            [2],
            [6],
            [12],
            [2, 2, 2],
            [2, 2, 2, 2, 2, 2]
        ],
        "epoch_num": [
            400
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
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"ensemble_size"
    Y_AXIS = f"$\\rho$"
    FIGURE_TITLE = f"$\\rho$ vs ensemble_size\n(each base learner is trained with 400 epochs)"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"traditional_diversity_coefficient_against_ensemble_size_2_16_fixed_epoch_num_400.png"
    # ====================================================================

    df_traditional = data_helper.apply_dataframe_filter(
        df=raw_df_traditional,
        filter=filter_traditional
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(df_traditional["ensemble_size"].unique())

    # Plot lines
    # Arithmetic-mean-ensemble - hidden_size = [2]
    diversity_coefficient_traditional_ensemble_2 = np.array(
        df_traditional[df_traditional["hidden_size"].apply(lambda x: x == [2])]["diversity_coefficient_test"])
    assert len(diversity_coefficient_traditional_ensemble_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_ensemble_2 has length {len(diversity_coefficient_traditional_ensemble_2)}"
    axis.plot(data_x, diversity_coefficient_traditional_ensemble_2,
              marker="o", label=f"Arithmetic-mean-ensemble - [2]")

    # Arithmetic-mean-ensemble - hidden_size = [6]
    diversity_coefficient_traditional_ensemble_6 = np.array(
        df_traditional[df_traditional["hidden_size"].apply(lambda x: x == [6])]["diversity_coefficient_test"])
    assert len(diversity_coefficient_traditional_ensemble_6) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_ensemble_6 has length {len(diversity_coefficient_traditional_ensemble_6)}"
    axis.plot(data_x, diversity_coefficient_traditional_ensemble_6,
              marker="o", label=f"Arithmetic-mean-ensemble - [6]")

    # Arithmetic-mean-ensemble - hidden_size = [12]
    diversity_coefficient_traditional_ensemble_12 = np.array(
        df_traditional[df_traditional["hidden_size"].apply(lambda x: x == [12])]["diversity_coefficient_test"])
    assert len(diversity_coefficient_traditional_ensemble_12) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_ensemble_12 has length {len(diversity_coefficient_traditional_ensemble_12)}"
    axis.plot(data_x, diversity_coefficient_traditional_ensemble_12,
              marker="o", label=f"Arithmetic-mean-ensemble - [12]")

    # Arithmetic-mean-ensemble - hidden_size = [2, 2, 2]
    diversity_coefficient_traditional_ensemble_2_2_2 = np.array(
        df_traditional[df_traditional["hidden_size"].apply(lambda x: x == [2, 2, 2])]["diversity_coefficient_test"])
    assert len(diversity_coefficient_traditional_ensemble_2_2_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_ensemble_2_2_2 has length {len(diversity_coefficient_traditional_ensemble_2_2_2)}"
    axis.plot(data_x, diversity_coefficient_traditional_ensemble_2_2_2,
              marker="o", label=f"Arithmetic-mean-ensemble - [2, 2, 2]")

    # Arithmetic-mean-ensemble - hidden_size = [2, 2, 2, 2, 2, 2]
    diversity_coefficient_traditional_ensemble_2_2_2_2_2_2 = np.array(
        df_traditional[df_traditional["hidden_size"].apply(lambda x: x == [2, 2, 2, 2, 2, 2])]["diversity_coefficient_test"])
    assert len(diversity_coefficient_traditional_ensemble_2_2_2_2_2_2) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_ensemble_2_2_2_2_2_2 has length {len(diversity_coefficient_traditional_ensemble_2_2_2_2_2_2)}"
    axis.plot(data_x, diversity_coefficient_traditional_ensemble_2_2_2_2_2_2,
              marker="o", label=f"Arithmetic-mean-ensemble - [2, 2, 2, 2, 2, 2]")

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
    df_traditional = pd.read_csv(CSV_PATH_TRADITIONAL)
    # NOTE: If we read from csv, df_traditional["hidden_size"] will be of type class "str" instead of "list"
    df_traditional["hidden_size"] = df_traditional["hidden_size"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df_mlp = pd.read_csv(CSV_PATH_MLP)
    # NOTE: If we read from csv, df_mlp["hidden_size"] will be of type class "str" instead of "list"
    df_mlp["hidden_size"] = df_mlp["hidden_size"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    mse_against_ensemble_size_2_16_fixed_architecture_2_fixed_epoch_num_200(
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_TRADITIONAL
    )

    mse_against_ensemble_size_2_16_fixed_architecture_2_2_2_fixed_epoch_num_200(
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_TRADITIONAL
    )

    mse_against_ensemble_size_2_16_fixed_architecture_2_2_2_2_2_2_fixed_epoch_num_200(
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_TRADITIONAL
    )

    mse_against_ensemble_size_2_16_fixed_architecture_6_fixed_epoch_num_200(
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_TRADITIONAL
    )

    mse_against_ensemble_size_2_16_fixed_architecture_12_fixed_epoch_num_200(
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_TRADITIONAL
    )

    mse_against_ensemble_size_2_16_fixed_architecture_2_fixed_epoch_num_400(
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_TRADITIONAL
    )

    mse_against_ensemble_size_2_16_fixed_architecture_2_2_2_fixed_epoch_num_400(
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_TRADITIONAL
    )

    mse_against_ensemble_size_2_16_fixed_architecture_2_2_2_2_2_2_fixed_epoch_num_400(
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_TRADITIONAL
    )

    mse_against_ensemble_size_2_16_fixed_architecture_6_fixed_epoch_num_400(
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_TRADITIONAL
    )

    mse_against_ensemble_size_2_16_fixed_architecture_12_fixed_epoch_num_400(
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_TRADITIONAL
    )

    diversity_coefficient_against_ensemble_size_2_16_fixed_epoch_num_400(
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_TRADITIONAL
    )

    logger.log(f"Graphs of traditional experiment are saved ...")
