import ast
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from src.utils import data_helper, logger

"""
Run: python -m src.utils.experiment.data_plotter_neat_ncl
"""


# Constants
# Path to project root
ROOT_PATH = os.path.join(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Path to each csv file in csv/
CSV_PATH_NEAT_NCL = os.path.join(
    ROOT_PATH, f"data", f"experiment", f"neat-ncl", f"csv", f"experiment_runner_neat_ncl_unconnected_arithmetic_mean.csv")
CSV_PATH_STATIC_NCL = os.path.join(
    ROOT_PATH, f"data", f"experiment", f"static-ncl", f"csv", f"experiment_runner_static_ncl_arithmetic_mean.csv")
CSV_PATH_TRADITIONAL = os.path.join(
    ROOT_PATH, f"data", f"experiment", f"traditional", f"csv", f"experiment_runner_traditional_arithmetic_mean.csv")
CSV_PATH_MLP = os.path.join(
    ROOT_PATH, f"data", f"experiment", f"mlp", f"csv", f"experiment_runner_mlp.csv")

# Path to img/
IMG_REPOSITORY_PATH_NEAT_NCL = os.path.join(
    ROOT_PATH, f"data", f"experiment", f"neat-ncl", f"img")

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


def mse_against_generation_index_0_480(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and raw_df_static_ncl is not None and raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, raw_df_static_ncl, raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            50,
            100,
            150,
            200,
            250
        ]
    }
    filter_static_ncl = {
        "hidden_size": [
            # Pick best hidden_size
            [6]
        ],
        "ensemble_size": [
            50,
            100,
            150,
            200,
            250
        ],
        "correlation_penalty_coefficient": [
            # Pick best correlation_penalty_coefficient
            0.7
        ],
        "epoch_num": [
            400
        ]
    }
    filter_traditional = {
        "hidden_size": [
            # Pick best hidden_size
            [12]
        ],
        "ensemble_size": [
            50,
            100,
            150,
            200,
            250
        ],
        "epoch_num": [
            400
        ]
    }
    filter_mlp = {
        "hidden_size": [
            # Pick best hidden_size
            [12]
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"Number of generations"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs Number of generations"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_mse_against_generation_index_0_480.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )
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
    data_x = np.array(df_neat_ncl["generation_index"].unique())

    # Plot lines
    # NEAT NCL - ensemble_size = 50
    mse_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["loss_test"])
    assert len(mse_neat_ncl_50) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_neat_ncl_50 has length {len(mse_neat_ncl_50)}"
    axis.plot(data_x, mse_neat_ncl_50, marker="o",
              label=f"NEAT NCL - ensemble_size: 50")

    # NEAT NCL - ensemble_size = 100
    mse_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["loss_test"])
    assert len(mse_neat_ncl_100) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_neat_ncl_100 has length {len(mse_neat_ncl_100)}"
    axis.plot(data_x, mse_neat_ncl_100, marker="o",
              label=f"NEAT NCL - ensemble_size: 100")

    # NEAT NCL - ensemble_size = 150
    mse_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["loss_test"])
    assert len(mse_neat_ncl_150) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_neat_ncl_150 has length {len(mse_neat_ncl_150)}"
    axis.plot(data_x, mse_neat_ncl_150, marker="o",
              label=f"NEAT NCL - ensemble_size: 150")

    # NEAT NCL - ensemble_size = 200
    mse_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["loss_test"])
    assert len(mse_neat_ncl_200) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_neat_ncl_200 has length {len(mse_neat_ncl_200)}"
    axis.plot(data_x, mse_neat_ncl_200, marker="o",
              label=f"NEAT NCL - ensemble_size: 200")

    # NEAT NCL - ensemble_size = 250
    mse_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["loss_test"])
    assert len(mse_neat_ncl_250) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_neat_ncl_250 has length {len(mse_neat_ncl_250)}"
    axis.plot(data_x, mse_neat_ncl_250, marker="o",
              label=f"NEAT NCL - ensemble_size: 250")

    # Static NCL - ensemble_size = 50
    # Expect only 1 data point for static NCL originally
    mse_static_ncl_50 = np.repeat(np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 50]["loss_test"]), len(data_x))
    assert len(mse_static_ncl_50) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_50 has length {len(mse_static_ncl_50)}"
    axis.plot(data_x, mse_static_ncl_50, marker="o",
              label=f"NCL - ensemble_size: 50")

    # Static NCL - ensemble_size = 100
    # Expect only 1 data point for static NCL originally
    mse_static_ncl_100 = np.repeat(np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 100]["loss_test"]), len(data_x))
    assert len(mse_static_ncl_100) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_100 has length {len(mse_static_ncl_100)}"
    axis.plot(data_x, mse_static_ncl_100, marker="o",
              label=f"NCL - ensemble_size: 100")

    # Static NCL - ensemble_size = 150
    # Expect only 1 data point for static NCL originally
    mse_static_ncl_150 = np.repeat(np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 150]["loss_test"]), len(data_x))
    assert len(mse_static_ncl_150) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_150 has length {len(mse_static_ncl_150)}"
    axis.plot(data_x, mse_static_ncl_150, marker="o",
              label=f"NCL - ensemble_size: 150")

    # Static NCL - ensemble_size = 200
    # Expect only 1 data point for static NCL originally
    mse_static_ncl_200 = np.repeat(np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 200]["loss_test"]), len(data_x))
    assert len(mse_static_ncl_200) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_200 has length {len(mse_static_ncl_200)}"
    axis.plot(data_x, mse_static_ncl_200, marker="o",
              label=f"NCL - ensemble_size: 200")

    # Static NCL - ensemble_size = 250
    # Expect only 1 data point for static NCL originally
    mse_static_ncl_250 = np.repeat(np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 250]["loss_test"]), len(data_x))
    assert len(mse_static_ncl_250) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_250 has length {len(mse_static_ncl_250)}"
    axis.plot(data_x, mse_static_ncl_250, marker="o",
              label=f"NCL - ensemble_size: 250")

    # Arithmetic-mean-ensemble - ensemble_size = 50
    # Expect only 1 data point for traditional originally
    mse_traditional_50 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 50]["loss_test"]), len(data_x))
    assert len(mse_traditional_50) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_50 has length {len(mse_traditional_50)}"
    axis.plot(data_x, mse_traditional_50, marker="o",
              label=f"Arithmetic-mean-ensemble - ensemble_size: 50")

    # Arithmetic-mean-ensemble - ensemble_size = 100
    # Expect only 1 data point for traditional originally
    mse_traditional_100 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 100]["loss_test"]), len(data_x))
    assert len(mse_traditional_100) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_100 has length {len(mse_traditional_100)}"
    axis.plot(data_x, mse_traditional_100, marker="o",
              label=f"Arithmetic-mean-ensemble - ensemble_size: 100")

    # Arithmetic-mean-ensemble - ensemble_size = 150
    # Expect only 1 data point for traditional originally
    mse_traditional_150 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 150]["loss_test"]), len(data_x))
    assert len(mse_traditional_150) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_150 has length {len(mse_traditional_150)}"
    axis.plot(data_x, mse_traditional_150, marker="o",
              label=f"Arithmetic-mean-ensemble - ensemble_size: 150")

    # Arithmetic-mean-ensemble - ensemble_size = 200
    # Expect only 1 data point for traditional originally
    mse_traditional_200 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 200]["loss_test"]), len(data_x))
    assert len(mse_traditional_200) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_200 has length {len(mse_traditional_200)}"
    axis.plot(data_x, mse_traditional_200, marker="o",
              label=f"Arithmetic-mean-ensemble - ensemble_size: 200")

    # Arithmetic-mean-ensemble - ensemble_size = 250
    # Expect only 1 data point for traditional originally
    mse_traditional_250 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 250]["loss_test"]), len(data_x))
    assert len(mse_traditional_250) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_250 has length {len(mse_traditional_250)}"
    axis.plot(data_x, mse_traditional_250, marker="o",
              label=f"Arithmetic-mean-ensemble - ensemble_size: 250")

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


def mse_against_generation_index_0_480_fixed_ensemble_size_50(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and raw_df_static_ncl is not None and raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, raw_df_static_ncl, raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            50
        ]
    }
    filter_static_ncl = {
        "hidden_size": [
            # Pick best hidden_size
            [6]
        ],
        "ensemble_size": [
            50
        ],
        "correlation_penalty_coefficient": [
            # Pick best correlation_penalty_coefficient
            0.7
        ],
        "epoch_num": [
            400
        ]
    }
    filter_traditional = {
        "hidden_size": [
            # Pick best hidden_size
            [12]
        ],
        "ensemble_size": [
            50
        ],
        "epoch_num": [
            400
        ]
    }
    filter_mlp = {
        "hidden_size": [
            # Pick best hidden_size
            [12]
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"Number of generations"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs Number of generations"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_mse_against_generation_index_0_480_fixed_ensemble_size_50.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )
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
    data_x = np.array(df_neat_ncl["generation_index"].unique())

    # Plot lines
    # NEAT NCL - ensemble_size = 50
    mse_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["loss_test"])
    assert len(mse_neat_ncl_50) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_neat_ncl_50 has length {len(mse_neat_ncl_50)}"
    axis.plot(data_x, mse_neat_ncl_50, marker="o",
              label=f"NEAT NCL - ensemble_size: 50")

    # Static NCL - ensemble_size = 50
    # Expect only 1 data point for static NCL originally
    mse_static_ncl_50 = np.repeat(np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 50]["loss_test"]), len(data_x))
    assert len(mse_static_ncl_50) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_50 has length {len(mse_static_ncl_50)}"
    axis.plot(data_x, mse_static_ncl_50, marker="o",
              label=f"NCL - ensemble_size: 50")

    # Arithmetic-mean-ensemble - ensemble_size = 50
    # Expect only 1 data point for traditional originally
    mse_traditional_50 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 50]["loss_test"]), len(data_x))
    assert len(mse_traditional_50) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_50 has length {len(mse_traditional_50)}"
    axis.plot(data_x, mse_traditional_50, marker="o",
              label=f"Arithmetic-mean-ensemble - ensemble_size: 50")

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


def mse_against_generation_index_0_480_fixed_ensemble_size_100(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and raw_df_static_ncl is not None and raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, raw_df_static_ncl, raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            100
        ]
    }
    filter_static_ncl = {
        "hidden_size": [
            # Pick best hidden_size
            [6]
        ],
        "ensemble_size": [
            100
        ],
        "correlation_penalty_coefficient": [
            # Pick best correlation_penalty_coefficient
            0.7
        ],
        "epoch_num": [
            400
        ]
    }
    filter_traditional = {
        "hidden_size": [
            # Pick best hidden_size
            [12]
        ],
        "ensemble_size": [
            100
        ],
        "epoch_num": [
            400
        ]
    }
    filter_mlp = {
        "hidden_size": [
            # Pick best hidden_size
            [12]
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"Number of generations"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs Number of generations"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_mse_against_generation_index_0_480_fixed_ensemble_size_100.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )
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
    data_x = np.array(df_neat_ncl["generation_index"].unique())

    # Plot lines
    # NEAT NCL - ensemble_size = 100
    mse_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["loss_test"])
    assert len(mse_neat_ncl_100) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_neat_ncl_100 has length {len(mse_neat_ncl_100)}"
    axis.plot(data_x, mse_neat_ncl_100, marker="o",
              label=f"NEAT NCL - ensemble_size: 100")

    # Static NCL - ensemble_size = 100
    # Expect only 1 data point for static NCL originally
    mse_static_ncl_100 = np.repeat(np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 100]["loss_test"]), len(data_x))
    assert len(mse_static_ncl_100) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_100 has length {len(mse_static_ncl_100)}"
    axis.plot(data_x, mse_static_ncl_100, marker="o",
              label=f"NCL - ensemble_size: 100")

    # Arithmetic-mean-ensemble - ensemble_size = 100
    # Expect only 1 data point for traditional originally
    mse_traditional_100 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 100]["loss_test"]), len(data_x))
    assert len(mse_traditional_100) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_100 has length {len(mse_traditional_100)}"
    axis.plot(data_x, mse_traditional_100, marker="o",
              label=f"Arithmetic-mean-ensemble - ensemble_size: 100")

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


def mse_against_generation_index_0_480_fixed_ensemble_size_150(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and raw_df_static_ncl is not None and raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, raw_df_static_ncl, raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            150
        ]
    }
    filter_static_ncl = {
        "hidden_size": [
            # Pick best hidden_size
            [6]
        ],
        "ensemble_size": [
            150
        ],
        "correlation_penalty_coefficient": [
            # Pick best correlation_penalty_coefficient
            0.7
        ],
        "epoch_num": [
            400
        ]
    }
    filter_traditional = {
        "hidden_size": [
            # Pick best hidden_size
            [12]
        ],
        "ensemble_size": [
            150
        ],
        "epoch_num": [
            400
        ]
    }
    filter_mlp = {
        "hidden_size": [
            # Pick best hidden_size
            [12]
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"Number of generations"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs Number of generations"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_mse_against_generation_index_0_480_fixed_ensemble_size_150.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )
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
    data_x = np.array(df_neat_ncl["generation_index"].unique())

    # Plot lines
    # NEAT NCL - ensemble_size = 150
    mse_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["loss_test"])
    assert len(mse_neat_ncl_150) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_neat_ncl_150 has length {len(mse_neat_ncl_150)}"
    axis.plot(data_x, mse_neat_ncl_150, marker="o",
              label=f"NEAT NCL - ensemble_size: 150")

    # Static NCL - ensemble_size = 150
    mse_static_ncl_150 = np.repeat(np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 150]["loss_test"]), len(data_x))
    assert len(mse_static_ncl_150) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_150 has length {len(mse_static_ncl_150)}"
    axis.plot(data_x, mse_static_ncl_150, marker="o",
              label=f"NCL - ensemble_size: 150")

    # Arithmetic-mean-ensemble - ensemble_size = 150
    # Expect only 1 data point for traditional originally
    mse_traditional_150 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 150]["loss_test"]), len(data_x))
    assert len(mse_traditional_150) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_150 has length {len(mse_traditional_150)}"
    axis.plot(data_x, mse_traditional_150, marker="o",
              label=f"Arithmetic-mean-ensemble - ensemble_size: 150")

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


def mse_against_generation_index_0_480_fixed_ensemble_size_200(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and raw_df_static_ncl is not None and raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, raw_df_static_ncl, raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            200
        ]
    }
    filter_static_ncl = {
        "hidden_size": [
            # Pick best hidden_size
            [6]
        ],
        "ensemble_size": [
            200
        ],
        "correlation_penalty_coefficient": [
            # Pick best correlation_penalty_coefficient
            0.7
        ],
        "epoch_num": [
            400
        ]
    }
    filter_traditional = {
        "hidden_size": [
            # Pick best hidden_size
            [12]
        ],
        "ensemble_size": [
            200
        ],
        "epoch_num": [
            400
        ]
    }
    filter_mlp = {
        "hidden_size": [
            # Pick best hidden_size
            [12]
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"Number of generations"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs Number of generations"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_mse_against_generation_index_0_480_fixed_ensemble_size_200.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )
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
    data_x = np.array(df_neat_ncl["generation_index"].unique())

    # Plot lines
    # NEAT NCL - ensemble_size = 200
    mse_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["loss_test"])
    assert len(mse_neat_ncl_200) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_neat_ncl_200 has length {len(mse_neat_ncl_200)}"
    axis.plot(data_x, mse_neat_ncl_200, marker="o",
              label=f"NEAT NCL - ensemble_size: 200")

    # Static NCL - ensemble_size = 200
    # Expect only 1 data point for static NCL originally
    mse_static_ncl_200 = np.repeat(np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 200]["loss_test"]), len(data_x))
    assert len(mse_static_ncl_200) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_200 has length {len(mse_static_ncl_200)}"
    axis.plot(data_x, mse_static_ncl_200, marker="o",
              label=f"NCL - ensemble_size: 200")

    # Arithmetic-mean-ensemble - ensemble_size = 200
    # Expect only 1 data point for traditional originally
    mse_traditional_200 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 200]["loss_test"]), len(data_x))
    assert len(mse_traditional_200) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_200 has length {len(mse_traditional_200)}"
    axis.plot(data_x, mse_traditional_200, marker="o",
              label=f"Arithmetic-mean-ensemble - ensemble_size: 200")

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


def mse_against_generation_index_0_480_fixed_ensemble_size_250(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and raw_df_static_ncl is not None and raw_df_traditional is not None and raw_df_mlp is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, raw_df_static_ncl, raw_df_traditional, raw_df_mlp, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            250
        ]
    }
    filter_static_ncl = {
        "hidden_size": [
            # Pick best hidden_size
            [6]
        ],
        "ensemble_size": [
            250
        ],
        "correlation_penalty_coefficient": [
            # Pick best correlation_penalty_coefficient
            0.7
        ],
        "epoch_num": [
            400
        ]
    }
    filter_traditional = {
        "hidden_size": [
            # Pick best hidden_size
            [12]
        ],
        "ensemble_size": [
            250
        ],
        "epoch_num": [
            400
        ]
    }
    filter_mlp = {
        "hidden_size": [
            # Pick best hidden_size
            [12]
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"Number of generations"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs Number of generations"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_mse_against_generation_index_0_480_fixed_ensemble_size_250.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )
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
    data_x = np.array(df_neat_ncl["generation_index"].unique())

    # Plot lines
    # NEAT NCL - ensemble_size = 250
    mse_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["loss_test"])
    assert len(mse_neat_ncl_250) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_neat_ncl_250 has length {len(mse_neat_ncl_250)}"
    axis.plot(data_x, mse_neat_ncl_250, marker="o",
              label=f"NEAT NCL - ensemble_size: 250")

    # Static NCL - ensemble_size = 250
    mse_static_ncl_250 = np.repeat(np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 250]["loss_test"]), len(data_x))
    assert len(mse_static_ncl_250) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_static_ncl_250 has length {len(mse_static_ncl_250)}"
    axis.plot(data_x, mse_static_ncl_250, marker="o",
              label=f"NCL - ensemble_size: 250")

    # Arithmetic-mean-ensemble - ensemble_size = 250
    # Expect only 1 data point for traditional originally
    mse_traditional_250 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 250]["loss_test"]), len(data_x))
    assert len(mse_traditional_250) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_traditional_250 has length {len(mse_traditional_250)}"
    axis.plot(data_x, mse_traditional_250, marker="o",
              label=f"Arithmetic-mean-ensemble - ensemble_size: 250")

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


def mse_against_diversity_coefficient_fixed_ensemble_size_50(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and raw_df_static_ncl is not None and raw_df_traditional is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, raw_df_static_ncl, raw_df_traditional, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            50
        ]
    }
    filter_static_ncl = {
        "hidden_size": [
            # Pick best hidden_size
            [6]
        ],
        "ensemble_size": [
            50
        ],
        "correlation_penalty_coefficient": [
            # Pick best correlation_penalty_coefficient
            0.7
        ],
        "epoch_num": [
            400
        ]
    }
    filter_traditional = {
        "hidden_size": [
            # Pick best hidden_size
            [12]
        ],
        "ensemble_size": [
            50
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\rho$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\rho$"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_mse_against_diversity_coefficient_fixed_ensemble_size_50.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )
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

    # Plot lines
    # NEAT NCL - ensemble_size = 50
    mse_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["loss_test"])
    diversity_coefficient_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["diversity_coefficient_test"])
    assert len(mse_neat_ncl_50) == len(
        diversity_coefficient_neat_ncl_50), f"Invalid number of data points (expect {len(diversity_coefficient_neat_ncl_50)}): mse_neat_ncl_50 has length {len(mse_neat_ncl_50)}"
    axis.scatter(diversity_coefficient_neat_ncl_50, mse_neat_ncl_50,
                 marker="o", label=f"NEAT NCL - ensemble_size: 50")

    # Static NCL - ensemble_size = 50
    # Expect only 1 data point for static NCL originally
    mse_static_ncl_50 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 50]["loss_test"])
    diversity_coefficient_static_ncl_50 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 50]["diversity_coefficient_test"])
    assert len(mse_static_ncl_50) == len(
        diversity_coefficient_static_ncl_50), f"Invalid number of data points (expect {len(diversity_coefficient_static_ncl_50)}): mse_static_ncl_50 has length {len(mse_static_ncl_50)}"
    axis.scatter(diversity_coefficient_static_ncl_50, mse_static_ncl_50,
                 marker="o", label=f"NCL - ensemble_size: 50")

    # Arithmetic-mean-ensemble - ensemble_size = 50
    # Expect only 1 data point for traditional originally
    mse_traditional_50 = np.array(
        df_traditional[df_traditional["ensemble_size"] == 50]["loss_test"])
    diversity_coefficient_traditional_50 = np.array(
        df_traditional[df_traditional["ensemble_size"] == 50]["diversity_coefficient_test"])
    assert len(mse_traditional_50) == len(
        diversity_coefficient_traditional_50), f"Invalid number of data points (expect {len(diversity_coefficient_traditional_50)}): mse_traditional_50 has length {len(mse_traditional_50)}"
    axis.scatter(diversity_coefficient_traditional_50, mse_traditional_50,
                 marker="o", label=f"Arithmetic-mean-ensemble - ensemble_size: 50")

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


def mse_against_diversity_coefficient_fixed_ensemble_size_100(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and raw_df_static_ncl is not None and raw_df_traditional is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, raw_df_static_ncl, raw_df_traditional, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            100
        ]
    }
    filter_static_ncl = {
        "hidden_size": [
            # Pick best hidden_size
            [6]
        ],
        "ensemble_size": [
            100
        ],
        "correlation_penalty_coefficient": [
            # Pick best correlation_penalty_coefficient
            0.7
        ],
        "epoch_num": [
            400
        ]
    }
    filter_traditional = {
        "hidden_size": [
            # Pick best hidden_size
            [12]
        ],
        "ensemble_size": [
            100
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\rho$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\rho$"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_mse_against_diversity_coefficient_fixed_ensemble_size_100.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )
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

    # Plot lines
    # NEAT NCL - ensemble_size = 100
    mse_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["loss_test"])
    diversity_coefficient_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["diversity_coefficient_test"])
    assert len(mse_neat_ncl_100) == len(
        diversity_coefficient_neat_ncl_100), f"Invalid number of data points (expect {len(diversity_coefficient_neat_ncl_100)}): mse_neat_ncl_100 has length {len(mse_neat_ncl_100)}"
    axis.scatter(diversity_coefficient_neat_ncl_100, mse_neat_ncl_100,
                 marker="o", label=f"NEAT NCL - ensemble_size: 100")

    # Static NCL - ensemble_size = 100
    # Expect only 1 data point for static NCL originally
    mse_static_ncl_100 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 100]["loss_test"])
    diversity_coefficient_static_ncl_100 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 100]["diversity_coefficient_test"])
    assert len(mse_static_ncl_100) == len(
        diversity_coefficient_static_ncl_100), f"Invalid number of data points (expect {len(diversity_coefficient_static_ncl_100)}): mse_static_ncl_100 has length {len(mse_static_ncl_100)}"
    axis.scatter(diversity_coefficient_static_ncl_100, mse_static_ncl_100,
                 marker="o", label=f"NCL - ensemble_size: 100")

    # Arithmetic-mean-ensemble - ensemble_size = 100
    # Expect only 1 data point for traditional originally
    mse_traditional_100 = np.array(
        df_traditional[df_traditional["ensemble_size"] == 100]["loss_test"])
    diversity_coefficient_traditional_100 = np.array(
        df_traditional[df_traditional["ensemble_size"] == 100]["diversity_coefficient_test"])
    assert len(mse_traditional_100) == len(
        diversity_coefficient_traditional_100), f"Invalid number of data points (expect {len(diversity_coefficient_traditional_100)}): mse_traditional_100 has length {len(mse_traditional_100)}"
    axis.scatter(diversity_coefficient_traditional_100, mse_traditional_100,
                 marker="o", label=f"Arithmetic-mean-ensemble - ensemble_size: 100")

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


def mse_against_diversity_coefficient_fixed_ensemble_size_150(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and raw_df_static_ncl is not None and raw_df_traditional is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, raw_df_static_ncl, raw_df_traditional, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            150
        ]
    }
    filter_static_ncl = {
        "hidden_size": [
            # Pick best hidden_size
            [6]
        ],
        "ensemble_size": [
            150
        ],
        "correlation_penalty_coefficient": [
            # Pick best correlation_penalty_coefficient
            0.7
        ],
        "epoch_num": [
            400
        ]
    }
    filter_traditional = {
        "hidden_size": [
            # Pick best hidden_size
            [12]
        ],
        "ensemble_size": [
            150
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\rho$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\rho$"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_mse_against_diversity_coefficient_fixed_ensemble_size_150.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )
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

    # Plot lines
    # NEAT NCL - ensemble_size = 150
    mse_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["loss_test"])
    diversity_coefficient_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["diversity_coefficient_test"])
    assert len(mse_neat_ncl_150) == len(
        diversity_coefficient_neat_ncl_150), f"Invalid number of data points (expect {len(diversity_coefficient_neat_ncl_150)}): mse_neat_ncl_150 has length {len(mse_neat_ncl_150)}"
    axis.scatter(diversity_coefficient_neat_ncl_150, mse_neat_ncl_150,
                 marker="o", label=f"NEAT NCL - ensemble_size: 150")

    # Static NCL - ensemble_size = 150
    # Expect only 1 data point for static NCL originally
    mse_static_ncl_150 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 150]["loss_test"])
    diversity_coefficient_static_ncl_150 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 150]["diversity_coefficient_test"])
    assert len(mse_static_ncl_150) == len(
        diversity_coefficient_static_ncl_150), f"Invalid number of data points (expect {len(diversity_coefficient_static_ncl_150)}): mse_static_ncl_150 has length {len(mse_static_ncl_150)}"
    axis.scatter(diversity_coefficient_static_ncl_150, mse_static_ncl_150,
                 marker="o", label=f"NCL - ensemble_size: 150")

    # Arithmetic-mean-ensemble - ensemble_size = 150
    # Expect only 1 data point for traditional originally
    mse_traditional_150 = np.array(
        df_traditional[df_traditional["ensemble_size"] == 150]["loss_test"])
    diversity_coefficient_traditional_150 = np.array(
        df_traditional[df_traditional["ensemble_size"] == 150]["diversity_coefficient_test"])
    assert len(mse_traditional_150) == len(
        diversity_coefficient_traditional_150), f"Invalid number of data points (expect {len(diversity_coefficient_traditional_150)}): mse_traditional_150 has length {len(mse_traditional_150)}"
    axis.scatter(diversity_coefficient_traditional_150, mse_traditional_150,
                 marker="o", label=f"Arithmetic-mean-ensemble - ensemble_size: 150")

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


def mse_against_diversity_coefficient_fixed_ensemble_size_200(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and raw_df_static_ncl is not None and raw_df_traditional is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, raw_df_static_ncl, raw_df_traditional, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            200
        ]
    }
    filter_static_ncl = {
        "hidden_size": [
            # Pick best hidden_size
            [6]
        ],
        "ensemble_size": [
            200
        ],
        "correlation_penalty_coefficient": [
            # Pick best correlation_penalty_coefficient
            0.7
        ],
        "epoch_num": [
            400
        ]
    }
    filter_traditional = {
        "hidden_size": [
            # Pick best hidden_size
            [12]
        ],
        "ensemble_size": [
            200
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\rho$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\rho$"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_mse_against_diversity_coefficient_fixed_ensemble_size_200.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )
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

    # Plot lines
    # NEAT NCL - ensemble_size = 200
    mse_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["loss_test"])
    diversity_coefficient_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["diversity_coefficient_test"])
    assert len(mse_neat_ncl_200) == len(
        diversity_coefficient_neat_ncl_200), f"Invalid number of data points (expect {len(diversity_coefficient_neat_ncl_200)}): mse_neat_ncl_200 has length {len(mse_neat_ncl_200)}"
    axis.scatter(diversity_coefficient_neat_ncl_200, mse_neat_ncl_200,
                 marker="o", label=f"NEAT NCL - ensemble_size: 200")

    # Static NCL - ensemble_size = 200
    # Expect only 1 data point for static NCL originally
    mse_static_ncl_200 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 200]["loss_test"])
    diversity_coefficient_static_ncl_200 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 200]["diversity_coefficient_test"])
    assert len(mse_static_ncl_200) == len(
        diversity_coefficient_static_ncl_200), f"Invalid number of data points (expect {len(diversity_coefficient_static_ncl_200)}): mse_static_ncl_200 has length {len(mse_static_ncl_200)}"
    axis.scatter(diversity_coefficient_static_ncl_200, mse_static_ncl_200,
                 marker="o", label=f"NCL - ensemble_size: 200")

    # Arithmetic-mean-ensemble - ensemble_size = 200
    # Expect only 1 data point for traditional originally
    mse_traditional_200 = np.array(
        df_traditional[df_traditional["ensemble_size"] == 200]["loss_test"])
    diversity_coefficient_traditional_200 = np.array(
        df_traditional[df_traditional["ensemble_size"] == 200]["diversity_coefficient_test"])
    assert len(mse_traditional_200) == len(
        diversity_coefficient_traditional_200), f"Invalid number of data points (expect {len(diversity_coefficient_traditional_200)}): mse_traditional_200 has length {len(mse_traditional_200)}"
    axis.scatter(diversity_coefficient_traditional_200, mse_traditional_200,
                 marker="o", label=f"Arithmetic-mean-ensemble - ensemble_size: 200")

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


def mse_against_diversity_coefficient_fixed_ensemble_size_250(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and raw_df_static_ncl is not None and raw_df_traditional is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, raw_df_static_ncl, raw_df_traditional, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            250
        ]
    }
    filter_static_ncl = {
        "hidden_size": [
            # Pick best hidden_size
            [6]
        ],
        "ensemble_size": [
            250
        ],
        "correlation_penalty_coefficient": [
            # Pick best correlation_penalty_coefficient
            0.7
        ],
        "epoch_num": [
            400
        ]
    }
    filter_traditional = {
        "hidden_size": [
            # Pick best hidden_size
            [12]
        ],
        "ensemble_size": [
            250
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\rho$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $\\rho$"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_mse_against_diversity_coefficient_fixed_ensemble_size_250.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )
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

    # Plot lines
    # NEAT NCL - ensemble_size = 250
    mse_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["loss_test"])
    diversity_coefficient_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["diversity_coefficient_test"])
    assert len(mse_neat_ncl_250) == len(
        diversity_coefficient_neat_ncl_250), f"Invalid number of data points (expect {len(diversity_coefficient_neat_ncl_250)}): mse_neat_ncl_250 has length {len(mse_neat_ncl_250)}"
    axis.scatter(diversity_coefficient_neat_ncl_250, mse_neat_ncl_250,
                 marker="o", label=f"NEAT NCL - ensemble_size: 250")

    # Static NCL - ensemble_size = 250
    # Expect only 1 data point for static NCL originally
    mse_static_ncl_250 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 250]["loss_test"])
    diversity_coefficient_static_ncl_250 = np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 250]["diversity_coefficient_test"])
    assert len(mse_static_ncl_250) == len(
        diversity_coefficient_static_ncl_250), f"Invalid number of data points (expect {len(diversity_coefficient_static_ncl_250)}): mse_static_ncl_250 has length {len(mse_static_ncl_250)}"
    axis.scatter(diversity_coefficient_static_ncl_250, mse_static_ncl_250,
                 marker="o", label=f"NCL - ensemble_size: 250")

    # Arithmetic-mean-ensemble - ensemble_size = 250
    # Expect only 1 data point for traditional originally
    mse_traditional_250 = np.array(
        df_traditional[df_traditional["ensemble_size"] == 250]["loss_test"])
    diversity_coefficient_traditional_250 = np.array(
        df_traditional[df_traditional["ensemble_size"] == 250]["diversity_coefficient_test"])
    assert len(mse_traditional_250) == len(
        diversity_coefficient_traditional_250), f"Invalid number of data points (expect {len(diversity_coefficient_traditional_250)}): mse_traditional_250 has length {len(mse_traditional_250)}"
    axis.scatter(diversity_coefficient_traditional_250, mse_traditional_250,
                 marker="o", label=f"Arithmetic-mean-ensemble - ensemble_size: 250")

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


def mse_against_population_diversity(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            50,
            100,
            150,
            200,
            250
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$D$"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs $D$"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_mse_against_population_diversity.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # Plot lines
    # NEAT NCL - ensemble_size = 50
    mse_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["loss_test"])
    population_diversity_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["population_diversity_test"])
    assert len(mse_neat_ncl_50) == len(
        population_diversity_neat_ncl_50), f"Invalid number of data points (expect {len(population_diversity_neat_ncl_50)}): mse_neat_ncl_50 has length {len(mse_neat_ncl_50)}"
    axis.scatter(population_diversity_neat_ncl_50, mse_neat_ncl_50,
                 marker="o", label=f"NEAT NCL - ensemble_size: 50")

    # NEAT NCL - ensemble_size = 100
    mse_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["loss_test"])
    population_diversity_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["population_diversity_test"])
    assert len(mse_neat_ncl_100) == len(
        population_diversity_neat_ncl_100), f"Invalid number of data points (expect {len(population_diversity_neat_ncl_100)}): mse_neat_ncl_100 has length {len(mse_neat_ncl_100)}"
    axis.scatter(population_diversity_neat_ncl_100, mse_neat_ncl_100,
                 marker="o", label=f"NEAT NCL - ensemble_size: 100")

    # NEAT NCL - ensemble_size = 150
    mse_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["loss_test"])
    population_diversity_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["population_diversity_test"])
    assert len(mse_neat_ncl_150) == len(
        population_diversity_neat_ncl_150), f"Invalid number of data points (expect {len(population_diversity_neat_ncl_150)}): mse_neat_ncl_150 has length {len(mse_neat_ncl_150)}"
    axis.scatter(population_diversity_neat_ncl_150, mse_neat_ncl_150,
                 marker="o", label=f"NEAT NCL - ensemble_size: 150")

    # NEAT NCL - ensemble_size = 200
    mse_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["loss_test"])
    population_diversity_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["population_diversity_test"])
    assert len(mse_neat_ncl_200) == len(
        population_diversity_neat_ncl_200), f"Invalid number of data points (expect {len(population_diversity_neat_ncl_200)}): mse_neat_ncl_200 has length {len(mse_neat_ncl_200)}"
    axis.scatter(population_diversity_neat_ncl_200, mse_neat_ncl_200,
                 marker="o", label=f"NEAT NCL - ensemble_size: 200")

    # NEAT NCL - ensemble_size = 250
    mse_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["loss_test"])
    population_diversity_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["population_diversity_test"])
    assert len(mse_neat_ncl_250) == len(
        population_diversity_neat_ncl_250), f"Invalid number of data points (expect {len(population_diversity_neat_ncl_250)}): mse_neat_ncl_250 has length {len(mse_neat_ncl_250)}"
    axis.scatter(population_diversity_neat_ncl_250, mse_neat_ncl_250,
                 marker="o", label=f"NEAT NCL - ensemble_size: 250")

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


def correlation_penalty_coefficient_against_generation_index_0_480(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            50,
            100,
            150,
            200,
            250
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"Number of generations"
    Y_AXIS = f"$\\lambda$"
    FIGURE_TITLE = f"$\\lambda$ vs Number of generations"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_correlation_penalty_coefficient_against_generation_index_0_480.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(df_neat_ncl["generation_index"].unique())

    # Plot lines
    # NEAT NCL - ensemble_size = 50
    correlation_penalty_coefficient_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["correlation_penalty_coefficient_test"])
    assert len(correlation_penalty_coefficient_neat_ncl_50) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): correlation_penalty_coefficient_neat_ncl_50 has length {len(correlation_penalty_coefficient_neat_ncl_50)}"
    axis.plot(data_x, correlation_penalty_coefficient_neat_ncl_50,
              marker="o", label=f"NEAT NCL - ensemble_size: 50")

    # NEAT NCL - ensemble_size = 100
    correlation_penalty_coefficient_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["correlation_penalty_coefficient_test"])
    assert len(correlation_penalty_coefficient_neat_ncl_100) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): correlation_penalty_coefficient_neat_ncl_100 has length {len(correlation_penalty_coefficient_neat_ncl_100)}"
    axis.plot(data_x, correlation_penalty_coefficient_neat_ncl_100,
              marker="o", label=f"NEAT NCL - ensemble_size: 100")

    # NEAT NCL - ensemble_size = 150
    correlation_penalty_coefficient_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["correlation_penalty_coefficient_test"])
    assert len(correlation_penalty_coefficient_neat_ncl_150) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): correlation_penalty_coefficient_neat_ncl_150 has length {len(correlation_penalty_coefficient_neat_ncl_150)}"
    axis.plot(data_x, correlation_penalty_coefficient_neat_ncl_150,
              marker="o", label=f"NEAT NCL - ensemble_size: 150")

    # NEAT NCL - ensemble_size = 200
    correlation_penalty_coefficient_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["correlation_penalty_coefficient_test"])
    assert len(correlation_penalty_coefficient_neat_ncl_200) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): correlation_penalty_coefficient_neat_ncl_200 has length {len(correlation_penalty_coefficient_neat_ncl_200)}"
    axis.plot(data_x, correlation_penalty_coefficient_neat_ncl_200,
              marker="o", label=f"NEAT NCL - ensemble_size: 200")

    # NEAT NCL - ensemble_size = 250
    correlation_penalty_coefficient_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["correlation_penalty_coefficient_test"])
    assert len(correlation_penalty_coefficient_neat_ncl_250) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): correlation_penalty_coefficient_neat_ncl_250 has length {len(correlation_penalty_coefficient_neat_ncl_250)}"
    axis.plot(data_x, correlation_penalty_coefficient_neat_ncl_250,
              marker="o", label=f"NEAT NCL - ensemble_size: 250")

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


def correlation_penalty_coefficient_against_population_diversity(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            50,
            100,
            150,
            200,
            250
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$D$"
    Y_AXIS = f"$\\lambda$"
    FIGURE_TITLE = f"$\\lambda$ vs $D$"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_correlation_penalty_coefficient_against_population_diversity.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # Plot lines
    # NEAT NCL - ensemble_size = 50
    correlation_penalty_coefficient_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["correlation_penalty_coefficient_test"])
    population_diversity_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["population_diversity_test"])
    assert len(correlation_penalty_coefficient_neat_ncl_50) == len(
        population_diversity_neat_ncl_50), f"Invalid number of data points (expect {len(population_diversity_neat_ncl_50)}): correlation_penalty_coefficient_neat_ncl_50 has length {len(correlation_penalty_coefficient_neat_ncl_50)}"
    axis.scatter(population_diversity_neat_ncl_50, correlation_penalty_coefficient_neat_ncl_50,
                 marker="o", label=f"NEAT NCL - ensemble_size: 50")

    # NEAT NCL - ensemble_size = 100
    correlation_penalty_coefficient_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["correlation_penalty_coefficient_test"])
    population_diversity_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["population_diversity_test"])
    assert len(correlation_penalty_coefficient_neat_ncl_100) == len(
        population_diversity_neat_ncl_100), f"Invalid number of data points (expect {len(population_diversity_neat_ncl_100)}): correlation_penalty_coefficient_neat_ncl_100 has length {len(correlation_penalty_coefficient_neat_ncl_100)}"
    axis.scatter(population_diversity_neat_ncl_100, correlation_penalty_coefficient_neat_ncl_100,
                 marker="o", label=f"NEAT NCL - ensemble_size: 100")

    # NEAT NCL - ensemble_size = 150
    correlation_penalty_coefficient_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["correlation_penalty_coefficient_test"])
    population_diversity_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["population_diversity_test"])
    assert len(correlation_penalty_coefficient_neat_ncl_150) == len(
        population_diversity_neat_ncl_150), f"Invalid number of data points (expect {len(population_diversity_neat_ncl_150)}): correlation_penalty_coefficient_neat_ncl_150 has length {len(correlation_penalty_coefficient_neat_ncl_150)}"
    axis.scatter(population_diversity_neat_ncl_150, correlation_penalty_coefficient_neat_ncl_150,
                 marker="o", label=f"NEAT NCL - ensemble_size: 150")

    # NEAT NCL - ensemble_size = 200
    correlation_penalty_coefficient_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["correlation_penalty_coefficient_test"])
    population_diversity_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["population_diversity_test"])
    assert len(correlation_penalty_coefficient_neat_ncl_200) == len(
        population_diversity_neat_ncl_200), f"Invalid number of data points (expect {len(population_diversity_neat_ncl_200)}): correlation_penalty_coefficient_neat_ncl_200 has length {len(correlation_penalty_coefficient_neat_ncl_200)}"
    axis.scatter(population_diversity_neat_ncl_200, correlation_penalty_coefficient_neat_ncl_200,
                 marker="o", label=f"NEAT NCL - ensemble_size: 200")

    # NEAT NCL - ensemble_size = 250
    correlation_penalty_coefficient_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["correlation_penalty_coefficient_test"])
    population_diversity_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["population_diversity_test"])
    assert len(correlation_penalty_coefficient_neat_ncl_250) == len(
        population_diversity_neat_ncl_250), f"Invalid number of data points (expect {len(population_diversity_neat_ncl_250)}): correlation_penalty_coefficient_neat_ncl_250 has length {len(correlation_penalty_coefficient_neat_ncl_250)}"
    axis.scatter(population_diversity_neat_ncl_250, correlation_penalty_coefficient_neat_ncl_250,
                 marker="o", label=f"NEAT NCL - ensemble_size: 250")

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


def correlation_penalty_coefficient_against_population_diversity_ratio(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            50,
            100,
            150,
            200,
            250
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\frac{{D}}{{D_{{max}}}}$"
    Y_AXIS = f"$\\lambda$"
    FIGURE_TITLE = f"$\\lambda$ vs $\\frac{{D}}{{D_{{max}}}}$"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_correlation_penalty_coefficient_against_population_diversity_ratio.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # Plot lines
    # NEAT NCL - ensemble_size = 50
    correlation_penalty_coefficient_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["correlation_penalty_coefficient_test"])
    numerator_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["population_diversity_test"])
    denominator_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["max_population_diversity_test"])
    population_diversity_ratio_neat_ncl_50 = np.divide(
        numerator_neat_ncl_50,
        denominator_neat_ncl_50,
        # NOTE: We assume D / D_{max} = 0 when D_{max} = 0 (this decision is made based on analysing the graph of correlation_penalty_coefficient against population_diversity_ratio
        out=np.zeros_like(numerator_neat_ncl_50, dtype=float),
        where=denominator_neat_ncl_50 != 0
    )
    assert len(correlation_penalty_coefficient_neat_ncl_50) == len(
        population_diversity_ratio_neat_ncl_50), f"Invalid number of data points (expect {len(population_diversity_ratio_neat_ncl_50)}): correlation_penalty_coefficient_neat_ncl_50 has length {len(correlation_penalty_coefficient_neat_ncl_50)}"
    axis.scatter(population_diversity_ratio_neat_ncl_50, correlation_penalty_coefficient_neat_ncl_50,
                 marker="o", label=f"NEAT NCL - ensemble_size: 50")

    # NEAT NCL - ensemble_size = 100
    correlation_penalty_coefficient_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["correlation_penalty_coefficient_test"])
    numerator_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["population_diversity_test"])
    denominator_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["max_population_diversity_test"])
    population_diversity_ratio_neat_ncl_100 = np.divide(
        numerator_neat_ncl_100,
        denominator_neat_ncl_100,
        # NOTE: We assume D / D_{max} = 0 when D_{max} = 0 (this decision is made based on analysing the graph of correlation_penalty_coefficient against population_diversity_ratio
        out=np.zeros_like(numerator_neat_ncl_100, dtype=float),
        where=denominator_neat_ncl_100 != 0
    )
    assert len(correlation_penalty_coefficient_neat_ncl_100) == len(
        population_diversity_ratio_neat_ncl_100), f"Invalid number of data points (expect {len(population_diversity_ratio_neat_ncl_100)}): correlation_penalty_coefficient_neat_ncl_100 has length {len(correlation_penalty_coefficient_neat_ncl_100)}"
    axis.scatter(population_diversity_ratio_neat_ncl_100, correlation_penalty_coefficient_neat_ncl_100,
                 marker="o", label=f"NEAT NCL - ensemble_size: 100")

    # NEAT NCL - ensemble_size = 150
    correlation_penalty_coefficient_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["correlation_penalty_coefficient_test"])
    numerator_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["population_diversity_test"])
    denominator_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["max_population_diversity_test"])
    population_diversity_ratio_neat_ncl_150 = np.divide(
        numerator_neat_ncl_150,
        denominator_neat_ncl_150,
        # NOTE: We assume D / D_{max} = 0 when D_{max} = 0 (this decision is made based on analysing the graph of correlation_penalty_coefficient against population_diversity_ratio
        out=np.zeros_like(numerator_neat_ncl_150, dtype=float),
        where=denominator_neat_ncl_150 != 0
    )
    assert len(correlation_penalty_coefficient_neat_ncl_150) == len(
        population_diversity_ratio_neat_ncl_150), f"Invalid number of data points (expect {len(population_diversity_ratio_neat_ncl_150)}): correlation_penalty_coefficient_neat_ncl_150 has length {len(correlation_penalty_coefficient_neat_ncl_150)}"
    axis.scatter(population_diversity_ratio_neat_ncl_150, correlation_penalty_coefficient_neat_ncl_150,
                 marker="o", label=f"NEAT NCL - ensemble_size: 150")

    # NEAT NCL - ensemble_size = 200
    correlation_penalty_coefficient_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["correlation_penalty_coefficient_test"])
    numerator_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["population_diversity_test"])
    denominator_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["max_population_diversity_test"])
    population_diversity_ratio_neat_ncl_200 = np.divide(
        numerator_neat_ncl_200,
        denominator_neat_ncl_200,
        # NOTE: We assume D / D_{max} = 0 when D_{max} = 0 (this decision is made based on analysing the graph of correlation_penalty_coefficient against population_diversity_ratio
        out=np.zeros_like(numerator_neat_ncl_200, dtype=float),
        where=denominator_neat_ncl_200 != 0
    )
    assert len(correlation_penalty_coefficient_neat_ncl_200) == len(
        population_diversity_ratio_neat_ncl_200), f"Invalid number of data points (expect {len(population_diversity_ratio_neat_ncl_200)}): correlation_penalty_coefficient_neat_ncl_200 has length {len(correlation_penalty_coefficient_neat_ncl_200)}"
    axis.scatter(population_diversity_ratio_neat_ncl_200, correlation_penalty_coefficient_neat_ncl_200,
                 marker="o", label=f"NEAT NCL - ensemble_size: 200")

    # NEAT NCL - ensemble_size = 250
    correlation_penalty_coefficient_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["correlation_penalty_coefficient_test"])
    numerator_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["population_diversity_test"])
    denominator_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["max_population_diversity_test"])
    population_diversity_ratio_neat_ncl_250 = np.divide(
        numerator_neat_ncl_250,
        denominator_neat_ncl_250,
        # NOTE: We assume D / D_{max} = 0 when D_{max} = 0 (this decision is made based on analysing the graph of correlation_penalty_coefficient against population_diversity_ratio
        out=np.zeros_like(numerator_neat_ncl_250, dtype=float),
        where=denominator_neat_ncl_250 != 0
    )
    assert len(correlation_penalty_coefficient_neat_ncl_250) == len(
        population_diversity_ratio_neat_ncl_250), f"Invalid number of data points (expect {len(population_diversity_ratio_neat_ncl_250)}): correlation_penalty_coefficient_neat_ncl_250 has length {len(correlation_penalty_coefficient_neat_ncl_250)}"
    axis.scatter(population_diversity_ratio_neat_ncl_250, correlation_penalty_coefficient_neat_ncl_250,
                 marker="o", label=f"NEAT NCL - ensemble_size: 250")

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


def population_diversity_against_generation_index_0_480(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            50,
            100,
            150,
            200,
            250
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"Number of generations"
    Y_AXIS = f"$D$"
    FIGURE_TITLE = f"$D$ vs Number of generations"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_population_diversity_against_generation_index_0_480.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(df_neat_ncl["generation_index"].unique())

    # Plot lines
    # NEAT NCL - ensemble_size = 50
    population_diversity_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["population_diversity_test"])
    assert len(population_diversity_neat_ncl_50) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): population_diversity_neat_ncl_50 has length {len(population_diversity_neat_ncl_50)}"
    axis.plot(data_x, population_diversity_neat_ncl_50,
              marker="o", label=f"NEAT NCL - ensemble_size: 50")

    # NEAT NCL - ensemble_size = 100
    population_diversity_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["population_diversity_test"])
    assert len(population_diversity_neat_ncl_100) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): population_diversity_neat_ncl_100 has length {len(population_diversity_neat_ncl_100)}"
    axis.plot(data_x, population_diversity_neat_ncl_100,
              marker="o", label=f"NEAT NCL - ensemble_size: 100")

    # NEAT NCL - ensemble_size = 150
    population_diversity_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["population_diversity_test"])
    assert len(population_diversity_neat_ncl_150) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): population_diversity_neat_ncl_150 has length {len(population_diversity_neat_ncl_150)}"
    axis.plot(data_x, population_diversity_neat_ncl_150,
              marker="o", label=f"NEAT NCL - ensemble_size: 150")

    # NEAT NCL - ensemble_size = 200
    population_diversity_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["population_diversity_test"])
    assert len(population_diversity_neat_ncl_200) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): population_diversity_neat_ncl_200 has length {len(population_diversity_neat_ncl_200)}"
    axis.plot(data_x, population_diversity_neat_ncl_200,
              marker="o", label=f"NEAT NCL - ensemble_size: 200")

    # NEAT NCL - ensemble_size = 250
    population_diversity_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["population_diversity_test"])
    assert len(population_diversity_neat_ncl_250) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): population_diversity_neat_ncl_250 has length {len(population_diversity_neat_ncl_250)}"
    axis.plot(data_x, population_diversity_neat_ncl_250,
              marker="o", label=f"NEAT NCL - ensemble_size: 250")

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


def max_population_diversity_against_generation_index_0_480(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            50,
            100,
            150,
            200,
            250
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"Number of generations"
    Y_AXIS = f"$D_{{max}}$"
    FIGURE_TITLE = f"$D_{{max}}$ vs Number of generations"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_max_population_diversity_against_generation_index_0_480.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(df_neat_ncl["generation_index"].unique())

    # Plot lines
    # NEAT NCL - ensemble_size = 50
    population_diversity_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["max_population_diversity_test"])
    assert len(population_diversity_neat_ncl_50) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): population_diversity_neat_ncl_50 has length {len(population_diversity_neat_ncl_50)}"
    axis.plot(data_x, population_diversity_neat_ncl_50,
              marker="o", label=f"NEAT NCL - ensemble_size: 50")

    # NEAT NCL - ensemble_size = 100
    population_diversity_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["max_population_diversity_test"])
    assert len(population_diversity_neat_ncl_100) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): population_diversity_neat_ncl_100 has length {len(population_diversity_neat_ncl_100)}"
    axis.plot(data_x, population_diversity_neat_ncl_100,
              marker="o", label=f"NEAT NCL - ensemble_size: 100")

    # NEAT NCL - ensemble_size = 150
    population_diversity_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["max_population_diversity_test"])
    assert len(population_diversity_neat_ncl_150) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): population_diversity_neat_ncl_150 has length {len(population_diversity_neat_ncl_150)}"
    axis.plot(data_x, population_diversity_neat_ncl_150,
              marker="o", label=f"NEAT NCL - ensemble_size: 150")

    # NEAT NCL - ensemble_size = 200
    population_diversity_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["max_population_diversity_test"])
    assert len(population_diversity_neat_ncl_200) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): population_diversity_neat_ncl_200 has length {len(population_diversity_neat_ncl_200)}"
    axis.plot(data_x, population_diversity_neat_ncl_200,
              marker="o", label=f"NEAT NCL - ensemble_size: 200")

    # NEAT NCL - ensemble_size = 250
    population_diversity_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["max_population_diversity_test"])
    assert len(population_diversity_neat_ncl_250) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): population_diversity_neat_ncl_250 has length {len(population_diversity_neat_ncl_250)}"
    axis.plot(data_x, population_diversity_neat_ncl_250,
              marker="o", label=f"NEAT NCL - ensemble_size: 250")

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


def population_diversity_ratio_against_generation_index_0_480(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            50,
            100,
            150,
            200,
            250
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"Number of generations"
    Y_AXIS = f"$\\frac{{D}}{{D_{{max}}}}$"
    FIGURE_TITLE = f"$\\frac{{D}}{{D_{{max}}}}$ vs Number of generations"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_population_diversity_ratio_against_generation_index_0_480.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(df_neat_ncl["generation_index"].unique())

    # Plot lines
    # NEAT NCL - ensemble_size = 50
    numerator_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["population_diversity_test"])
    denominator_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["max_population_diversity_test"])
    population_diversity_ratio_neat_ncl_50 = np.divide(
        numerator_neat_ncl_50,
        denominator_neat_ncl_50,
        # NOTE: We assume D / D_{max} = 0 when D_{max} = 0 (this decision is made based on analysing the graph of correlation_penalty_coefficient against population_diversity_ratio
        out=np.zeros_like(numerator_neat_ncl_50, dtype=float),
        where=denominator_neat_ncl_50 != 0
    )
    assert len(population_diversity_ratio_neat_ncl_50) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): population_diversity_ratio_neat_ncl_50 has length {len(population_diversity_ratio_neat_ncl_50)}"
    axis.plot(data_x, population_diversity_ratio_neat_ncl_50,
              marker="o", label=f"NEAT NCL - ensemble_size: 50")

    # NEAT NCL - ensemble_size = 100
    numerator_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["population_diversity_test"])
    denominator_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["max_population_diversity_test"])
    population_diversity_ratio_neat_ncl_100 = np.divide(
        numerator_neat_ncl_100,
        denominator_neat_ncl_100,
        # NOTE: We assume D / D_{max} = 0 when D_{max} = 0 (this decision is made based on analysing the graph of correlation_penalty_coefficient against population_diversity_ratio
        out=np.zeros_like(numerator_neat_ncl_100, dtype=float),
        where=denominator_neat_ncl_100 != 0
    )
    assert len(population_diversity_ratio_neat_ncl_100) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): population_diversity_ratio_neat_ncl_100 has length {len(population_diversity_ratio_neat_ncl_100)}"
    axis.plot(data_x, population_diversity_ratio_neat_ncl_100,
              marker="o", label=f"NEAT NCL - ensemble_size: 100")

    # NEAT NCL - ensemble_size = 150
    numerator_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["population_diversity_test"])
    denominator_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["max_population_diversity_test"])
    population_diversity_ratio_neat_ncl_150 = np.divide(
        numerator_neat_ncl_150,
        denominator_neat_ncl_150,
        # NOTE: We assume D / D_{max} = 0 when D_{max} = 0 (this decision is made based on analysing the graph of correlation_penalty_coefficient against population_diversity_ratio
        out=np.zeros_like(numerator_neat_ncl_150, dtype=float),
        where=denominator_neat_ncl_150 != 0
    )
    assert len(population_diversity_ratio_neat_ncl_150) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): population_diversity_ratio_neat_ncl_150 has length {len(population_diversity_ratio_neat_ncl_150)}"
    axis.plot(data_x, population_diversity_ratio_neat_ncl_150,
              marker="o", label=f"NEAT NCL - ensemble_size: 150")

    # NEAT NCL - ensemble_size = 200
    numerator_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["population_diversity_test"])
    denominator_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["max_population_diversity_test"])
    population_diversity_ratio_neat_ncl_200 = np.divide(
        numerator_neat_ncl_200,
        denominator_neat_ncl_200,
        # NOTE: We assume D / D_{max} = 0 when D_{max} = 0 (this decision is made based on analysing the graph of correlation_penalty_coefficient against population_diversity_ratio
        out=np.zeros_like(numerator_neat_ncl_200, dtype=float),
        where=denominator_neat_ncl_200 != 0
    )
    assert len(population_diversity_ratio_neat_ncl_200) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): population_diversity_ratio_neat_ncl_200 has length {len(population_diversity_ratio_neat_ncl_200)}"
    axis.plot(data_x, population_diversity_ratio_neat_ncl_200,
              marker="o", label=f"NEAT NCL - ensemble_size: 200")

    # NEAT NCL - ensemble_size = 250
    numerator_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["population_diversity_test"])
    denominator_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["max_population_diversity_test"])
    population_diversity_ratio_neat_ncl_250 = np.divide(
        numerator_neat_ncl_250,
        denominator_neat_ncl_250,
        # NOTE: We assume D / D_{max} = 0 when D_{max} = 0 (this decision is made based on analysing the graph of correlation_penalty_coefficient against population_diversity_ratio
        out=np.zeros_like(numerator_neat_ncl_250, dtype=float),
        where=denominator_neat_ncl_250 != 0
    )
    assert len(population_diversity_ratio_neat_ncl_250) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): population_diversity_ratio_neat_ncl_250 has length {len(population_diversity_ratio_neat_ncl_250)}"
    axis.plot(data_x, population_diversity_ratio_neat_ncl_250,
              marker="o", label=f"NEAT NCL - ensemble_size: 250")

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


def average_niche_radius_against_generation_index_0_480(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            50,
            100,
            150,
            200,
            250
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"Number of generations"
    Y_AXIS = f"$\\sigma$"
    FIGURE_TITLE = f"$\\sigma$ vs Number of generations"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_average_niche_radius_against_generation_index_0_480.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(df_neat_ncl["generation_index"].unique())

    # Plot lines
    # NEAT NCL - ensemble_size = 50
    average_niche_radius_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["average_niche_radius"])
    assert len(average_niche_radius_neat_ncl_50) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): average_niche_radius_neat_ncl_50 has length {len(average_niche_radius_neat_ncl_50)}"
    axis.plot(data_x, average_niche_radius_neat_ncl_50,
              marker="o", label=f"NEAT NCL - ensemble_size: 50")

    # NEAT NCL - ensemble_size = 100
    average_niche_radius_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["average_niche_radius"])
    assert len(average_niche_radius_neat_ncl_100) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): average_niche_radius_neat_ncl_100 has length {len(average_niche_radius_neat_ncl_100)}"
    axis.plot(data_x, average_niche_radius_neat_ncl_100,
              marker="o", label=f"NEAT NCL - ensemble_size: 100")

    # NEAT NCL - ensemble_size = 150
    average_niche_radius_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["average_niche_radius"])
    assert len(average_niche_radius_neat_ncl_150) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): average_niche_radius_neat_ncl_150 has length {len(average_niche_radius_neat_ncl_150)}"
    axis.plot(data_x, average_niche_radius_neat_ncl_150,
              marker="o", label=f"NEAT NCL - ensemble_size: 150")

    # NEAT NCL - ensemble_size = 200
    average_niche_radius_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["average_niche_radius"])
    assert len(average_niche_radius_neat_ncl_200) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): average_niche_radius_neat_ncl_200 has length {len(average_niche_radius_neat_ncl_200)}"
    axis.plot(data_x, average_niche_radius_neat_ncl_200,
              marker="o", label=f"NEAT NCL - ensemble_size: 200")

    # NEAT NCL - ensemble_size = 250
    average_niche_radius_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["average_niche_radius"])
    assert len(average_niche_radius_neat_ncl_250) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): average_niche_radius_neat_ncl_250 has length {len(average_niche_radius_neat_ncl_250)}"
    axis.plot(data_x, average_niche_radius_neat_ncl_250,
              marker="o", label=f"NEAT NCL - ensemble_size: 250")

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


def population_diversity_against_average_niche_radius(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            50,
            100,
            150,
            200,
            250
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\sigma$"
    Y_AXIS = f"$D$"
    FIGURE_TITLE = f"$D$ vs $\\sigma$"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_population_diversity_against_average_niche_radius.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # Plot lines
    # NEAT NCL - ensemble_size = 50
    population_diversity_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["population_diversity_test"])
    average_niche_radius_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["average_niche_radius"])
    assert len(population_diversity_neat_ncl_50) == len(
        average_niche_radius_neat_ncl_50), f"Invalid number of data points (expect {len(average_niche_radius_neat_ncl_50)}): population_diversity_neat_ncl_50 has length {len(population_diversity_neat_ncl_50)}"
    axis.scatter(average_niche_radius_neat_ncl_50, population_diversity_neat_ncl_50,
                 marker="o", label=f"NEAT NCL - ensemble_size: 50")

    # NEAT NCL - ensemble_size = 100
    population_diversity_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["population_diversity_test"])
    average_niche_radius_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["average_niche_radius"])
    assert len(population_diversity_neat_ncl_100) == len(
        average_niche_radius_neat_ncl_100), f"Invalid number of data points (expect {len(average_niche_radius_neat_ncl_100)}): population_diversity_neat_ncl_100 has length {len(population_diversity_neat_ncl_100)}"
    axis.scatter(average_niche_radius_neat_ncl_100, population_diversity_neat_ncl_100,
                 marker="o", label=f"NEAT NCL - ensemble_size: 100")

    # NEAT NCL - ensemble_size = 150
    population_diversity_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["population_diversity_test"])
    average_niche_radius_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["average_niche_radius"])
    assert len(population_diversity_neat_ncl_150) == len(
        average_niche_radius_neat_ncl_150), f"Invalid number of data points (expect {len(average_niche_radius_neat_ncl_150)}): population_diversity_neat_ncl_150 has length {len(population_diversity_neat_ncl_150)}"
    axis.scatter(average_niche_radius_neat_ncl_150, population_diversity_neat_ncl_150,
                 marker="o", label=f"NEAT NCL - ensemble_size: 150")

    # NEAT NCL - ensemble_size = 200
    population_diversity_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["population_diversity_test"])
    average_niche_radius_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["average_niche_radius"])
    assert len(population_diversity_neat_ncl_200) == len(
        average_niche_radius_neat_ncl_200), f"Invalid number of data points (expect {len(average_niche_radius_neat_ncl_200)}): population_diversity_neat_ncl_200 has length {len(population_diversity_neat_ncl_200)}"
    axis.scatter(average_niche_radius_neat_ncl_200, population_diversity_neat_ncl_200,
                 marker="o", label=f"NEAT NCL - ensemble_size: 200")

    # NEAT NCL - ensemble_size = 250
    population_diversity_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["population_diversity_test"])
    average_niche_radius_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["average_niche_radius"])
    assert len(population_diversity_neat_ncl_250) == len(
        average_niche_radius_neat_ncl_250), f"Invalid number of data points (expect {len(average_niche_radius_neat_ncl_250)}): population_diversity_neat_ncl_250 has length {len(population_diversity_neat_ncl_250)}"
    axis.scatter(average_niche_radius_neat_ncl_250, population_diversity_neat_ncl_250,
                 marker="o", label=f"NEAT NCL - ensemble_size: 250")

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


def average_sharing_factor_against_average_niche_radius(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            50,
            100,
            150,
            200,
            250
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\sigma$"
    Y_AXIS = f"Sharing factor"
    FIGURE_TITLE = f"Sharing factor vs $\\sigma$"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_average_sharing_factor_against_average_niche_radius.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # Plot lines
    # NEAT NCL - ensemble_size = 50
    average_sharing_factor_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["average_sharing_factor"])
    average_niche_radius_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["average_niche_radius"])
    assert len(average_sharing_factor_neat_ncl_50) == len(
        average_niche_radius_neat_ncl_50), f"Invalid number of data points (expect {len(average_niche_radius_neat_ncl_50)}): average_sharing_factor_neat_ncl_50 has length {len(average_sharing_factor_neat_ncl_50)}"
    axis.scatter(average_niche_radius_neat_ncl_50, average_sharing_factor_neat_ncl_50,
                 marker="o", label=f"NEAT NCL - ensemble_size: 50")

    # NEAT NCL - ensemble_size = 100
    average_sharing_factor_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["average_sharing_factor"])
    average_niche_radius_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["average_niche_radius"])
    assert len(average_sharing_factor_neat_ncl_100) == len(
        average_niche_radius_neat_ncl_100), f"Invalid number of data points (expect {len(average_niche_radius_neat_ncl_100)}): average_sharing_factor_neat_ncl_100 has length {len(average_sharing_factor_neat_ncl_100)}"
    axis.scatter(average_niche_radius_neat_ncl_100, average_sharing_factor_neat_ncl_100,
                 marker="o", label=f"NEAT NCL - ensemble_size: 100")

    # NEAT NCL - ensemble_size = 150
    average_sharing_factor_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["average_sharing_factor"])
    average_niche_radius_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["average_niche_radius"])
    assert len(average_sharing_factor_neat_ncl_150) == len(
        average_niche_radius_neat_ncl_150), f"Invalid number of data points (expect {len(average_niche_radius_neat_ncl_150)}): average_sharing_factor_neat_ncl_150 has length {len(average_sharing_factor_neat_ncl_150)}"
    axis.scatter(average_niche_radius_neat_ncl_150, average_sharing_factor_neat_ncl_150,
                 marker="o", label=f"NEAT NCL - ensemble_size: 150")

    # NEAT NCL - ensemble_size = 200
    average_sharing_factor_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["average_sharing_factor"])
    average_niche_radius_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["average_niche_radius"])
    assert len(average_sharing_factor_neat_ncl_200) == len(
        average_niche_radius_neat_ncl_200), f"Invalid number of data points (expect {len(average_niche_radius_neat_ncl_200)}): average_sharing_factor_neat_ncl_200 has length {len(average_sharing_factor_neat_ncl_200)}"
    axis.scatter(average_niche_radius_neat_ncl_200, average_sharing_factor_neat_ncl_200,
                 marker="o", label=f"NEAT NCL - ensemble_size: 200")

    # NEAT NCL - ensemble_size = 250
    average_sharing_factor_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["average_sharing_factor"])
    average_niche_radius_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["average_niche_radius"])
    assert len(average_sharing_factor_neat_ncl_250) == len(
        average_niche_radius_neat_ncl_250), f"Invalid number of data points (expect {len(average_niche_radius_neat_ncl_250)}): average_sharing_factor_neat_ncl_250 has length {len(average_sharing_factor_neat_ncl_250)}"
    axis.scatter(average_niche_radius_neat_ncl_250, average_sharing_factor_neat_ncl_250,
                 marker="o", label=f"NEAT NCL - ensemble_size: 250")

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


def average_adjusted_fitness_against_average_niche_radius(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            50,
            100,
            150,
            200,
            250
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"$\\sigma$"
    Y_AXIS = f"Adjusted fitness"
    FIGURE_TITLE = f"Adjusted fitness vs $\\sigma$"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_average_adjusted_fitness_against_average_niche_radius.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # Plot lines
    # NEAT NCL - ensemble_size = 50
    average_adjusted_fitness_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["average_adjusted_fitness"])
    average_niche_radius_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["average_niche_radius"])
    assert len(average_adjusted_fitness_neat_ncl_50) == len(
        average_niche_radius_neat_ncl_50), f"Invalid number of data points (expect {len(average_niche_radius_neat_ncl_50)}): average_adjusted_fitness_neat_ncl_50 has length {len(average_adjusted_fitness_neat_ncl_50)}"
    axis.scatter(average_niche_radius_neat_ncl_50, average_adjusted_fitness_neat_ncl_50,
                 marker="o", label=f"NEAT NCL - ensemble_size: 50")

    # NEAT NCL - ensemble_size = 100
    average_adjusted_fitness_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["average_adjusted_fitness"])
    average_niche_radius_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["average_niche_radius"])
    assert len(average_adjusted_fitness_neat_ncl_100) == len(
        average_niche_radius_neat_ncl_100), f"Invalid number of data points (expect {len(average_niche_radius_neat_ncl_100)}): average_adjusted_fitness_neat_ncl_100 has length {len(average_adjusted_fitness_neat_ncl_100)}"
    axis.scatter(average_niche_radius_neat_ncl_100, average_adjusted_fitness_neat_ncl_100,
                 marker="o", label=f"NEAT NCL - ensemble_size: 100")

    # NEAT NCL - ensemble_size = 150
    average_adjusted_fitness_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["average_adjusted_fitness"])
    average_niche_radius_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["average_niche_radius"])
    assert len(average_adjusted_fitness_neat_ncl_150) == len(
        average_niche_radius_neat_ncl_150), f"Invalid number of data points (expect {len(average_niche_radius_neat_ncl_150)}): average_adjusted_fitness_neat_ncl_150 has length {len(average_adjusted_fitness_neat_ncl_150)}"
    axis.scatter(average_niche_radius_neat_ncl_150, average_adjusted_fitness_neat_ncl_150,
                 marker="o", label=f"NEAT NCL - ensemble_size: 150")

    # NEAT NCL - ensemble_size = 200
    average_adjusted_fitness_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["average_adjusted_fitness"])
    average_niche_radius_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["average_niche_radius"])
    assert len(average_adjusted_fitness_neat_ncl_200) == len(
        average_niche_radius_neat_ncl_200), f"Invalid number of data points (expect {len(average_niche_radius_neat_ncl_200)}): average_adjusted_fitness_neat_ncl_200 has length {len(average_adjusted_fitness_neat_ncl_200)}"
    axis.scatter(average_niche_radius_neat_ncl_200, average_adjusted_fitness_neat_ncl_200,
                 marker="o", label=f"NEAT NCL - ensemble_size: 200")

    # NEAT NCL - ensemble_size = 250
    average_adjusted_fitness_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["average_adjusted_fitness"])
    average_niche_radius_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["average_niche_radius"])
    assert len(average_adjusted_fitness_neat_ncl_250) == len(
        average_niche_radius_neat_ncl_250), f"Invalid number of data points (expect {len(average_niche_radius_neat_ncl_250)}): average_adjusted_fitness_neat_ncl_250 has length {len(average_adjusted_fitness_neat_ncl_250)}"
    axis.scatter(average_niche_radius_neat_ncl_250, average_adjusted_fitness_neat_ncl_250,
                 marker="o", label=f"NEAT NCL - ensemble_size: 250")

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


def average_sharing_factor_against_generation_index_0_480(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            50,
            100,
            150,
            200,
            250
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"Number of generations"
    Y_AXIS = f"Sharing factor"
    FIGURE_TITLE = f"Sharing factor vs Number of generations"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_average_sharing_factor_against_generation_index_0_480.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(df_neat_ncl["generation_index"].unique())

    # Plot lines
    # NEAT NCL - ensemble_size = 50
    average_sharing_factor_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["average_sharing_factor"])
    assert len(average_sharing_factor_neat_ncl_50) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): average_sharing_factor_neat_ncl_50 has length {len(average_sharing_factor_neat_ncl_50)}"
    axis.plot(data_x, average_sharing_factor_neat_ncl_50,
              marker="o", label=f"NEAT NCL - ensemble_size: 50")

    # NEAT NCL - ensemble_size = 100
    average_sharing_factor_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["average_sharing_factor"])
    assert len(average_sharing_factor_neat_ncl_100) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): average_sharing_factor_neat_ncl_100 has length {len(average_sharing_factor_neat_ncl_100)}"
    axis.plot(data_x, average_sharing_factor_neat_ncl_100,
              marker="o", label=f"NEAT NCL - ensemble_size: 100")

    # NEAT NCL - ensemble_size = 150
    average_sharing_factor_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["average_sharing_factor"])
    assert len(average_sharing_factor_neat_ncl_150) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): average_sharing_factor_neat_ncl_150 has length {len(average_sharing_factor_neat_ncl_150)}"
    axis.plot(data_x, average_sharing_factor_neat_ncl_150,
              marker="o", label=f"NEAT NCL - ensemble_size: 150")

    # NEAT NCL - ensemble_size = 200
    average_sharing_factor_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["average_sharing_factor"])
    assert len(average_sharing_factor_neat_ncl_200) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): average_sharing_factor_neat_ncl_200 has length {len(average_sharing_factor_neat_ncl_200)}"
    axis.plot(data_x, average_sharing_factor_neat_ncl_200,
              marker="o", label=f"NEAT NCL - ensemble_size: 200")

    # NEAT NCL - ensemble_size = 250
    average_sharing_factor_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["average_sharing_factor"])
    assert len(average_sharing_factor_neat_ncl_250) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): average_sharing_factor_neat_ncl_250 has length {len(average_sharing_factor_neat_ncl_250)}"
    axis.plot(data_x, average_sharing_factor_neat_ncl_250,
              marker="o", label=f"NEAT NCL - ensemble_size: 250")

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


def population_diversity_against_average_sharing_factor(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            50,
            100,
            150,
            200,
            250
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"Sharing factor"
    Y_AXIS = f"$D$"
    FIGURE_TITLE = f"$D$ vs Sharing factor"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_population_diversity_against_average_sharing_factor.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # Plot lines
    # NEAT NCL - ensemble_size = 50
    population_diversity_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["population_diversity_test"])
    average_sharing_factor_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["average_sharing_factor"])
    assert len(population_diversity_neat_ncl_50) == len(
        average_sharing_factor_neat_ncl_50), f"Invalid number of data points (expect {len(average_sharing_factor_neat_ncl_50)}): average_sharing_factor_neat_ncl_50 has length {len(population_diversity_neat_ncl_50)}"
    axis.scatter(average_sharing_factor_neat_ncl_50, population_diversity_neat_ncl_50,
                 marker="o", label=f"NEAT NCL - ensemble_size: 50")

    # NEAT NCL - ensemble_size = 100
    population_diversity_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["population_diversity_test"])
    average_sharing_factor_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["average_sharing_factor"])
    assert len(population_diversity_neat_ncl_100) == len(
        average_sharing_factor_neat_ncl_100), f"Invalid number of data points (expect {len(average_sharing_factor_neat_ncl_100)}): average_sharing_factor_neat_ncl_100 has length {len(population_diversity_neat_ncl_100)}"
    axis.scatter(average_sharing_factor_neat_ncl_100, population_diversity_neat_ncl_100,
                 marker="o", label=f"NEAT NCL - ensemble_size: 100")

    # NEAT NCL - ensemble_size = 150
    population_diversity_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["population_diversity_test"])
    average_sharing_factor_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["average_sharing_factor"])
    assert len(population_diversity_neat_ncl_150) == len(
        average_sharing_factor_neat_ncl_150), f"Invalid number of data points (expect {len(average_sharing_factor_neat_ncl_150)}): average_sharing_factor_neat_ncl_150 has length {len(population_diversity_neat_ncl_150)}"
    axis.scatter(average_sharing_factor_neat_ncl_150, population_diversity_neat_ncl_150,
                 marker="o", label=f"NEAT NCL - ensemble_size: 150")

    # NEAT NCL - ensemble_size = 200
    population_diversity_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["population_diversity_test"])
    average_sharing_factor_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["average_sharing_factor"])
    assert len(population_diversity_neat_ncl_200) == len(
        average_sharing_factor_neat_ncl_200), f"Invalid number of data points (expect {len(average_sharing_factor_neat_ncl_200)}): average_sharing_factor_neat_ncl_200 has length {len(population_diversity_neat_ncl_200)}"
    axis.scatter(average_sharing_factor_neat_ncl_200, population_diversity_neat_ncl_200,
                 marker="o", label=f"NEAT NCL - ensemble_size: 200")

    # NEAT NCL - ensemble_size = 250
    population_diversity_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["population_diversity_test"])
    average_sharing_factor_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["average_sharing_factor"])
    assert len(population_diversity_neat_ncl_250) == len(
        average_sharing_factor_neat_ncl_250), f"Invalid number of data points (expect {len(average_sharing_factor_neat_ncl_250)}): average_sharing_factor_neat_ncl_250 has length {len(population_diversity_neat_ncl_250)}"
    axis.scatter(average_sharing_factor_neat_ncl_250, population_diversity_neat_ncl_250,
                 marker="o", label=f"NEAT NCL - ensemble_size: 250")

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


def average_adjusted_fitness_against_average_sharing_factor(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            50,
            100,
            150,
            200,
            250
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"Sharing factor"
    Y_AXIS = f"Adjusted fitness"
    FIGURE_TITLE = f"Adjusted fitness vs Sharing factor"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_average_adjusted_fitness_against_average_sharing_factor.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # Plot lines
    # NEAT NCL - ensemble_size = 50
    average_adjusted_fitness_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["average_adjusted_fitness"])
    average_sharing_factor_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["average_sharing_factor"])
    assert len(average_adjusted_fitness_neat_ncl_50) == len(
        average_sharing_factor_neat_ncl_50), f"Invalid number of data points (expect {len(average_sharing_factor_neat_ncl_50)}): average_sharing_factor_neat_ncl_50 has length {len(average_adjusted_fitness_neat_ncl_50)}"
    axis.scatter(average_sharing_factor_neat_ncl_50, average_adjusted_fitness_neat_ncl_50,
                 marker="o", label=f"NEAT NCL - ensemble_size: 50")

    # NEAT NCL - ensemble_size = 100
    average_adjusted_fitness_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["average_adjusted_fitness"])
    average_sharing_factor_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["average_sharing_factor"])
    assert len(average_adjusted_fitness_neat_ncl_100) == len(
        average_sharing_factor_neat_ncl_100), f"Invalid number of data points (expect {len(average_sharing_factor_neat_ncl_100)}): average_sharing_factor_neat_ncl_100 has length {len(average_adjusted_fitness_neat_ncl_100)}"
    axis.scatter(average_sharing_factor_neat_ncl_100, average_adjusted_fitness_neat_ncl_100,
                 marker="o", label=f"NEAT NCL - ensemble_size: 100")

    # NEAT NCL - ensemble_size = 150
    average_adjusted_fitness_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["average_adjusted_fitness"])
    average_sharing_factor_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["average_sharing_factor"])
    assert len(average_adjusted_fitness_neat_ncl_150) == len(
        average_sharing_factor_neat_ncl_150), f"Invalid number of data points (expect {len(average_sharing_factor_neat_ncl_150)}): average_sharing_factor_neat_ncl_150 has length {len(average_adjusted_fitness_neat_ncl_150)}"
    axis.scatter(average_sharing_factor_neat_ncl_150, average_adjusted_fitness_neat_ncl_150,
                 marker="o", label=f"NEAT NCL - ensemble_size: 150")

    # NEAT NCL - ensemble_size = 200
    average_adjusted_fitness_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["average_adjusted_fitness"])
    average_sharing_factor_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["average_sharing_factor"])
    assert len(average_adjusted_fitness_neat_ncl_200) == len(
        average_sharing_factor_neat_ncl_200), f"Invalid number of data points (expect {len(average_sharing_factor_neat_ncl_200)}): average_sharing_factor_neat_ncl_200 has length {len(average_adjusted_fitness_neat_ncl_200)}"
    axis.scatter(average_sharing_factor_neat_ncl_200, average_adjusted_fitness_neat_ncl_200,
                 marker="o", label=f"NEAT NCL - ensemble_size: 200")

    # NEAT NCL - ensemble_size = 250
    average_adjusted_fitness_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["average_adjusted_fitness"])
    average_sharing_factor_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["average_sharing_factor"])
    assert len(average_adjusted_fitness_neat_ncl_250) == len(
        average_sharing_factor_neat_ncl_250), f"Invalid number of data points (expect {len(average_sharing_factor_neat_ncl_250)}): average_sharing_factor_neat_ncl_250 has length {len(average_adjusted_fitness_neat_ncl_250)}"
    axis.scatter(average_sharing_factor_neat_ncl_250, average_adjusted_fitness_neat_ncl_250,
                 marker="o", label=f"NEAT NCL - ensemble_size: 250")

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


def average_raw_fitness_against_generation_index_0_480(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            50,
            100,
            150,
            200,
            250
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"Number of generations"
    Y_AXIS = f"Raw fitness"
    FIGURE_TITLE = f"Raw fitness vs Number of generations"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_average_raw_fitness_against_generation_index_0_480.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(df_neat_ncl["generation_index"].unique())

    # Plot lines
    # NEAT NCL - ensemble_size = 50
    average_raw_fitness_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["average_raw_fitness"])
    assert len(average_raw_fitness_neat_ncl_50) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): average_raw_fitness_neat_ncl_50 has length {len(average_raw_fitness_neat_ncl_50)}"
    axis.plot(data_x, average_raw_fitness_neat_ncl_50,
              marker="o", label=f"NEAT NCL - ensemble_size: 50")

    # NEAT NCL - ensemble_size = 100
    average_raw_fitness_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["average_raw_fitness"])
    assert len(average_raw_fitness_neat_ncl_100) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): average_raw_fitness_neat_ncl_100 has length {len(average_raw_fitness_neat_ncl_100)}"
    axis.plot(data_x, average_raw_fitness_neat_ncl_100,
              marker="o", label=f"NEAT NCL - ensemble_size: 100")

    # NEAT NCL - ensemble_size = 150
    average_raw_fitness_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["average_raw_fitness"])
    assert len(average_raw_fitness_neat_ncl_150) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): average_raw_fitness_neat_ncl_150 has length {len(average_raw_fitness_neat_ncl_150)}"
    axis.plot(data_x, average_raw_fitness_neat_ncl_150,
              marker="o", label=f"NEAT NCL - ensemble_size: 150")

    # NEAT NCL - ensemble_size = 200
    average_raw_fitness_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["average_raw_fitness"])
    assert len(average_raw_fitness_neat_ncl_200) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): average_raw_fitness_neat_ncl_200 has length {len(average_raw_fitness_neat_ncl_200)}"
    axis.plot(data_x, average_raw_fitness_neat_ncl_200,
              marker="o", label=f"NEAT NCL - ensemble_size: 200")

    # NEAT NCL - ensemble_size = 250
    average_raw_fitness_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["average_raw_fitness"])
    assert len(average_raw_fitness_neat_ncl_250) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): average_raw_fitness_neat_ncl_250 has length {len(average_raw_fitness_neat_ncl_250)}"
    axis.plot(data_x, average_raw_fitness_neat_ncl_250,
              marker="o", label=f"NEAT NCL - ensemble_size: 250")

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


def average_raw_fitness_against_average_active_hidden_node_num(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            50,
            100,
            150,
            200,
            250
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"Number of active hidden nodes"
    Y_AXIS = f"Raw fitness"
    FIGURE_TITLE = f"Raw fitness vs Number of active hidden nodes"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_average_raw_fitness_against_average_active_hidden_node_num.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # Plot lines
    # NEAT NCL - ensemble_size = 50
    average_raw_fitness_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["average_raw_fitness"])
    average_active_hidden_node_num_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["average_active_hidden_node_num"])
    assert len(average_raw_fitness_neat_ncl_50) == len(
        average_active_hidden_node_num_neat_ncl_50), f"Invalid number of data points (expect {len(average_active_hidden_node_num_neat_ncl_50)}): average_raw_fitness_neat_ncl_50 has length {len(average_raw_fitness_neat_ncl_50)}"
    axis.scatter(average_active_hidden_node_num_neat_ncl_50, average_raw_fitness_neat_ncl_50,
                 marker="o", label=f"NEAT NCL - ensemble_size: 50")

    # NEAT NCL - ensemble_size = 100
    average_raw_fitness_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["average_raw_fitness"])
    average_active_hidden_node_num_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["average_active_hidden_node_num"])
    assert len(average_raw_fitness_neat_ncl_100) == len(
        average_active_hidden_node_num_neat_ncl_100), f"Invalid number of data points (expect {len(average_active_hidden_node_num_neat_ncl_100)}): average_raw_fitness_neat_ncl_100 has length {len(average_raw_fitness_neat_ncl_100)}"
    axis.scatter(average_active_hidden_node_num_neat_ncl_100, average_raw_fitness_neat_ncl_100,
                 marker="o", label=f"NEAT NCL - ensemble_size: 100")

    # NEAT NCL - ensemble_size = 150
    average_raw_fitness_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["average_raw_fitness"])
    average_active_hidden_node_num_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["average_active_hidden_node_num"])
    assert len(average_raw_fitness_neat_ncl_150) == len(
        average_active_hidden_node_num_neat_ncl_150), f"Invalid number of data points (expect {len(average_active_hidden_node_num_neat_ncl_150)}): average_raw_fitness_neat_ncl_150 has length {len(average_raw_fitness_neat_ncl_150)}"
    axis.scatter(average_active_hidden_node_num_neat_ncl_150, average_raw_fitness_neat_ncl_150,
                 marker="o", label=f"NEAT NCL - ensemble_size: 150")

    # NEAT NCL - ensemble_size = 200
    average_raw_fitness_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["average_raw_fitness"])
    average_active_hidden_node_num_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["average_active_hidden_node_num"])
    assert len(average_raw_fitness_neat_ncl_200) == len(
        average_active_hidden_node_num_neat_ncl_200), f"Invalid number of data points (expect {len(average_active_hidden_node_num_neat_ncl_200)}): average_raw_fitness_neat_ncl_200 has length {len(average_raw_fitness_neat_ncl_200)}"
    axis.scatter(average_active_hidden_node_num_neat_ncl_200, average_raw_fitness_neat_ncl_200,
                 marker="o", label=f"NEAT NCL - ensemble_size: 200")

    # NEAT NCL - ensemble_size = 250
    average_raw_fitness_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["average_raw_fitness"])
    average_active_hidden_node_num_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["average_active_hidden_node_num"])
    assert len(average_raw_fitness_neat_ncl_250) == len(
        average_active_hidden_node_num_neat_ncl_250), f"Invalid number of data points (expect {len(average_active_hidden_node_num_neat_ncl_250)}): average_raw_fitness_neat_ncl_250 has length {len(average_raw_fitness_neat_ncl_250)}"
    axis.scatter(average_active_hidden_node_num_neat_ncl_250, average_raw_fitness_neat_ncl_250,
                 marker="o", label=f"NEAT NCL - ensemble_size: 250")

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


def average_adjusted_fitness_against_generation_index_0_480(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            50,
            100,
            150,
            200,
            250
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"Number of generations"
    Y_AXIS = f"Adjusted fitness"
    FIGURE_TITLE = f"Adjusted fitness vs Number of generations"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_average_adjusted_fitness_against_generation_index_0_480.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(df_neat_ncl["generation_index"].unique())

    # Plot lines
    # NEAT NCL - ensemble_size = 50
    average_adjusted_fitness_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["average_adjusted_fitness"])
    assert len(average_adjusted_fitness_neat_ncl_50) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): average_adjusted_fitness_neat_ncl_50 has length {len(average_adjusted_fitness_neat_ncl_50)}"
    axis.plot(data_x, average_adjusted_fitness_neat_ncl_50,
              marker="o", label=f"NEAT NCL - ensemble_size: 50")

    # NEAT NCL - ensemble_size = 100
    average_adjusted_fitness_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["average_adjusted_fitness"])
    assert len(average_adjusted_fitness_neat_ncl_100) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): average_adjusted_fitness_neat_ncl_100 has length {len(average_adjusted_fitness_neat_ncl_100)}"
    axis.plot(data_x, average_adjusted_fitness_neat_ncl_100,
              marker="o", label=f"NEAT NCL - ensemble_size: 100")

    # NEAT NCL - ensemble_size = 150
    average_adjusted_fitness_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["average_raw_fitness"])
    assert len(average_adjusted_fitness_neat_ncl_150) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): average_adjusted_fitness_neat_ncl_150 has length {len(average_adjusted_fitness_neat_ncl_150)}"
    axis.plot(data_x, average_adjusted_fitness_neat_ncl_150,
              marker="o", label=f"NEAT NCL - ensemble_size: 150")

    # NEAT NCL - ensemble_size = 200
    average_adjusted_fitness_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["average_adjusted_fitness"])
    assert len(average_adjusted_fitness_neat_ncl_200) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): average_adjusted_fitness_neat_ncl_200 has length {len(average_adjusted_fitness_neat_ncl_200)}"
    axis.plot(data_x, average_adjusted_fitness_neat_ncl_200,
              marker="o", label=f"NEAT NCL - ensemble_size: 200")

    # NEAT NCL - ensemble_size = 250
    average_adjusted_fitness_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["average_raw_fitness"])
    assert len(average_adjusted_fitness_neat_ncl_250) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): average_adjusted_fitness_neat_ncl_250 has length {len(average_adjusted_fitness_neat_ncl_250)}"
    axis.plot(data_x, average_adjusted_fitness_neat_ncl_250,
              marker="o", label=f"NEAT NCL - ensemble_size: 250")

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


def average_adjusted_fitness_against_average_active_hidden_node_num(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            50,
            100,
            150,
            200,
            250
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"Number of active hidden nodes"
    Y_AXIS = f"Adjusted fitness"
    FIGURE_TITLE = f"Adjusted fitness vs Number of active hidden nodes"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_average_adjusted_fitness_against_average_active_hidden_node_num.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # Plot lines
    # NEAT NCL - ensemble_size = 50
    average_adjusted_fitness_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["average_adjusted_fitness"])
    average_active_hidden_node_num_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["average_active_hidden_node_num"])
    assert len(average_adjusted_fitness_neat_ncl_50) == len(
        average_active_hidden_node_num_neat_ncl_50), f"Invalid number of data points (expect {len(average_active_hidden_node_num_neat_ncl_50)}): average_adjusted_fitness_neat_ncl_50 has length {len(average_adjusted_fitness_neat_ncl_50)}"
    axis.scatter(average_active_hidden_node_num_neat_ncl_50, average_adjusted_fitness_neat_ncl_50,
                 marker="o", label=f"NEAT NCL - ensemble_size: 50")

    # NEAT NCL - ensemble_size = 100
    average_adjusted_fitness_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["average_adjusted_fitness"])
    average_active_hidden_node_num_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["average_active_hidden_node_num"])
    assert len(average_adjusted_fitness_neat_ncl_100) == len(
        average_active_hidden_node_num_neat_ncl_100), f"Invalid number of data points (expect {len(average_active_hidden_node_num_neat_ncl_100)}): average_adjusted_fitness_neat_ncl_100 has length {len(average_adjusted_fitness_neat_ncl_100)}"
    axis.scatter(average_active_hidden_node_num_neat_ncl_100, average_adjusted_fitness_neat_ncl_100,
                 marker="o", label=f"NEAT NCL - ensemble_size: 100")

    # NEAT NCL - ensemble_size = 150
    average_adjusted_fitness_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["average_adjusted_fitness"])
    average_active_hidden_node_num_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["average_active_hidden_node_num"])
    assert len(average_adjusted_fitness_neat_ncl_150) == len(
        average_active_hidden_node_num_neat_ncl_150), f"Invalid number of data points (expect {len(average_active_hidden_node_num_neat_ncl_150)}): average_adjusted_fitness_neat_ncl_150 has length {len(average_adjusted_fitness_neat_ncl_150)}"
    axis.scatter(average_active_hidden_node_num_neat_ncl_150, average_adjusted_fitness_neat_ncl_150,
                 marker="o", label=f"NEAT NCL - ensemble_size: 150")

    # NEAT NCL - ensemble_size = 200
    average_adjusted_fitness_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["average_adjusted_fitness"])
    average_active_hidden_node_num_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["average_active_hidden_node_num"])
    assert len(average_adjusted_fitness_neat_ncl_200) == len(
        average_active_hidden_node_num_neat_ncl_200), f"Invalid number of data points (expect {len(average_active_hidden_node_num_neat_ncl_200)}): average_adjusted_fitness_neat_ncl_200 has length {len(average_adjusted_fitness_neat_ncl_200)}"
    axis.scatter(average_active_hidden_node_num_neat_ncl_200, average_adjusted_fitness_neat_ncl_200,
                 marker="o", label=f"NEAT NCL - ensemble_size: 200")

    # NEAT NCL - ensemble_size = 250
    average_adjusted_fitness_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["average_adjusted_fitness"])
    average_active_hidden_node_num_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["average_active_hidden_node_num"])
    assert len(average_adjusted_fitness_neat_ncl_250) == len(
        average_active_hidden_node_num_neat_ncl_250), f"Invalid number of data points (expect {len(average_active_hidden_node_num_neat_ncl_250)}): average_active_hidden_node_num_neat_ncl_250 has length {len(average_adjusted_fitness_neat_ncl_250)}"
    axis.scatter(average_active_hidden_node_num_neat_ncl_250, average_adjusted_fitness_neat_ncl_250,
                 marker="o", label=f"NEAT NCL - ensemble_size: 250")

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


def subpopulation_num_against_generation_index_0_480(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            50,
            100,
            150,
            200,
            250
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"Number of generations"
    Y_AXIS = f"Number of subpopulations"
    FIGURE_TITLE = f"Number of subpopulations vs Number of generations"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_subpopulation_num_against_generation_index_0_480.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(df_neat_ncl["generation_index"].unique())

    # Plot lines
    # NEAT NCL - ensemble_size = 50
    subpopulation_num_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["subpopulation_num"])
    assert len(subpopulation_num_neat_ncl_50) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): subpopulation_num_neat_ncl_50 has length {len(subpopulation_num_neat_ncl_50)}"
    axis.plot(data_x, subpopulation_num_neat_ncl_50,
              marker="o", label=f"NEAT NCL - ensemble_size: 50")

    # NEAT NCL - ensemble_size = 100
    subpopulation_num_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["subpopulation_num"])
    assert len(subpopulation_num_neat_ncl_100) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): subpopulation_num_neat_ncl_100 has length {len(subpopulation_num_neat_ncl_100)}"
    axis.plot(data_x, subpopulation_num_neat_ncl_100,
              marker="o", label=f"NEAT NCL - ensemble_size: 100")

    # NEAT NCL - ensemble_size = 150
    subpopulation_num_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["subpopulation_num"])
    assert len(subpopulation_num_neat_ncl_150) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): subpopulation_num_neat_ncl_150 has length {len(subpopulation_num_neat_ncl_150)}"
    axis.plot(data_x, subpopulation_num_neat_ncl_150,
              marker="o", label=f"NEAT NCL - ensemble_size: 150")

    # NEAT NCL - ensemble_size = 200
    subpopulation_num_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["subpopulation_num"])
    assert len(subpopulation_num_neat_ncl_200) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): subpopulation_num_neat_ncl_200 has length {len(subpopulation_num_neat_ncl_200)}"
    axis.plot(data_x, subpopulation_num_neat_ncl_200,
              marker="o", label=f"NEAT NCL - ensemble_size: 200")

    # NEAT NCL - ensemble_size = 250
    subpopulation_num_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["subpopulation_num"])
    assert len(subpopulation_num_neat_ncl_250) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): subpopulation_num_neat_ncl_250 has length {len(subpopulation_num_neat_ncl_250)}"
    axis.plot(data_x, subpopulation_num_neat_ncl_250,
              marker="o", label=f"NEAT NCL - ensemble_size: 250")

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


def population_diversity_against_subpopulation_num(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            50,
            100,
            150,
            200,
            250
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"Number of subpopulations"
    Y_AXIS = f"$D$"
    FIGURE_TITLE = f"$D$ vs Number of subpopulations"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_population_diversity_against_subpopulation_num.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # Plot lines
    # NEAT NCL - ensemble_size = 50
    population_diversity_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["population_diversity_test"])
    subpopulation_num_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["subpopulation_num"])
    assert len(population_diversity_neat_ncl_50) == len(
        subpopulation_num_neat_ncl_50), f"Invalid number of data points (expect {len(subpopulation_num_neat_ncl_50)}): population_diversity_neat_ncl_50 has length {len(population_diversity_neat_ncl_50)}"
    axis.scatter(subpopulation_num_neat_ncl_50, population_diversity_neat_ncl_50,
                 marker="o", label=f"NEAT NCL - ensemble_size: 50")

    # NEAT NCL - ensemble_size = 100
    population_diversity_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["population_diversity_test"])
    subpopulation_num_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["subpopulation_num"])
    assert len(population_diversity_neat_ncl_100) == len(
        subpopulation_num_neat_ncl_100), f"Invalid number of data points (expect {len(subpopulation_num_neat_ncl_100)}): population_diversity_neat_ncl_100 has length {len(population_diversity_neat_ncl_100)}"
    axis.scatter(subpopulation_num_neat_ncl_100, population_diversity_neat_ncl_100,
                 marker="o", label=f"NEAT NCL - ensemble_size: 100")

    # NEAT NCL - ensemble_size = 150
    population_diversity_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["population_diversity_test"])
    subpopulation_num_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["subpopulation_num"])
    assert len(population_diversity_neat_ncl_150) == len(
        subpopulation_num_neat_ncl_150), f"Invalid number of data points (expect {len(subpopulation_num_neat_ncl_150)}): population_diversity_neat_ncl_150 has length {len(population_diversity_neat_ncl_150)}"
    axis.scatter(subpopulation_num_neat_ncl_150, population_diversity_neat_ncl_150,
                 marker="o", label=f"NEAT NCL - ensemble_size: 150")

    # NEAT NCL - ensemble_size = 200
    population_diversity_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["population_diversity_test"])
    subpopulation_num_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["subpopulation_num"])
    assert len(population_diversity_neat_ncl_200) == len(
        subpopulation_num_neat_ncl_200), f"Invalid number of data points (expect {len(subpopulation_num_neat_ncl_200)}): population_diversity_neat_ncl_200 has length {len(population_diversity_neat_ncl_200)}"
    axis.scatter(subpopulation_num_neat_ncl_200, population_diversity_neat_ncl_200,
                 marker="o", label=f"NEAT NCL - ensemble_size: 200")

    # NEAT NCL - ensemble_size = 250
    population_diversity_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["population_diversity_test"])
    subpopulation_num_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["subpopulation_num"])
    assert len(population_diversity_neat_ncl_250) == len(
        subpopulation_num_neat_ncl_250), f"Invalid number of data points (expect {len(subpopulation_num_neat_ncl_250)}): population_diversity_neat_ncl_250 has length {len(population_diversity_neat_ncl_250)}"
    axis.scatter(subpopulation_num_neat_ncl_250, population_diversity_neat_ncl_250,
                 marker="o", label=f"NEAT NCL - ensemble_size: 250")

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


def average_active_hidden_node_num_against_generation_index_0_480(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            50,
            100,
            150,
            200,
            250
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"Number of generations"
    Y_AXIS = f"Number of active hidden nodes"
    FIGURE_TITLE = f"Number of active hidden nodes vs Number of generations"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_average_active_hidden_node_num_against_generation_index_0_480.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(df_neat_ncl["generation_index"].unique())

    # Plot lines
    # NEAT NCL - ensemble_size = 50
    average_active_hidden_node_num_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["average_active_hidden_node_num"])
    assert len(average_active_hidden_node_num_neat_ncl_50) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): average_active_hidden_node_num_neat_ncl_50 has length {len(average_active_hidden_node_num_neat_ncl_50)}"
    axis.plot(data_x, average_active_hidden_node_num_neat_ncl_50,
              marker="o", label=f"NEAT NCL - ensemble_size: 50")

    # NEAT NCL - ensemble_size = 100
    average_active_hidden_node_num_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["average_active_hidden_node_num"])
    assert len(average_active_hidden_node_num_neat_ncl_100) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): average_active_hidden_node_num_neat_ncl_100 has length {len(average_active_hidden_node_num_neat_ncl_100)}"
    axis.plot(data_x, average_active_hidden_node_num_neat_ncl_100,
              marker="o", label=f"NEAT NCL - ensemble_size: 100")

    # NEAT NCL - ensemble_size = 150
    average_active_hidden_node_num_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["average_active_hidden_node_num"])
    assert len(average_active_hidden_node_num_neat_ncl_150) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): average_active_hidden_node_num_neat_ncl_150 has length {len(average_active_hidden_node_num_neat_ncl_150)}"
    axis.plot(data_x, average_active_hidden_node_num_neat_ncl_150,
              marker="o", label=f"NEAT NCL - ensemble_size: 150")

    # NEAT NCL - ensemble_size = 200
    average_active_hidden_node_num_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["average_active_hidden_node_num"])
    assert len(average_active_hidden_node_num_neat_ncl_200) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): average_active_hidden_node_num_neat_ncl_200 has length {len(average_active_hidden_node_num_neat_ncl_200)}"
    axis.plot(data_x, average_active_hidden_node_num_neat_ncl_200,
              marker="o", label=f"NEAT NCL - ensemble_size: 200")

    # NEAT NCL - ensemble_size = 250
    average_active_hidden_node_num_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["average_active_hidden_node_num"])
    assert len(average_active_hidden_node_num_neat_ncl_250) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): average_active_hidden_node_num_neat_ncl_250 has length {len(average_active_hidden_node_num_neat_ncl_250)}"
    axis.plot(data_x, average_active_hidden_node_num_neat_ncl_250,
              marker="o", label=f"NEAT NCL - ensemble_size: 250")

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


def population_diversity_against_average_active_hidden_node_num(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            50,
            100,
            150,
            200,
            250
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"Number of active hidden nodes"
    Y_AXIS = f"$D$"
    FIGURE_TITLE = f"$D$ vs Number of active hidden nodes"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_population_diversity_against_average_active_hidden_node_num.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # Plot lines
    # NEAT NCL - ensemble_size = 50
    population_diversity_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["population_diversity_test"])
    average_active_hidden_node_num_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["average_active_hidden_node_num"])
    assert len(population_diversity_neat_ncl_50) == len(
        average_active_hidden_node_num_neat_ncl_50), f"Invalid number of data points (expect {len(average_active_hidden_node_num_neat_ncl_50)}): population_diversity_neat_ncl_50 has length {len(population_diversity_neat_ncl_50)}"
    axis.scatter(average_active_hidden_node_num_neat_ncl_50, population_diversity_neat_ncl_50,
                 marker="o", label=f"NEAT NCL - ensemble_size: 50")

    # NEAT NCL - ensemble_size = 100
    population_diversity_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["population_diversity_test"])
    average_active_hidden_node_num_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["average_active_hidden_node_num"])
    assert len(population_diversity_neat_ncl_100) == len(
        average_active_hidden_node_num_neat_ncl_100), f"Invalid number of data points (expect {len(average_active_hidden_node_num_neat_ncl_100)}): population_diversity_neat_ncl_100 has length {len(population_diversity_neat_ncl_100)}"
    axis.scatter(average_active_hidden_node_num_neat_ncl_100, population_diversity_neat_ncl_100,
                 marker="o", label=f"NEAT NCL - ensemble_size: 100")

    # NEAT NCL - ensemble_size = 150
    population_diversity_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["population_diversity_test"])
    average_active_hidden_node_num_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["average_active_hidden_node_num"])
    assert len(population_diversity_neat_ncl_150) == len(
        average_active_hidden_node_num_neat_ncl_150), f"Invalid number of data points (expect {len(average_active_hidden_node_num_neat_ncl_150)}): population_diversity_neat_ncl_150 has length {len(population_diversity_neat_ncl_150)}"
    axis.scatter(average_active_hidden_node_num_neat_ncl_150, population_diversity_neat_ncl_150,
                 marker="o", label=f"NEAT NCL - ensemble_size: 150")

    # NEAT NCL - ensemble_size = 200
    population_diversity_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["population_diversity_test"])
    average_active_hidden_node_num_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["average_active_hidden_node_num"])
    assert len(population_diversity_neat_ncl_200) == len(
        average_active_hidden_node_num_neat_ncl_200), f"Invalid number of data points (expect {len(average_active_hidden_node_num_neat_ncl_200)}): population_diversity_neat_ncl_200 has length {len(population_diversity_neat_ncl_200)}"
    axis.scatter(average_active_hidden_node_num_neat_ncl_200, population_diversity_neat_ncl_200,
                 marker="o", label=f"NEAT NCL - ensemble_size: 200")

    # NEAT NCL - ensemble_size = 250
    population_diversity_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["population_diversity_test"])
    average_active_hidden_node_num_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["average_active_hidden_node_num"])
    assert len(population_diversity_neat_ncl_250) == len(
        average_active_hidden_node_num_neat_ncl_250), f"Invalid number of data points (expect {len(average_active_hidden_node_num_neat_ncl_250)}): population_diversity_neat_ncl_250 has length {len(population_diversity_neat_ncl_250)}"
    axis.scatter(average_active_hidden_node_num_neat_ncl_250, population_diversity_neat_ncl_250,
                 marker="o", label=f"NEAT NCL - ensemble_size: 250")

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


def diversity_coefficient_against_generation_index_0_480_fixed_ensemble_size_50(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and raw_df_static_ncl is not None and raw_df_traditional is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, raw_df_static_ncl, raw_df_traditional, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            50
        ]
    }
    filter_static_ncl = {
        "hidden_size": [
            # Pick best hidden_size
            [6]
        ],
        "ensemble_size": [
            50
        ],
        "correlation_penalty_coefficient": [
            # Pick best correlation_penalty_coefficient
            0.7
        ],
        "epoch_num": [
            400
        ]
    }
    filter_traditional = {
        "hidden_size": [
            # Pick best hidden_size
            [12]
        ],
        "ensemble_size": [
            50
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"Number of generations"
    Y_AXIS = f"$\\rho$"
    FIGURE_TITLE = f"$\\rho$ vs Number of generations"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_diversity_coefficient_against_generation_index_0_480_fixed_ensemble_size_50.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )
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
    data_x = np.array(df_neat_ncl["generation_index"].unique())

    # Plot lines
    # NEAT NCL - ensemble_size = 50
    diversity_coefficient_neat_ncl_50 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 50]["diversity_coefficient_test"])
    assert len(diversity_coefficient_neat_ncl_50) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_neat_ncl_50 has length {len(diversity_coefficient_neat_ncl_50)}"
    axis.plot(data_x, diversity_coefficient_neat_ncl_50,
              marker="o", label=f"NEAT NCL - ensemble_size: 50")

    # Static NCL - ensemble_size = 50
    # Expect only 1 data point for static NCL originally
    diversity_coefficient_static_ncl_50 = np.repeat(np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 50]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_static_ncl_50) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_50 has length {len(diversity_coefficient_static_ncl_50)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_50,
              marker="o", label=f"NCL - ensemble_size: 50")

    # Arithmetic-mean-ensemble - ensemble_size = 50
    # Expect only 1 data point for traditional originally
    diversity_coefficient_traditional_50 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 50]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_50) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_50 has length {len(diversity_coefficient_traditional_50)}"
    axis.plot(data_x, diversity_coefficient_traditional_50,
              marker="o", label=f"Arithmetic-mean-ensemble - ensemble_size: 50")

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


def diversity_coefficient_against_generation_index_0_480_fixed_ensemble_size_100(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and raw_df_static_ncl is not None and raw_df_traditional is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, raw_df_static_ncl, raw_df_traditional, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            100
        ]
    }
    filter_static_ncl = {
        "hidden_size": [
            # Pick best hidden_size
            [6]
        ],
        "ensemble_size": [
            100
        ],
        "correlation_penalty_coefficient": [
            # Pick best correlation_penalty_coefficient
            0.7
        ],
        "epoch_num": [
            400
        ]
    }
    filter_traditional = {
        "hidden_size": [
            # Pick best hidden_size
            [12]
        ],
        "ensemble_size": [
            100
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"Number of generations"
    Y_AXIS = f"$\\rho$"
    FIGURE_TITLE = f"$\\rho$ vs Number of generations"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_diversity_coefficient_against_generation_index_0_480_fixed_ensemble_size_100.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )
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
    data_x = np.array(df_neat_ncl["generation_index"].unique())

    # Plot lines
    # NEAT NCL - ensemble_size = 100
    diversity_coefficient_neat_ncl_100 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 100]["diversity_coefficient_test"])
    assert len(diversity_coefficient_neat_ncl_100) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_neat_ncl_100 has length {len(diversity_coefficient_neat_ncl_100)}"
    axis.plot(data_x, diversity_coefficient_neat_ncl_100,
              marker="o", label=f"NEAT NCL - ensemble_size: 100")

    # Static NCL - ensemble_size = 100
    # Expect only 1 data point for static NCL originally
    diversity_coefficient_static_ncl_100 = np.repeat(np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 100]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_static_ncl_100) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_100 has length {len(diversity_coefficient_static_ncl_100)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_100,
              marker="o", label=f"NCL - ensemble_size: 100")

    # Arithmetic-mean-ensemble - ensemble_size = 100
    # Expect only 1 data point for traditional originally
    diversity_coefficient_traditional_100 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 100]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_100) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_100 has length {len(diversity_coefficient_traditional_100)}"
    axis.plot(data_x, diversity_coefficient_traditional_100,
              marker="o", label=f"Arithmetic-mean-ensemble - ensemble_size: 100")

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


def diversity_coefficient_against_generation_index_0_480_fixed_ensemble_size_150(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and raw_df_static_ncl is not None and raw_df_traditional is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, raw_df_static_ncl, raw_df_traditional, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            150
        ]
    }
    filter_static_ncl = {
        "hidden_size": [
            # Pick best hidden_size
            [6]
        ],
        "ensemble_size": [
            150
        ],
        "correlation_penalty_coefficient": [
            # Pick best correlation_penalty_coefficient
            0.7
        ],
        "epoch_num": [
            400
        ]
    }
    filter_traditional = {
        "hidden_size": [
            # Pick best hidden_size
            [12]
        ],
        "ensemble_size": [
            150
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"Number of generations"
    Y_AXIS = f"$\\rho$"
    FIGURE_TITLE = f"$\\rho$ vs Number of generations"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_diversity_coefficient_against_generation_index_0_480_fixed_ensemble_size_150.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )
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
    data_x = np.array(df_neat_ncl["generation_index"].unique())

    # Plot lines
    # NEAT NCL - ensemble_size = 150
    diversity_coefficient_neat_ncl_150 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 150]["diversity_coefficient_test"])
    assert len(diversity_coefficient_neat_ncl_150) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_neat_ncl_150 has length {len(diversity_coefficient_neat_ncl_150)}"
    axis.plot(data_x, diversity_coefficient_neat_ncl_150,
              marker="o", label=f"NEAT NCL - ensemble_size: 150")

    # Static NCL - ensemble_size = 150
    # Expect only 1 data point for static NCL originally
    diversity_coefficient_static_ncl_150 = np.repeat(np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 150]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_static_ncl_150) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_150 has length {len(diversity_coefficient_static_ncl_150)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_150,
              marker="o", label=f"NCL - ensemble_size: 150")

    # Arithmetic-mean-ensemble - ensemble_size = 150
    # Expect only 1 data point for traditional originally
    diversity_coefficient_traditional_150 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 150]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_150) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_150 has length {len(diversity_coefficient_traditional_150)}"
    axis.plot(data_x, diversity_coefficient_traditional_150,
              marker="o", label=f"Arithmetic-mean-ensemble - ensemble_size: 150")

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


def diversity_coefficient_against_generation_index_0_480_fixed_ensemble_size_200(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and raw_df_static_ncl is not None and raw_df_traditional is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, raw_df_static_ncl, raw_df_traditional, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            200
        ]
    }
    filter_static_ncl = {
        "hidden_size": [
            # Pick best hidden_size
            [6]
        ],
        "ensemble_size": [
            200
        ],
        "correlation_penalty_coefficient": [
            # Pick best correlation_penalty_coefficient
            0.7
        ],
        "epoch_num": [
            400
        ]
    }
    filter_traditional = {
        "hidden_size": [
            # Pick best hidden_size
            [12]
        ],
        "ensemble_size": [
            200
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"Number of generations"
    Y_AXIS = f"$\\rho$"
    FIGURE_TITLE = f"$\\rho$ vs Number of generations"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_diversity_coefficient_against_generation_index_0_480_fixed_ensemble_size_200.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )
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
    data_x = np.array(df_neat_ncl["generation_index"].unique())

    # Plot lines
    # NEAT NCL - ensemble_size = 200
    diversity_coefficient_neat_ncl_200 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 200]["diversity_coefficient_test"])
    assert len(diversity_coefficient_neat_ncl_200) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_neat_ncl_200 has length {len(diversity_coefficient_neat_ncl_200)}"
    axis.plot(data_x, diversity_coefficient_neat_ncl_200,
              marker="o", label=f"NEAT NCL - ensemble_size: 200")

    # Static NCL - ensemble_size = 200
    # Expect only 1 data point for static NCL originally
    diversity_coefficient_static_ncl_200 = np.repeat(np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 200]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_static_ncl_200) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_200 has length {len(diversity_coefficient_static_ncl_200)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_200,
              marker="o", label=f"NCL - ensemble_size: 200")

    # Arithmetic-mean-ensemble - ensemble_size = 200
    # Expect only 1 data point for traditional originally
    diversity_coefficient_traditional_200 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 200]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_200) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_200 has length {len(diversity_coefficient_traditional_200)}"
    axis.plot(data_x, diversity_coefficient_traditional_200,
              marker="o", label=f"Arithmetic-mean-ensemble - ensemble_size: 200")

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


def diversity_coefficient_against_generation_index_0_480_fixed_ensemble_size_250(raw_df_neat_ncl: pd.DataFrame, raw_df_static_ncl: pd.DataFrame, raw_df_traditional: pd.DataFrame, raw_df_mlp: pd.DataFrame, img_repository_path: str):
    assert raw_df_neat_ncl is not None and raw_df_static_ncl is not None and raw_df_traditional is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_neat_ncl, raw_df_static_ncl, raw_df_traditional, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_neat_ncl = {
        "ensemble_size": [
            250
        ]
    }
    filter_static_ncl = {
        "hidden_size": [
            # Pick best hidden_size
            [6]
        ],
        "ensemble_size": [
            250
        ],
        "correlation_penalty_coefficient": [
            # Pick best correlation_penalty_coefficient
            0.7
        ],
        "epoch_num": [
            400
        ]
    }
    filter_traditional = {
        "hidden_size": [
            # Pick best hidden_size
            [12]
        ],
        "ensemble_size": [
            250
        ],
        "epoch_num": [
            400
        ]
    }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"Number of generations"
    Y_AXIS = f"$\\rho$"
    FIGURE_TITLE = f"$\\rho$ vs Number of generations"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"neat_ncl_diversity_coefficient_against_generation_index_0_480_fixed_ensemble_size_250.png"
    # ====================================================================

    df_neat_ncl = data_helper.apply_dataframe_filter(
        df=raw_df_neat_ncl,
        filter=filter_neat_ncl
    )
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
    data_x = np.array(df_neat_ncl["generation_index"].unique())

    # Plot lines
    # NEAT NCL - ensemble_size = 250
    diversity_coefficient_neat_ncl_250 = np.array(
        df_neat_ncl[df_neat_ncl["ensemble_size"] == 250]["diversity_coefficient_test"])
    assert len(diversity_coefficient_neat_ncl_250) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_neat_ncl_250 has length {len(diversity_coefficient_neat_ncl_250)}"
    axis.plot(data_x, diversity_coefficient_neat_ncl_250,
              marker="o", label=f"NEAT NCL - ensemble_size: 250")

    # Static NCL - ensemble_size = 250
    # Expect only 1 data point for static NCL originally
    diversity_coefficient_static_ncl_250 = np.repeat(np.array(
        df_static_ncl[df_static_ncl["ensemble_size"] == 250]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_static_ncl_250) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_static_ncl_250 has length {len(diversity_coefficient_static_ncl_250)}"
    axis.plot(data_x, diversity_coefficient_static_ncl_250,
              marker="o", label=f"NCL - ensemble_size: 250")

    # Arithmetic-mean-ensemble - ensemble_size = 250
    # Expect only 1 data point for traditional originally
    diversity_coefficient_traditional_250 = np.repeat(np.array(
        df_traditional[df_traditional["ensemble_size"] == 250]["diversity_coefficient_test"]), len(data_x))
    assert len(diversity_coefficient_traditional_250) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): diversity_coefficient_traditional_250 has length {len(diversity_coefficient_traditional_250)}"
    axis.plot(data_x, diversity_coefficient_traditional_250,
              marker="o", label=f"Arithmetic-mean-ensemble - ensemble_size: 250")

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
    df_neat_ncl = pd.read_csv(CSV_PATH_NEAT_NCL)
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

    mse_against_generation_index_0_480(
        raw_df_neat_ncl=df_neat_ncl,
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    )

    mse_against_generation_index_0_480_fixed_ensemble_size_50(
        raw_df_neat_ncl=df_neat_ncl,
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    )

    mse_against_generation_index_0_480_fixed_ensemble_size_100(
        raw_df_neat_ncl=df_neat_ncl,
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    )

    mse_against_generation_index_0_480_fixed_ensemble_size_150(
        raw_df_neat_ncl=df_neat_ncl,
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    )

    mse_against_generation_index_0_480_fixed_ensemble_size_200(
        raw_df_neat_ncl=df_neat_ncl,
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    )

    mse_against_generation_index_0_480_fixed_ensemble_size_250(
        raw_df_neat_ncl=df_neat_ncl,
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    )

    mse_against_diversity_coefficient_fixed_ensemble_size_50(
        raw_df_neat_ncl=df_neat_ncl,
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    )

    mse_against_diversity_coefficient_fixed_ensemble_size_100(
        raw_df_neat_ncl=df_neat_ncl,
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    )

    mse_against_diversity_coefficient_fixed_ensemble_size_150(
        raw_df_neat_ncl=df_neat_ncl,
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    )

    mse_against_diversity_coefficient_fixed_ensemble_size_200(
        raw_df_neat_ncl=df_neat_ncl,
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    )

    mse_against_diversity_coefficient_fixed_ensemble_size_250(
        raw_df_neat_ncl=df_neat_ncl,
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    )

    mse_against_population_diversity(
        raw_df_neat_ncl=df_neat_ncl,
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    )

    correlation_penalty_coefficient_against_generation_index_0_480(
        raw_df_neat_ncl=df_neat_ncl,
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    )

    correlation_penalty_coefficient_against_population_diversity(
        raw_df_neat_ncl=df_neat_ncl,
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    )

    correlation_penalty_coefficient_against_population_diversity_ratio(
        raw_df_neat_ncl=df_neat_ncl,
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    )

    population_diversity_against_generation_index_0_480(
        raw_df_neat_ncl=df_neat_ncl,
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    )

    max_population_diversity_against_generation_index_0_480(
        raw_df_neat_ncl=df_neat_ncl,
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    )

    population_diversity_ratio_against_generation_index_0_480(
        raw_df_neat_ncl=df_neat_ncl,
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    )

    average_niche_radius_against_generation_index_0_480(
        raw_df_neat_ncl=df_neat_ncl,
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    )

    population_diversity_against_average_niche_radius(
        raw_df_neat_ncl=df_neat_ncl,
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    )

    average_sharing_factor_against_average_niche_radius(
        raw_df_neat_ncl=df_neat_ncl,
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    )

    average_adjusted_fitness_against_average_niche_radius(
        raw_df_neat_ncl=df_neat_ncl,
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    )

    average_sharing_factor_against_generation_index_0_480(
        raw_df_neat_ncl=df_neat_ncl,
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    )

    population_diversity_against_average_sharing_factor(
        raw_df_neat_ncl=df_neat_ncl,
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    )

    average_adjusted_fitness_against_average_sharing_factor(
        raw_df_neat_ncl=df_neat_ncl,
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    )

    average_raw_fitness_against_generation_index_0_480(
        raw_df_neat_ncl=df_neat_ncl,
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    )

    average_raw_fitness_against_average_active_hidden_node_num(
        raw_df_neat_ncl=df_neat_ncl,
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    )

    average_adjusted_fitness_against_generation_index_0_480(
        raw_df_neat_ncl=df_neat_ncl,
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    )

    average_adjusted_fitness_against_average_active_hidden_node_num(
        raw_df_neat_ncl=df_neat_ncl,
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    )

    subpopulation_num_against_generation_index_0_480(
        raw_df_neat_ncl=df_neat_ncl,
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    )

    population_diversity_against_subpopulation_num(
        raw_df_neat_ncl=df_neat_ncl,
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    )

    average_active_hidden_node_num_against_generation_index_0_480(
        raw_df_neat_ncl=df_neat_ncl,
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    )

    population_diversity_against_average_active_hidden_node_num(
        raw_df_neat_ncl=df_neat_ncl,
        raw_df_static_ncl=df_static_ncl,
        raw_df_traditional=df_traditional,
        raw_df_mlp=df_mlp,
        img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    )

    # NOTE: Will not be plotted as they will not be used in my report
    # diversity_coefficient_against_generation_index_0_480_fixed_ensemble_size_50(
    #     raw_df_neat_ncl=df_neat_ncl,
    #     raw_df_static_ncl=df_static_ncl,
    #     raw_df_traditional=df_traditional,
    #     raw_df_mlp=df_mlp,
    #     img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    # )

    # NOTE: Will not be plotted as they will not be used in my report
    # diversity_coefficient_against_generation_index_0_480_fixed_ensemble_size_100(
    #     raw_df_neat_ncl=df_neat_ncl,
    #     raw_df_static_ncl=df_static_ncl,
    #     raw_df_traditional=df_traditional,
    #     raw_df_mlp=df_mlp,
    #     img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    # )

    # NOTE: Will not be plotted as they will not be used in my report
    # diversity_coefficient_against_generation_index_0_480_fixed_ensemble_size_150(
    #     raw_df_neat_ncl=df_neat_ncl,
    #     raw_df_static_ncl=df_static_ncl,
    #     raw_df_traditional=df_traditional,
    #     raw_df_mlp=df_mlp,
    #     img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    # )

    # NOTE: Will not be plotted as they will not be used in my report
    # diversity_coefficient_against_generation_index_0_480_fixed_ensemble_size_200(
    #     raw_df_neat_ncl=df_neat_ncl,
    #     raw_df_static_ncl=df_static_ncl,
    #     raw_df_traditional=df_traditional,
    #     raw_df_mlp=df_mlp,
    #     img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    # )

    # NOTE: Will not be plotted as they will not be used in my report
    # diversity_coefficient_against_generation_index_0_480_fixed_ensemble_size_250(
    #     raw_df_neat_ncl=df_neat_ncl,
    #     raw_df_static_ncl=df_static_ncl,
    #     raw_df_traditional=df_traditional,
    #     raw_df_mlp=df_mlp,
    #     img_repository_path=IMG_REPOSITORY_PATH_NEAT_NCL
    # )

    logger.log(f"Graphs of NEAT NCL experiment are saved ...")
