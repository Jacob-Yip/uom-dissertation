import ast
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from src.utils import data_helper, logger

"""
Run: python -m src.utils.experiment.data_plotter_voter_neat_ncl
"""


# Constants
# Path to project root
ROOT_PATH = os.path.join(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Path to each csv file in csv/
CSV_PATH_ARITHMETIC_MEAN = os.path.join(
    ROOT_PATH, f"data", f"experiment", f"neat-ncl", f"csv", f"experiment_runner_neat_ncl_unconnected_arithmetic_mean.csv")
CSV_PATH_MEDIAN = os.path.join(
    ROOT_PATH, f"data", f"experiment", f"neat-ncl", f"csv", f"experiment_runner_neat_ncl_unconnected_median.csv")
# CSV_PATH_NN = os.path.join(
#     ROOT_PATH, f"data", f"experiment", f"neat-ncl", f"csv", f"experiment_runner_neat_ncl_unconnected_nn.csv")

# Path to img/
IMG_REPOSITORY_PATH_VOTER_NEAT_NCL = os.path.join(
    ROOT_PATH, f"data", f"experiment", f"voter", f"neat-ncl", f"img")

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


def mse_against_generation_index_0_400_fixed_ensemble_size_50(raw_df_arithmetic_mean: pd.DataFrame, raw_df_median: pd.DataFrame, img_repository_path: str):
    assert raw_df_arithmetic_mean is not None and raw_df_median is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_arithmetic_mean, raw_df_median, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_arithmetic_mean = {
        "ensemble_size": [
            50
        ],
        "generation_index": list(range(400))
    }
    filter_median = {
        "ensemble_size": [
            50
        ],
        "generation_index": list(range(400))
    }
    # filter_nn = {
    #     "ensemble_size": [
    #         50
    #     ],
    #     "generation_index": list(range(400))
    # }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"Number of generations"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs Number of generations"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"voter_neat_ncl_mse_against_generation_index_0_400_fixed_ensemble_size_50.png"
    # ====================================================================

    df_arithmetic_mean = data_helper.apply_dataframe_filter(
        df=raw_df_arithmetic_mean,
        filter=filter_arithmetic_mean
    )
    df_median = data_helper.apply_dataframe_filter(
        df=raw_df_median,
        filter=filter_median
    )
    # df_nn = data_helper.apply_dataframe_filter(
    #     df=raw_df_nn,
    #     filter=filter_nn
    # )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(df_arithmetic_mean["generation_index"].unique())

    # Plot lines
    # Arithmetic mean
    mse_arithmetic_mean = np.array(
        df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 50]["loss_test"])
    assert len(mse_arithmetic_mean) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean has length {len(mse_arithmetic_mean)}"
    axis.plot(data_x, mse_arithmetic_mean, marker="o",
              label=f"Arithmetic mean - ensemble_size: 50")

    # Median
    mse_median = np.array(
        df_median[df_median["ensemble_size"] == 50]["loss_test"])
    assert len(mse_median) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median has length {len(mse_median)}"
    axis.plot(data_x, mse_median, marker="o",
              label=f"Median - ensemble_size: 50")

    # # Neural network
    # mse_nn = np.array(df_nn[df_nn["ensemble_size"] == 50]["loss_test"])
    # assert len(mse_nn) == len(data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn has length {len(mse_nn)}"
    # axis.plot(data_x, mse_nn, marker="o", label=f"Neural network - ensemble_size: 50")

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


def mse_against_generation_index_0_400_fixed_ensemble_size_100(raw_df_arithmetic_mean: pd.DataFrame, raw_df_median: pd.DataFrame, img_repository_path: str):
    assert raw_df_arithmetic_mean is not None and raw_df_median is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_arithmetic_mean, raw_df_median, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_arithmetic_mean = {
        "ensemble_size": [
            100
        ],
        "generation_index": list(range(400))
    }
    filter_median = {
        "ensemble_size": [
            100
        ],
        "generation_index": list(range(400))
    }
    # filter_nn = {
    #     "ensemble_size": [
    #         100
    #     ],
    #     "generation_index": list(range(400))
    # }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"Number of generations"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs Number of generations"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"voter_neat_ncl_mse_against_generation_index_0_400_fixed_ensemble_size_100.png"
    # ====================================================================

    df_arithmetic_mean = data_helper.apply_dataframe_filter(
        df=raw_df_arithmetic_mean,
        filter=filter_arithmetic_mean
    )
    df_median = data_helper.apply_dataframe_filter(
        df=raw_df_median,
        filter=filter_median
    )
    # df_nn = data_helper.apply_dataframe_filter(
    #     df=raw_df_nn,
    #     filter=filter_nn
    # )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(df_arithmetic_mean["generation_index"].unique())

    # Plot lines
    # Arithmetic mean
    mse_arithmetic_mean = np.array(
        df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 100]["loss_test"])
    assert len(mse_arithmetic_mean) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean has length {len(mse_arithmetic_mean)}"
    axis.plot(data_x, mse_arithmetic_mean, marker="o",
              label=f"Arithmetic mean - ensemble_size: 100")

    # Median
    mse_median = np.array(
        df_median[df_median["ensemble_size"] == 100]["loss_test"])
    assert len(mse_median) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median has length {len(mse_median)}"
    axis.plot(data_x, mse_median, marker="o",
              label=f"Median - ensemble_size: 100")

    # # Neural network
    # mse_nn = np.array(df_nn[df_nn["ensemble_size"] == 100]["loss_test"])
    # assert len(mse_nn) == len(data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn has length {len(mse_nn)}"
    # axis.plot(data_x, mse_nn, marker="o", label=f"Neural network - ensemble_size: 100")

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


def mse_against_generation_index_0_400_fixed_ensemble_size_150(raw_df_arithmetic_mean: pd.DataFrame, raw_df_median: pd.DataFrame, img_repository_path: str):
    assert raw_df_arithmetic_mean is not None and raw_df_median is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_arithmetic_mean, raw_df_median, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_arithmetic_mean = {
        "ensemble_size": [
            150
        ],
        "generation_index": list(range(400))
    }
    filter_median = {
        "ensemble_size": [
            150
        ],
        "generation_index": list(range(400))
    }
    # filter_nn = {
    #     "ensemble_size": [
    #         150
    #     ],
    #     "generation_index": list(range(400))
    # }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"Number of generations"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs Number of generations"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"voter_neat_ncl_mse_against_generation_index_0_400_fixed_ensemble_size_150.png"
    # ====================================================================

    df_arithmetic_mean = data_helper.apply_dataframe_filter(
        df=raw_df_arithmetic_mean,
        filter=filter_arithmetic_mean
    )
    df_median = data_helper.apply_dataframe_filter(
        df=raw_df_median,
        filter=filter_median
    )
    # df_nn = data_helper.apply_dataframe_filter(
    #     df=raw_df_nn,
    #     filter=filter_nn
    # )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(df_arithmetic_mean["generation_index"].unique())

    # Plot lines
    # Arithmetic mean
    mse_arithmetic_mean = np.array(
        df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 150]["loss_test"])
    assert len(mse_arithmetic_mean) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean has length {len(mse_arithmetic_mean)}"
    axis.plot(data_x, mse_arithmetic_mean, marker="o",
              label=f"Arithmetic mean - ensemble_size: 150")

    # Median
    mse_median = np.array(
        df_median[df_median["ensemble_size"] == 150]["loss_test"])
    assert len(mse_median) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median has length {len(mse_median)}"
    axis.plot(data_x, mse_median, marker="o",
              label=f"Median - ensemble_size: 150")

    # # Neural network
    # mse_nn = np.array(df_nn[df_nn["ensemble_size"] == 150]["loss_test"])
    # assert len(mse_nn) == len(data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn has length {len(mse_nn)}"
    # axis.plot(data_x, mse_nn, marker="o", label=f"Neural network - ensemble_size: 150")

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


def mse_against_generation_index_0_400_fixed_ensemble_size_200(raw_df_arithmetic_mean: pd.DataFrame, raw_df_median: pd.DataFrame, img_repository_path: str):
    assert raw_df_arithmetic_mean is not None and raw_df_median is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_arithmetic_mean, raw_df_median, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_arithmetic_mean = {
        "ensemble_size": [
            200
        ], 
        "generation_index": list(range(400))
    }
    filter_median = {
        "ensemble_size": [
            200
        ],
        "generation_index": list(range(400))
    }
    # filter_nn = {
    #     "ensemble_size": [
    #         200
    #     ],
    #     "generation_index": list(range(400))
    # }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"Number of generations"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs Number of generations"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"voter_neat_ncl_mse_against_generation_index_0_400_fixed_ensemble_size_200.png"
    # ====================================================================

    df_arithmetic_mean = data_helper.apply_dataframe_filter(
        df=raw_df_arithmetic_mean,
        filter=filter_arithmetic_mean
    )
    df_median = data_helper.apply_dataframe_filter(
        df=raw_df_median,
        filter=filter_median
    )
    # df_nn = data_helper.apply_dataframe_filter(
    #     df=raw_df_nn,
    #     filter=filter_nn
    # )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(df_arithmetic_mean["generation_index"].unique())

    # Plot lines
    # Arithmetic mean
    mse_arithmetic_mean = np.array(
        df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 200]["loss_test"])
    assert len(mse_arithmetic_mean) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean has length {len(mse_arithmetic_mean)}"
    axis.plot(data_x, mse_arithmetic_mean, marker="o",
              label=f"Arithmetic mean - ensemble_size: 200")

    # Median
    mse_median = np.array(
        df_median[df_median["ensemble_size"] == 200]["loss_test"])
    assert len(mse_median) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median has length {len(mse_median)}"
    axis.plot(data_x, mse_median, marker="o",
              label=f"Median - ensemble_size: 200")

    # # Neural network
    # mse_nn = np.array(df_nn[df_nn["ensemble_size"] == 200]["loss_test"])
    # assert len(mse_nn) == len(data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn has length {len(mse_nn)}"
    # axis.plot(data_x, mse_nn, marker="o", label=f"Neural network - ensemble_size: 200")

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


def mse_against_generation_index_0_400_fixed_ensemble_size_250(raw_df_arithmetic_mean: pd.DataFrame, raw_df_median: pd.DataFrame, img_repository_path: str):
    assert raw_df_arithmetic_mean is not None and raw_df_median is not None and img_repository_path is not None, f"Missing required arguments (expect not None): raw_df_arithmetic_mean, raw_df_median, img_repository_path"
    assert os.path.exists(
        img_repository_path), f"Absolute path to repository img/ does not exist: {img_repository_path}"

    # Configurations
    # Dataframe-related
    filter_arithmetic_mean = {
        "ensemble_size": [
            250
        ],
        "generation_index": list(range(400))
    }
    filter_median = {
        "ensemble_size": [
            250
        ],
        "generation_index": list(range(400))
    }
    # filter_nn = {
    #     "ensemble_size": [
    #         250
    #     ],
    #     "generation_index": list(range(400))
    # }
    # Graph-related
    FIGURE_SIZE = (6, 4)
    X_AXIS = f"Number of generations"
    Y_AXIS = f"MSE"
    FIGURE_TITLE = f"MSE vs Number of generations"
    # Related to the saved image file in img/
    IMG_RESOLUTION = 300
    IMG_NAME = f"voter_neat_ncl_mse_against_generation_index_0_400_fixed_ensemble_size_250.png"
    # ====================================================================

    df_arithmetic_mean = data_helper.apply_dataframe_filter(
        df=raw_df_arithmetic_mean,
        filter=filter_arithmetic_mean
    )
    df_median = data_helper.apply_dataframe_filter(
        df=raw_df_median,
        filter=filter_median
    )
    # df_nn = data_helper.apply_dataframe_filter(
    #     df=raw_df_nn,
    #     filter=filter_nn
    # )

    # Plot graph
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    # X-axis
    data_x = np.array(df_arithmetic_mean["generation_index"].unique())

    # Plot lines
    # Arithmetic mean
    mse_arithmetic_mean = np.array(
        df_arithmetic_mean[df_arithmetic_mean["ensemble_size"] == 250]["loss_test"])
    assert len(mse_arithmetic_mean) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_arithmetic_mean has length {len(mse_arithmetic_mean)}"
    axis.plot(data_x, mse_arithmetic_mean, marker="o",
              label=f"Arithmetic mean - ensemble_size: 250")

    # Median
    mse_median = np.array(
        df_median[df_median["ensemble_size"] == 250]["loss_test"])
    assert len(mse_median) == len(
        data_x), f"Invalid number of data points (expect {len(data_x)}): mse_median has length {len(mse_median)}"
    axis.plot(data_x, mse_median, marker="o",
              label=f"Median - ensemble_size: 250")

    # # Neural network
    # mse_nn = np.array(df_nn[df_nn["ensemble_size"] == 250]["loss_test"])
    # assert len(mse_nn) == len(data_x), f"Invalid number of data points (expect {len(data_x)}): mse_nn has length {len(mse_nn)}"
    # axis.plot(data_x, mse_nn, marker="o", label=f"Neural network - ensemble_size: 250")

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
    df_median = pd.read_csv(CSV_PATH_MEDIAN)
    # df_nn = pd.read_csv(CSV_PATH_NN)

    mse_against_generation_index_0_400_fixed_ensemble_size_50(
        raw_df_arithmetic_mean=df_arithmetic_mean,
        raw_df_median=df_median,
        img_repository_path=IMG_REPOSITORY_PATH_VOTER_NEAT_NCL
    )

    mse_against_generation_index_0_400_fixed_ensemble_size_100(
        raw_df_arithmetic_mean=df_arithmetic_mean,
        raw_df_median=df_median,
        img_repository_path=IMG_REPOSITORY_PATH_VOTER_NEAT_NCL
    )

    mse_against_generation_index_0_400_fixed_ensemble_size_150(
        raw_df_arithmetic_mean=df_arithmetic_mean,
        raw_df_median=df_median,
        img_repository_path=IMG_REPOSITORY_PATH_VOTER_NEAT_NCL
    )

    mse_against_generation_index_0_400_fixed_ensemble_size_200(
        raw_df_arithmetic_mean=df_arithmetic_mean,
        raw_df_median=df_median,
        img_repository_path=IMG_REPOSITORY_PATH_VOTER_NEAT_NCL
    )

    mse_against_generation_index_0_400_fixed_ensemble_size_250(
        raw_df_arithmetic_mean=df_arithmetic_mean,
        raw_df_median=df_median,
        img_repository_path=IMG_REPOSITORY_PATH_VOTER_NEAT_NCL
    )

    logger.log(f"Graphs of voter NEAT NCL experiment are saved ...")
