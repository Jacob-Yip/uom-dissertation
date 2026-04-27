import ast
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

"""
Run command: python -m csv_to_image
"""

# Constant
# TODO: Update me
REPOSITORY_EXPERIMENT_PATH = "data/experiment/static-ncl"
REPOSITORY_EXPERIMENT_IMAGE_PATH = f"{REPOSITORY_EXPERIMENT_PATH}/img"
CSV_PATH = f"{REPOSITORY_EXPERIMENT_PATH}/csv/experiment_static_ncl.csv"
IMG_NAME = f"experiment_static_ncl"
IMAGE_RESOLUTION = 300

# Global configurations of experiment graphs
FONT_SIZE = 16
FONT_FAMILY = "serif"
FONT_SERIF = ["Times New Roman"]
AXES_TITLESIZE = 16
AXES_TITLEWEIGHT = "normal"
AXES_LABELSIZE = 16
LEGEND_FONTSIZE = 12

# Traditional NCL
# ================================== Boundary ====================================
# Set plot style for nicer plots
# sns.set_theme(style="whitegrid", rc={
#     "font.size": FONT_SIZE,  # Default font size
#     "font.family": FONT_FAMILY,
#     "font.serif": FONT_SERIF,
#     "axes.titlesize": AXES_TITLESIZE,
#     "axes.titleweight": AXES_TITLEWEIGHT,
#     "axes.labelsize": AXES_LABELSIZE,
#     "legend.fontsize": LEGEND_FONTSIZE
# })

# plt.figure(figsize=(16, 24))  # Larger figure for 6 plots

# row_num = len(hyperparameters)
# column_num = 2

# for i, hyperparameter in enumerate(hyperparameters, 1):
#     if hyperparameter == "ensemble_size":
#         hyperparameter_text = "ensemble size"
#     elif hyperparameter == "epoch_num":
#         hyperparameter_text = "train epoch"
#     elif hyperparameter == "learning_rate":
#         hyperparameter_text = "learning rate"
#     else:
#         hyperparameter_text = hyperparameter

#     column_index = 0  # 0-indexed

#     # Average loss
#     plt.subplot(row_num, column_num, (i - 1) *
#                 column_num + (column_index + 1))
#     # sns.lineplot(
#     #     data=df_experiment,
#     #     x=hyperparameter,
#     #     y="average_loss_train_traditional",
#     #     marker='o',
#     #     label=f"Train - traditional ensemble model"
#     # )
#     sns.lineplot(
#         data=df_experiment,
#         x=hyperparameter,
#         y="average_loss_test_traditional",
#         marker='o',
#         label=f"Test - traditional ensemble model"
#     )
#     # sns.lineplot(
#     #     data=df_experiment,
#     #     x=hyperparameter,
#     #     y="average_loss_train_mlp",
#     #     marker='o',
#     #     label=f"Train - MLP model"
#     # )
#     sns.lineplot(
#         data=df_experiment,
#         x=hyperparameter,
#         y="average_loss_test_mlp",
#         marker='o',
#         label=f"Test - MLP model"
#     )
#     # sns.lineplot(
#     #     data=df_experiment,
#     #     x=hyperparameter,
#     #     y="average_individual_loss_train_traditional",
#     #     marker='o',
#     #     label=f"Train - base learner (average)"
#     # )
#     sns.lineplot(
#         data=df_experiment,
#         x=hyperparameter,
#         y="average_individual_loss_test_traditional",
#         marker='o',
#         label=f"Test - base learner (average)"
#     )
#     plt.title(f"MSE vs {hyperparameter_text}")
#     plt.xlabel(hyperparameter_text)
#     plt.ylabel("MSE")
#     plt.legend()

#     # ===================================================================

#     # Diversity Coefficient
#     column_index = 1  # 0-indexed

#     plt.subplot(row_num, column_num, (i - 1) *
#                 column_num + (column_index + 1))
#     # sns.lineplot(
#     #     data=df_experiment,
#     #     x=hyperparameter,
#     #     y="diversity_coefficient_train_traditional",
#     #     marker='o',
#     #     label=f"Train - traditional ensemble model"
#     # )
#     sns.lineplot(
#         data=df_experiment,
#         x=hyperparameter,
#         y="diversity_coefficient_test_traditional",
#         marker='o',
#         label=f"Test - traditional ensemble model"
#     )
#     plt.title(
#         f"$\\rho$ vs {hyperparameter_text}")
#     plt.xlabel(hyperparameter_text)
#     plt.ylabel("$\\rho$")
#     plt.legend()

# plt.tight_layout()
# ================================== Boundary ====================================
# Static NCL
# ================================== Boundary ====================================
# # Set plot style for nicer plots
# sns.set_theme(style="whitegrid", rc={
#     "font.size": FONT_SIZE,  # Default font size
#     "font.family": FONT_FAMILY,
#     "font.serif": FONT_SERIF,
#     "axes.titlesize": AXES_TITLESIZE,
#     "axes.titleweight": AXES_TITLEWEIGHT,
#     "axes.labelsize": AXES_LABELSIZE,
#     "legend.fontsize": LEGEND_FONTSIZE
# })

# plt.figure(figsize=(16, 24))  # Larger figure for 10 plots

# row_num = len(hyperparameters)
# column_num = 3

# for i, hyperparameter in enumerate(hyperparameters, 1):
#     hyperparameter_text = hyperparameter
#     if hyperparameter == "correlation_penalty_coefficient":
#         hyperparameter_text = "$\\lambda$"
#     elif hyperparameter == "ensemble_size":
#         hyperparameter_text = "ensemble size"
#     elif hyperparameter == "epoch_num":
#         hyperparameter_text = "train epoch"
#     elif hyperparameter == "learning_rate":
#         hyperparameter_text = "learning rate"

#     column_index = 0  # 0-indexed

#     # Average loss
#     plt.subplot(row_num, column_num, (i - 1) *
#                 column_num + (column_index + 1))
#     # sns.lineplot(
#     #     data=df_experiment,
#     #     x=hyperparameter,
#     #     y="average_loss_train_static_ncl",
#     #     marker='o',
#     #     label=f"Train - static NCL ensemble model"
#     # )
#     sns.lineplot(
#         data=df_experiment,
#         x=hyperparameter,
#         y="average_loss_test_static_ncl",
#         marker='o',
#         label=f"Test - static NCL ensemble model"
#     )
#     # sns.lineplot(
#     #     data=df_experiment,
#     #     x=hyperparameter,
#     #     y="average_loss_train_traditional",
#     #     marker='o',
#     #     label=f"Train - traditional ensemble model"
#     # )
#     sns.lineplot(
#         data=df_experiment,
#         x=hyperparameter,
#         y="average_loss_test_traditional",
#         marker='o',
#         label=f"Test - traditional ensemble model"
#     )
#     # sns.lineplot(
#     #     data=df_experiment,
#     #     x=hyperparameter,
#     #     y="average_loss_train_mlp",
#     #     marker='o',
#     #     label=f"Train - MLP model"
#     # )
#     sns.lineplot(
#         data=df_experiment,
#         x=hyperparameter,
#         y="average_loss_test_mlp",
#         marker='o',
#         label=f"Test - MLP model"
#     )
#     plt.title(f"MSE vs {hyperparameter_text}")
#     plt.xlabel(hyperparameter_text)
#     plt.ylabel("MSE")
#     plt.legend()

#     # ===================================================================

#     # Diversity Coefficient
#     column_index = 1  # 0-indexed

#     plt.subplot(row_num, column_num, (i - 1) *
#                 column_num + (column_index + 1))
#     # sns.lineplot(
#     #     data=df_experiment,
#     #     x=hyperparameter,
#     #     y="diversity_coefficient_train_static_ncl",
#     #     marker='o',
#     #     label=f"Train - static NCL ensemble model"
#     # )
#     sns.lineplot(
#         data=df_experiment,
#         x=hyperparameter,
#         y="diversity_coefficient_test_static_ncl",
#         marker='o',
#         label=f"Test - static NCL ensemble model"
#     )
#     # sns.lineplot(
#     #     data=df_experiment,
#     #     x=hyperparameter,
#     #     y="diversity_coefficient_train_traditional",
#     #     marker='o',
#     #     label=f"Train - traditional ensemble model"
#     # )
#     sns.lineplot(
#         data=df_experiment,
#         x=hyperparameter,
#         y="diversity_coefficient_test_traditional",
#         marker='o',
#         label=f"Test - traditional ensemble model"
#     )
#     plt.title(
#         f"$\\rho$ vs {hyperparameter_text}")
#     plt.xlabel(hyperparameter_text)
#     plt.ylabel("$\\rho$")
#     plt.legend()

#     # ===================================================================

#     # Loss against diversity coefficient
#     column_index = 2  # 0-indexed

#     if i == 1:
#         # We only want to plot 1 graph for all hyparameters for this graph
#         # Remember i starts with 1 instead of 0

#         plt.subplot(row_num, column_num, (i - 1) *
#                     column_num + (column_index + 1))
#         # sns.lineplot(
#         #     data=df_experiment,
#         #     x=f"diversity_coefficient_train_static_ncl",
#         #     y="average_loss_train_static_ncl",
#         #     marker='o',
#         #     label=f"Train - static NCL ensemble model"
#         # )
#         sns.lineplot(
#             data=df_experiment,
#             x=f"diversity_coefficient_test_static_ncl",
#             y="average_loss_test_static_ncl",
#             marker='o',
#             label=f"Test - static NCL ensemble model"
#         )
#         # sns.lineplot(
#         #     data=df_experiment,
#         #     x=f"diversity_coefficient_train_traditional",
#         #     y="average_loss_train_traditional",
#         #     marker='o',
#         #     label=f"Train - traditional ensemble model"
#         # )
#         sns.lineplot(
#             data=df_experiment,
#             x=f"diversity_coefficient_test_traditional",
#             y="average_loss_test_traditional",
#             marker='o',
#             label=f"Test - traditional ensemble model"
#         )
#         plt.title(
#             f"MSE vs $\\rho$")
#         plt.xlabel(f"$\\rho$")
#         plt.ylabel("MSE")
#         plt.legend()

#     # ===================================================================

#     # Loss against lambda (fixed hidden node num)
#     column_index = 2  # 0-indexed

#     if i == 2:
#         # We only want to plot 1 graph for all hyparameters for this graph
#         # Remember i starts with 1 instead of 0
#         fixed_hidden_sizes = [6]
#         target_ensemble_sizes = [2, 6, 12]

#         """
#         NOTE: If we read from csv, df_experiment["hidden_size"] will be of type class "str"
#         Deprecated: If we create the csv, df_experiment["hidden_size"] will be of type class "list"
#         NOTE: If we create the csv, df_experiment["hidden_size"] will be of type class "str" as I convert it to string before saving to csv for graph plotting
#         """
#         df_experiment["hidden_size_list"] = df_experiment["hidden_size"].apply(
#             lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
#         df_filtered_hidden_size = df_experiment[df_experiment["hidden_size_list"].apply(
#             lambda x: x == fixed_hidden_sizes
#         )]
#         df_filtered_hidden_size = df_filtered_hidden_size[df_filtered_hidden_size["ensemble_size"].apply(
#             lambda x: x in target_ensemble_sizes
#         )]

#         plt.subplot(row_num, column_num, (i - 1) *
#                     column_num + (column_index + 1))
#         sns.lineplot(
#             data=df_filtered_hidden_size,
#             x="correlation_penalty_coefficient",
#             y="average_loss_test_static_ncl",
#             hue="ensemble_size",
#             marker='o',
#             palette="tab10"
#         )
#         sns.lineplot(
#             data=df_experiment,
#             x="correlation_penalty_coefficient",
#             y="average_loss_test_mlp",
#             marker='o',
#             label=f"Test - MLP model"
#         )
#         plt.title(
#             f"MSE vs $\\lambda$ \n({sum(fixed_hidden_sizes)} hidden nodes per base learner)")
#         plt.xlabel(f"$\\lambda$")
#         plt.ylabel("MSE")
#         plt.legend()

#     # ===================================================================

#     # Loss against lambda (fixed base learner num)
#     column_index = 2  # 0-indexed

#     if i == 3:
#         # We only want to plot 1 graph for all hyparameters for this graph
#         # Remember i starts with 1 instead of 0
#         fixed_ensemble_size = 6
#         target_hidden_sizes = [[2], [6], [12]]

#         df_filtered_ensemble_size = df_experiment[df_experiment["ensemble_size"].apply(
#             lambda x: x == fixed_ensemble_size
#         )]
#         df_filtered_ensemble_size = df_filtered_ensemble_size[df_filtered_ensemble_size["hidden_size_list"].apply(
#             lambda x: x in target_hidden_sizes
#         )]

#         plt.subplot(row_num, column_num, (i - 1) *
#                     column_num + (column_index + 1))
#         sns.lineplot(
#             data=df_filtered_ensemble_size,
#             x="correlation_penalty_coefficient",
#             y="average_loss_test_static_ncl",
#             hue="hidden_size",
#             marker='o',
#             palette="tab10"
#         )
#         sns.lineplot(
#             data=df_experiment,
#             x="correlation_penalty_coefficient",
#             y="average_loss_test_mlp",
#             marker='o',
#             label=f"Test - MLP model"
#         )
#         plt.title(
#             f"MSE vs $\\lambda$ \n({fixed_ensemble_size} base learners per ensemble model)")
#         plt.xlabel(f"$\\lambda$")
#         plt.ylabel("MSE")
#         plt.legend()

#     # ===================================================================

#     # Diversity coefficient against lambda (fixed hidden node num)
#     column_index = 2  # 0-indexed

#     if i == 4:
#         # We only want to plot 1 graph for all hyparameters for this graph
#         # Remember i starts with 1 instead of 0
#         fixed_hidden_sizes = [6]
#         target_ensemble_sizes = [2, 6, 12]

#         """
#         NOTE: If we read from csv, df_experiment["hidden_size"] will be of type class "str"
#         Deprecated: If we create the csv, df_experiment["hidden_size"] will be of type class "list"
#         NOTE: If we create the csv, df_experiment["hidden_size"] will be of type class "str" as I convert it to string before saving to csv for graph plotting
#         """
#         df_experiment["hidden_size_list"] = df_experiment["hidden_size"].apply(
#             lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
#         df_filtered_hidden_size = df_experiment[df_experiment["hidden_size_list"].apply(
#             lambda x: x == fixed_hidden_sizes
#         )]
#         df_filtered_hidden_size = df_filtered_hidden_size[df_filtered_hidden_size["ensemble_size"].apply(
#             lambda x: x in target_ensemble_sizes
#         )]

#         plt.subplot(row_num, column_num, (i - 1) *
#                     column_num + (column_index + 1))
#         sns.lineplot(
#             data=df_filtered_hidden_size,
#             x="correlation_penalty_coefficient",
#             y="diversity_coefficient_test_static_ncl",
#             hue="ensemble_size",
#             marker='o',
#             palette="tab10"
#         )
#         plt.title(
#             f"$\\rho$ vs $\\lambda$ \n({sum(fixed_hidden_sizes)} hidden nodes per base learner)")
#         plt.xlabel(f"$\\lambda$")
#         plt.ylabel("$\\rho$")
#         plt.legend()

#     # ===================================================================

#     # Loss against lambda (fixed base learner num)
#     column_index = 2  # 0-indexed

#     if i == 5:
#         # We only want to plot 1 graph for all hyparameters for this graph
#         # Remember i starts with 1 instead of 0
#         fixed_ensemble_size = 6
#         target_hidden_sizes = [[2], [6], [12]]

#         df_filtered_ensemble_size = df_experiment[df_experiment["ensemble_size"].apply(
#             lambda x: x == fixed_ensemble_size
#         )]
#         df_filtered_ensemble_size = df_filtered_ensemble_size[df_filtered_ensemble_size["hidden_size_list"].apply(
#             lambda x: x in target_hidden_sizes
#         )]

#         plt.subplot(row_num, column_num, (i - 1) *
#                     column_num + (column_index + 1))
#         sns.lineplot(
#             data=df_filtered_ensemble_size,
#             x="correlation_penalty_coefficient",
#             y="diversity_coefficient_test_static_ncl",
#             hue="hidden_size",
#             marker='o',
#             palette="tab10"
#         )
#         plt.title(
#             f"$\\rho$ vs $\\lambda$ \n({fixed_ensemble_size} base learners per ensemble model)")
#         plt.xlabel(f"$\\lambda$")
#         plt.ylabel("$\\rho$")
#         plt.legend()

# plt.tight_layout()
# ================================== Boundary ====================================


def csv_to_png(csv_path: str, repository_experiment_img_path: str, img_name: str, image_resolution=300) -> None:
    """
    Given a csv file, generate the image file of the graph

    :param: csv_path: Path to the csv file
    :param: repository_experiment_img_path: Path to the directory img/ containing all experiment images (in this case, only 1 image)
    :param: img_name: Name of the image file (without file extension) to be stored
    :param: image_resolution: Resolution of the image of experiment graph
    """
    # TODO: Update me
    hyperparameters = ["correlation_penalty_coefficient",
                       "ensemble_size", "epoch_num", "learning_rate", "hidden_size"]
    df_experiment = pd.read_csv(csv_path)

    def generate_graph():
        """
        TODO: Change the content of this function to match your experiment
        """
        # Set plot style for nicer plots
        sns.set_theme(style="whitegrid", rc={
            "font.size": FONT_SIZE,  # Default font size
            "font.family": FONT_FAMILY,
            "font.serif": FONT_SERIF,
            "axes.titlesize": AXES_TITLESIZE,
            "axes.titleweight": AXES_TITLEWEIGHT,
            "axes.labelsize": AXES_LABELSIZE,
            "legend.fontsize": LEGEND_FONTSIZE
        })

        plt.figure(figsize=(16, 24))  # Larger figure for 10 plots

        row_num = len(hyperparameters)
        column_num = 3

        for i, hyperparameter in enumerate(hyperparameters, 1):
            hyperparameter_text = hyperparameter
            if hyperparameter == "correlation_penalty_coefficient":
                hyperparameter_text = "$\\lambda$"
            elif hyperparameter == "ensemble_size":
                hyperparameter_text = "ensemble size"
            elif hyperparameter == "epoch_num":
                hyperparameter_text = "train epoch"
            elif hyperparameter == "learning_rate":
                hyperparameter_text = "learning rate"

            column_index = 0  # 0-indexed

            # Average loss
            plt.subplot(row_num, column_num, (i - 1) *
                        column_num + (column_index + 1))
            # sns.lineplot(
            #     data=df_experiment,
            #     x=hyperparameter,
            #     y="average_loss_train_static_ncl",
            #     marker='o',
            #     label=f"Train - static NCL ensemble model"
            # )
            sns.lineplot(
                data=df_experiment,
                x=hyperparameter,
                y="average_loss_test_static_ncl",
                marker='o',
                label=f"Test - static NCL ensemble model"
            )
            # sns.lineplot(
            #     data=df_experiment,
            #     x=hyperparameter,
            #     y="average_loss_train_traditional",
            #     marker='o',
            #     label=f"Train - traditional ensemble model"
            # )
            sns.lineplot(
                data=df_experiment,
                x=hyperparameter,
                y="average_loss_test_traditional",
                marker='o',
                label=f"Test - traditional ensemble model"
            )
            # sns.lineplot(
            #     data=df_experiment,
            #     x=hyperparameter,
            #     y="average_loss_train_mlp",
            #     marker='o',
            #     label=f"Train - MLP model"
            # )
            sns.lineplot(
                data=df_experiment,
                x=hyperparameter,
                y="average_loss_test_mlp",
                marker='o',
                label=f"Test - MLP model"
            )
            plt.title(f"MSE vs {hyperparameter_text}")
            plt.xlabel(hyperparameter_text)
            plt.ylabel("MSE")
            plt.legend()

            # ===================================================================

            # Diversity Coefficient
            column_index = 1  # 0-indexed

            plt.subplot(row_num, column_num, (i - 1) *
                        column_num + (column_index + 1))
            # sns.lineplot(
            #     data=df_experiment,
            #     x=hyperparameter,
            #     y="diversity_coefficient_train_static_ncl",
            #     marker='o',
            #     label=f"Train - static NCL ensemble model"
            # )
            sns.lineplot(
                data=df_experiment,
                x=hyperparameter,
                y="diversity_coefficient_test_static_ncl",
                marker='o',
                label=f"Test - static NCL ensemble model"
            )
            # sns.lineplot(
            #     data=df_experiment,
            #     x=hyperparameter,
            #     y="diversity_coefficient_train_traditional",
            #     marker='o',
            #     label=f"Train - traditional ensemble model"
            # )
            sns.lineplot(
                data=df_experiment,
                x=hyperparameter,
                y="diversity_coefficient_test_traditional",
                marker='o',
                label=f"Test - traditional ensemble model"
            )
            plt.title(
                f"$\\rho$ vs {hyperparameter_text}")
            plt.xlabel(hyperparameter_text)
            plt.ylabel("$\\rho$")
            plt.legend()

            # ===================================================================

            # Loss against diversity coefficient
            column_index = 2  # 0-indexed

            if i == 1:
                # We only want to plot 1 graph for all hyparameters for this graph
                # Remember i starts with 1 instead of 0

                plt.subplot(row_num, column_num, (i - 1) *
                            column_num + (column_index + 1))
                # sns.lineplot(
                #     data=df_experiment,
                #     x=f"diversity_coefficient_train_static_ncl",
                #     y="average_loss_train_static_ncl",
                #     marker='o',
                #     label=f"Train - static NCL ensemble model"
                # )
                sns.lineplot(
                    data=df_experiment,
                    x=f"diversity_coefficient_test_static_ncl",
                    y="average_loss_test_static_ncl",
                    marker='o',
                    label=f"Test - static NCL ensemble model"
                )
                # sns.lineplot(
                #     data=df_experiment,
                #     x=f"diversity_coefficient_train_traditional",
                #     y="average_loss_train_traditional",
                #     marker='o',
                #     label=f"Train - traditional ensemble model"
                # )
                sns.lineplot(
                    data=df_experiment,
                    x=f"diversity_coefficient_test_traditional",
                    y="average_loss_test_traditional",
                    marker='o',
                    label=f"Test - traditional ensemble model"
                )
                plt.title(
                    f"MSE vs $\\rho$")
                plt.xlabel(f"$\\rho$")
                plt.ylabel("MSE")
                plt.legend()

            # ===================================================================

            # Loss against lambda (fixed hidden node num)
            column_index = 2  # 0-indexed

            if i == 2:
                # We only want to plot 1 graph for all hyparameters for this graph
                # Remember i starts with 1 instead of 0
                # TODO: Update me
                fixed_hidden_sizes = [2, 2, 2]
                target_ensemble_sizes = [2, 6, 12]

                """
                NOTE: If we read from csv, df_experiment["hidden_size"] will be of type class "str"
                Deprecated: If we create the csv, df_experiment["hidden_size"] will be of type class "list"
                NOTE: If we create the csv, df_experiment["hidden_size"] will be of type class "str" as I convert it to string before saving to csv for graph plotting
                """
                df_experiment["hidden_size_list"] = df_experiment["hidden_size"].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                df_filtered_hidden_size = df_experiment[df_experiment["hidden_size_list"].apply(
                    lambda x: x == fixed_hidden_sizes
                )]
                df_filtered_hidden_size = df_filtered_hidden_size[df_filtered_hidden_size["ensemble_size"].apply(
                    lambda x: x in target_ensemble_sizes
                )]

                plt.subplot(row_num, column_num, (i - 1) *
                            column_num + (column_index + 1))
                sns.lineplot(
                    data=df_filtered_hidden_size,
                    x="correlation_penalty_coefficient",
                    y="average_loss_test_static_ncl",
                    hue="ensemble_size",
                    marker='o',
                    palette="tab10"
                )
                sns.lineplot(
                    data=df_experiment,
                    x="correlation_penalty_coefficient",
                    y="average_loss_test_mlp",
                    marker='o',
                    label=f"Test - MLP model"
                )
                plt.title(
                    f"MSE vs $\\lambda$ \n({sum(fixed_hidden_sizes)} hidden nodes per base learner)")
                plt.xlabel(f"$\\lambda$")
                plt.ylabel("MSE")
                plt.legend()

            # ===================================================================

            # Loss against lambda (fixed base learner num)
            column_index = 2  # 0-indexed

            if i == 3:
                # We only want to plot 1 graph for all hyparameters for this graph
                # Remember i starts with 1 instead of 0
                # TODO: Update me
                fixed_ensemble_size = 6
                target_hidden_sizes = [[2], [2, 2, 2]]

                df_filtered_ensemble_size = df_experiment[df_experiment["ensemble_size"].apply(
                    lambda x: x == fixed_ensemble_size
                )]
                df_filtered_ensemble_size = df_filtered_ensemble_size[df_filtered_ensemble_size["hidden_size_list"].apply(
                    lambda x: x in target_hidden_sizes
                )]

                plt.subplot(row_num, column_num, (i - 1) *
                            column_num + (column_index + 1))
                sns.lineplot(
                    data=df_filtered_ensemble_size,
                    x="correlation_penalty_coefficient",
                    y="average_loss_test_static_ncl",
                    hue="hidden_size",
                    marker='o',
                    palette="tab10"
                )
                sns.lineplot(
                    data=df_experiment,
                    x="correlation_penalty_coefficient",
                    y="average_loss_test_mlp",
                    marker='o',
                    label=f"Test - MLP model"
                )
                plt.title(
                    f"MSE vs $\\lambda$ \n({fixed_ensemble_size} base learners per ensemble model)")
                plt.xlabel(f"$\\lambda$")
                plt.ylabel("MSE")
                plt.legend()

            # ===================================================================

            # Diversity coefficient against lambda (fixed hidden node num)
            column_index = 2  # 0-indexed

            if i == 4:
                # We only want to plot 1 graph for all hyparameters for this graph
                # Remember i starts with 1 instead of 0
                fixed_hidden_sizes = [2, 2, 2]
                target_ensemble_sizes = [2, 6, 12]

                """
                NOTE: If we read from csv, df_experiment["hidden_size"] will be of type class "str"
                Deprecated: If we create the csv, df_experiment["hidden_size"] will be of type class "list"
                NOTE: If we create the csv, df_experiment["hidden_size"] will be of type class "str" as I convert it to string before saving to csv for graph plotting
                """
                df_experiment["hidden_size_list"] = df_experiment["hidden_size"].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                df_filtered_hidden_size = df_experiment[df_experiment["hidden_size_list"].apply(
                    lambda x: x == fixed_hidden_sizes
                )]
                df_filtered_hidden_size = df_filtered_hidden_size[df_filtered_hidden_size["ensemble_size"].apply(
                    lambda x: x in target_ensemble_sizes
                )]

                plt.subplot(row_num, column_num, (i - 1) *
                            column_num + (column_index + 1))
                sns.lineplot(
                    data=df_filtered_hidden_size,
                    x="correlation_penalty_coefficient",
                    y="diversity_coefficient_test_static_ncl",
                    hue="ensemble_size",
                    marker='o',
                    palette="tab10"
                )
                plt.title(
                    f"$\\rho$ vs $\\lambda$ \n({sum(fixed_hidden_sizes)} hidden nodes per base learner)")
                plt.xlabel(f"$\\lambda$")
                plt.ylabel("$\\rho$")
                plt.legend()

            # ===================================================================

            # Loss against lambda (fixed base learner num)
            column_index = 2  # 0-indexed

            if i == 5:
                # We only want to plot 1 graph for all hyparameters for this graph
                # Remember i starts with 1 instead of 0
                fixed_ensemble_size = 6
                target_hidden_sizes = [[2], [2, 2, 2]]

                df_filtered_ensemble_size = df_experiment[df_experiment["ensemble_size"].apply(
                    lambda x: x == fixed_ensemble_size
                )]
                df_filtered_ensemble_size = df_filtered_ensemble_size[df_filtered_ensemble_size["hidden_size_list"].apply(
                    lambda x: x in target_hidden_sizes
                )]

                plt.subplot(row_num, column_num, (i - 1) *
                            column_num + (column_index + 1))
                sns.lineplot(
                    data=df_filtered_ensemble_size,
                    x="correlation_penalty_coefficient",
                    y="diversity_coefficient_test_static_ncl",
                    hue="hidden_size",
                    marker='o',
                    palette="tab10"
                )
                plt.title(
                    f"$\\rho$ vs $\\lambda$ \n({fixed_ensemble_size} base learners per ensemble model)")
                plt.xlabel(f"$\\lambda$")
                plt.ylabel("$\\rho$")
                plt.legend()

        plt.tight_layout()
        # ================================== Don't update below code ====================================

        assert repository_experiment_img_path and img_name, f"Invalid final image path: {repository_experiment_img_path}/{img_name}.png"

        # Save graph as image if applicable
        plt.savefig(f"{repository_experiment_img_path}/{img_name}.png",
                    dpi=image_resolution)

    generate_graph()

    # Graph is saved


if __name__ == "__main__":
    csv_to_png(csv_path=CSV_PATH, repository_experiment_img_path=REPOSITORY_EXPERIMENT_IMAGE_PATH,
               img_name=IMG_NAME, image_resolution=IMAGE_RESOLUTION)

    print(f"Image of experiment graph is generated successfully ...")
