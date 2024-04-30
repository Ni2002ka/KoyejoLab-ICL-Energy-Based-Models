import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os

import pandas as pd
import seaborn as sns
import wandb


sns.set_style("whitegrid")


def plot_dataset_2D_energy_landscapes(
    real_data: np.ndarray,
    meshgrid: np.ndarray,
    energy_landscape: np.ndarray,
    wandb_logger=None,
    wandb_key: str = "energy_of_x",
    n_batch_elements_to_plot: int = 1,
    n_steps_between_time_indices: int = 5,
):
    # batch_size, max_seq_len, data_dim = real_data.shape
    batch_size, total_meshgrid_points, max_seq_len, energy_dim = energy_landscape.shape
    assert n_batch_elements_to_plot <= batch_size
    meshgrid_points_per_slice = int(np.sqrt(total_meshgrid_points))
    meshgrid_max_abs_val = 1.1 * np.max(np.abs(meshgrid))

    n_subfigures = int(np.ceil(max_seq_len / n_steps_between_time_indices))
    for batch_idx in range(n_batch_elements_to_plot):
        plt.close()
        fig, axes = plt.subplots(
            nrows=1,
            ncols=n_subfigures,
            figsize=(4 * n_subfigures, 4),
            sharey=True,
            sharex=True,
            constrained_layout=True,
        )
        mappable = None
        for ax_idx, seq_idx in enumerate(
            range(0, max_seq_len, n_steps_between_time_indices)
        ):
            ax = axes[ax_idx]
            mappable = ax.contourf(
                meshgrid[:, :, 0],
                meshgrid[:, :, 1],
                energy_landscape[batch_idx, :, seq_idx, 0].reshape(
                    meshgrid_points_per_slice, meshgrid_points_per_slice
                ),
                levels=100,
                cmap="coolwarm",
                norm=colors.SymLogNorm(
                    linthresh=0.03,
                    vmin=np.min(energy_landscape[batch_idx]),
                    vmax=np.max(energy_landscape[batch_idx]),
                ),
            )
            ax.scatter(
                x=real_data[batch_idx, :seq_idx, 0],
                y=real_data[batch_idx, :seq_idx, 1],
                label="Real Data",
                color="black",
                marker="x",
            )
            ax.set_xlabel(r"$x_1$")
            ax.set_ylabel(r"$x_2$")
            ax.set_xlim(-meshgrid_max_abs_val, meshgrid_max_abs_val)
            ax.set_ylim(-meshgrid_max_abs_val, meshgrid_max_abs_val)
            ax.set_title(r"$E(x|D)$" + f" at n={seq_idx}")
            ax.set_aspect("equal")
        if mappable is not None:
            fig.colorbar(mappable, ax=axes, label="Energy", orientation="vertical")
        if wandb_logger is not None:
            wandb_logger.log_image(
                key=wandb_key + "_batch_idx=" + str(batch_idx),
                images=[wandb.Image(fig)],
            )
        # plt.show()


def plot_dataset_2D_vector_fields(
    meshgrid: np.ndarray,
    negative_grad_energy_wrt_x: np.ndarray,
    step: int = 5,
    wandb_logger=None,
    wandb_key: str = "grad_energy_wrt_x",
    n_batch_elements_to_plot: int = 1,
    n_steps_between_time_indices: int = 5,
):
    # Assuming gradients shape is (batch_size, total_meshgrid_points, max_seq_len, 2)
    batch_size, total_meshgrid_points, max_seq_len, _ = negative_grad_energy_wrt_x.shape
    assert n_batch_elements_to_plot <= batch_size
    meshgrid_points_per_slice = int(np.sqrt(total_meshgrid_points))
    meshgrid_max_abs_val = 1.1 * np.max(np.abs(meshgrid))

    n_subfigures = int(np.ceil(max_seq_len / n_steps_between_time_indices))
    for batch_idx in range(n_batch_elements_to_plot):
        plt.close()
        fig, axes = plt.subplots(
            nrows=1,
            ncols=n_subfigures,
            figsize=(4 * n_subfigures, 4),
            sharey=True,
            sharex=True,
            constrained_layout=True,
        )
        for ax_idx, seq_idx in enumerate(range(0, max_seq_len, step)):
            ax = axes[ax_idx]
            ax.streamplot(
                meshgrid[:, :, 0],
                meshgrid[:, :, 1],
                negative_grad_energy_wrt_x[batch_idx, :, seq_idx, 0].reshape(
                    meshgrid_points_per_slice, meshgrid_points_per_slice
                ),
                negative_grad_energy_wrt_x[batch_idx, :, seq_idx, 1].reshape(
                    meshgrid_points_per_slice, meshgrid_points_per_slice
                ),
                color="black",
                density=2.0,
            )
            ax.set_xlim(-meshgrid_max_abs_val, meshgrid_max_abs_val)
            ax.set_ylim(-meshgrid_max_abs_val, meshgrid_max_abs_val)
            ax.set_xlabel(r"$x_1$")
            ax.set_ylabel(r"$x_2$")
            ax.set_title(r"$-\nabla_{x} E(x|D)$" + f" at seq_idx={seq_idx}")
            ax.set_aspect("equal")

        if wandb_logger is not None:
            wandb_logger.log_image(
                key=wandb_key + "_batch_idx=" + str(batch_idx),
                images=[wandb.Image(fig)],
            )
        # plt.show()


def plot_dataset_2D_real_data_and_sampled_data(
    real_data: np.ndarray,
    initial_sampled_data: np.ndarray,
    final_sampled_data: np.ndarray,
    wandb_logger=None,
    wandb_key: str = "real_data_and_sampled_data",
):
    # Shapes:
    # real_data: (batch_size, max_seq_len, data_dim)
    # initial_sampled_data: (batch_size, ratio of confabulations, max_seq_len, data_dim)
    # final_sampled_data: (batch_size, ratio of confabulations, max_seq_len, data_dim)
    plt.close()
    plt.figure(figsize=(8, 8))
    plt.scatter(
        x=real_data[0, :, 0],
        y=real_data[0, :, 1],
        label="Real Data",
        color="black",
        marker="x",
    )
    plt.scatter(
        x=initial_sampled_data[0, :, :, 0].flatten(),
        y=initial_sampled_data[0, :, :, 1].flatten(),
        label="Initial Sampled Data",
    )
    plt.scatter(
        x=final_sampled_data[0, :, :, 0].flatten(),
        y=final_sampled_data[0, :, :, 1].flatten(),
        label="Final Sampled Data",
    )
    # Draw arrow connecting initial sampled data to final sampled data.
    for confab_idx in range(initial_sampled_data.shape[1]):
        for particle_idx in range(initial_sampled_data.shape[2]):
            plt.arrow(
                x=initial_sampled_data[0, confab_idx, particle_idx, 0],
                y=initial_sampled_data[0, confab_idx, particle_idx, 1],
                dx=final_sampled_data[0, confab_idx, particle_idx, 0]
                - initial_sampled_data[0, confab_idx, particle_idx, 0],
                dy=final_sampled_data[0, confab_idx, particle_idx, 1]
                - initial_sampled_data[0, confab_idx, particle_idx, 1],
                color="black",
                alpha=0.5,
                width=0.01,
            )
    plt.xlim(-10.0, 10.0)
    plt.ylim(-10.0, 10.0)
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.legend()
    # https://docs.wandb.ai/guides/track/log/plots#matplotlib-and-plotly-plots
    fig = plt.gcf()
    if wandb_logger is not None:
        wandb_logger.log_image(
            key=wandb_key,
            images=[wandb.Image(fig)],
        )

    # plt.show()


def plot_linear_regression_in_context_error_vs_n_in_context_examples(
    squared_norm_diff_final_sampled_data: np.ndarray,
    wandb_logger=None,
    wandb_key: str = "squared_norm_diff_final_sampled_data",
):
    plt.close()

    # Shape: (batch size * confabulations, max seq length)
    df = pd.DataFrame(
        squared_norm_diff_final_sampled_data.reshape(
            -1, squared_norm_diff_final_sampled_data.shape[2]  # Max sequence length.
        ).T,
    )
    df["seq_idx"] = np.arange(squared_norm_diff_final_sampled_data.shape[2])

    tall_df = pd.melt(
        df,
        id_vars=["seq_idx"],
        var_name="confab_idx",
        value_name="squared_norm_diff_final_sampled_data",
    )

    sns.lineplot(
        data=tall_df,
        x="seq_idx",
        y="squared_norm_diff_final_sampled_data",
    )
    plt.xlabel("Num of In-Context Examples")
    plt.ylabel(r"$||y - \hat{y} ||^2$")
    plt.ylim(bottom=0.0)
    # plt.show()

    # https://docs.wandb.ai/guides/track/log/plots#matplotlib-and-plotly-plots
    fig = plt.gcf()
    if wandb_logger is not None:
        wandb_logger.log_image(
            key=wandb_key,
            images=[wandb.Image(fig)],
        )

    # plt.show()
