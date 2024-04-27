import lightning
import lightning.pytorch as pl
import numpy as np
import torch
import torch.distributions
import torch.nn
import torch.nn.functional
import torch.utils.data
from transformers import GPT2Model, GPT2Config
from typing import Any, Callable, Dict, List, Tuple, Union
import wandb
import src.data
import src.plot
import pdb

torch.autograd.set_detect_anomaly(True)


class InContextLearningEnergyBasedModelEvaluationCallback(lightning.Callback):
    # The normal Lightning validation step blocks gradients, even with respect to inputs.
    # Consequently, we'll need to implement our own validation step.
    def __init__(self, wandb_config: Dict[str, Any]):
        super().__init__()
        self.wandb_config = wandb_config
        self.val_dataset = {}
        self.val_dataloader = None

        if "n_workers" not in self.wandb_config:
            # n_workers = max(4, os.cpu_count() // 4)  # heuristic
            self.n_workers = 1
        else:
            self.n_workers = self.wandb_config["n_workers"]

        # TODO: remove this for sweeps
        self.n_meshgrid_points_in_1D_slice = wandb_config["eval_kwargs"][
            "n_meshgrid_points_in_1D_slice"
        ]

    def on_train_epoch_start(self, trainer, pl_module):
        # Only perform evaluation every n epochs.
        if (
            trainer.current_epoch % pl_module.wandb_config["check_val_every_n_epoch"]
            != 0
        ):
            return

        keys_to_copy_to_mock_wandb_config = ["mcmc_kwargs", "batch_size_val"]
        for eval_name, eval_data_kwargs in self.wandb_config["eval_datasets"].items():
            mock_wandb_config = {
                k: self.wandb_config[k] for k in keys_to_copy_to_mock_wandb_config
            }
            mock_wandb_config["dataset_kwargs"] = eval_data_kwargs

            self.val_dataset = src.data.create_dataset(
                wandb_config=mock_wandb_config,
                split="val",
            )

            self.val_dataloader = torch.utils.data.DataLoader(
                dataset=self.val_dataset,
                batch_size=self.wandb_config["batch_size_val"],
                num_workers=self.n_workers,
                pin_memory=False,
                shuffle=False,
            )

            # Create 2D mesh grid to evaluate energy on.
            # Shape: (n meshgrid points, n meshgrid points, 2)
            meshgrid = torch.stack(
                torch.meshgrid(
                    [
                        torch.linspace(
                            start=-10.0,
                            end=10.0,
                            steps=self.n_meshgrid_points_in_1D_slice,
                        ),
                        torch.linspace(
                            start=-10.0,
                            end=10.0,
                            steps=self.n_meshgrid_points_in_1D_slice,
                        ),
                    ],
                    indexing="xy",  # Necessary to match Numpy indexing.
                ),
                dim=-1,
            ).to(pl_module.device)
            meshgrid_numpy = meshgrid.cpu().numpy()

            for batch_idx, batch in enumerate(self.val_dataloader):
                real_data = batch["real_data"].to(pl_module.device)
                # Shape: (batch size, n in-context examples, data shape...)
                real_data_numpy = batch["real_data"].cpu().numpy()
                # Shape: (batch size, ratio_of_confabulated_samples_to_real_samples, n in-context examples, data shape...)
                initial_sampled_data = batch["initial_sampled_data"].to(
                    pl_module.device
                )
                sampling_update_mask = batch["sampling_update_mask"].to(
                    pl_module.device
                )

                # ground_truth_energy = EnergyFunctionMixtureOfGaussians(
                #     means=batch["means"][0].to(pl_module.device),
                #     # covariances=batch["covariances"][0].to(pl_module.device),
                #     wandb_config=pl_module.wandb_config,
                # )

                # ############################################################
                # Evaluate and plot energy on mesh grid at each time step.
                # ############################################################
                energy_meshgrid_results_dict = pl_module.measure_energy_landscape(
                    real_data=real_data,
                    meshgrid=meshgrid.reshape(
                        -1, 2
                    ),  # Shape: (n meshgrid points ** 2, 2)
                )
                src.plot.plot_dataset_2D_energy_landscapes(
                    real_data=real_data_numpy,
                    meshgrid=meshgrid_numpy,
                    energy_landscape=energy_meshgrid_results_dict["energy_landscape"]
                    .detach()
                    .cpu()
                    .numpy(),
                    wandb_logger=pl_module.wandb_logger,
                    wandb_key=eval_name + "_energy_of_x",
                )

                # ############################################################
                # Evaluate and plot gradient flow on mesh grid at each time step.
                # ############################################################
                src.plot.plot_dataset_2D_vector_fields(
                    meshgrid=meshgrid_numpy,
                    negative_grad_energy_wrt_x=energy_meshgrid_results_dict[
                        "neg_grad_energy_wrt_meshgrid"
                    ]
                    .detach()
                    .cpu()
                    .numpy(),
                    wandb_logger=pl_module.wandb_logger,
                    wandb_key=eval_name + "_grad_energy_wrt_x",
                )

                # ############################################################
                # Sample new data initialized using randomly sampled noise.
                # ############################################################
                # Shape: (batch size, n in-context examples, data shape...)
                if self.wandb_config["mcmc_kwargs"]["replay_buffer"]:
                    # TODO: Take replay buffer from Yilun's code
                    # https://github.com/yilundu/improved_contrastive_divergence/blob/master/train.py#L382-L392
                    raise NotImplementedError("Replay buffer not implemented yet.")

                if self.wandb_config["mcmc_kwargs"]["algorithm"] == "langevin_mcmc":
                    transformer_sampled_data_results_dict = (
                        pl_module.sample_data_with_langevin_mcmc(
                            real_data=real_data,
                            initial_sampled_data=initial_sampled_data,
                            sampling_update_mask=sampling_update_mask,
                            noise_scale=0.1,
                        )
                    )

                    true_sampled_data_results_dict = {"final_sampled_data": real_data}

                elif (
                    self.wandb_config["mcmc_kwargs"]["algorithm"] == "hamiltonian_mcmc"
                ):
                    transformer_sampled_data_results_dict = (
                        pl_module.sample_data_with_hamiltonian_mcmc(
                            real_data=real_data,
                            initial_sampled_data=initial_sampled_data,
                        )
                    )
                else:
                    raise ValueError("Invalid MCMC algorithm.")

                final_sampled_data = (
                    transformer_sampled_data_results_dict["final_sampled_data"]
                    .detach()
                    .cpu()
                    .numpy()
                )

                # We only calculate the MSE for the y coordinate
                # Shape: (batch size, ratio_of_confabulated_samples_to_real_samples, max seq length, )
                diff_final_sampled_data = torch.subtract(
                    transformer_sampled_data_results_dict["final_sampled_data"][
                        :, :, :, -1
                    ],
                    true_sampled_data_results_dict["final_sampled_data"][
                        :, :, -1
                    ].unsqueeze(0),
                )

                # Shape: (batch size, ratio_of_confabulated_samples_to_real_samples, max seq length)
                squared_norm_diff_final_sampled_data = torch.square(
                    diff_final_sampled_data
                )

                eval_log_dict = {
                    f"test_{eval_name}/mse_transformers_samples_vs_true_samples_in_context={seq_idx}": torch.mean(
                        squared_norm_diff_final_sampled_data[:, :, seq_idx]
                    )
                    for seq_idx in range(diff_final_sampled_data.shape[2])
                }

                wandb.log(eval_log_dict)

                src.plot.plot_in_context_error_vs_n_in_context_examples(
                    squared_norm_diff_final_sampled_data=squared_norm_diff_final_sampled_data.detach()
                    .cpu()
                    .numpy(),
                    wandb_logger=pl_module.wandb_logger,
                    wandb_key=eval_name + "_in_context_error_vs_n_in_context_examples",
                )

                # src.plot.plot_in_context_error_vs_n_in_context_examples(
                #     squared_norm_diff_final_sampled_data=torch.norm(
                #         true_sampled_data_results_dict["final_sampled_data"].detach(),
                #         dim=-1,
                #     )
                #     .cpu()
                #     .numpy(),
                #     wandb_logger=pl_module.wandb_logger,
                #     wandb_key=eval_name
                #     + "_in_context_error_vs_n_in_context_examples_for_true_energy",
                # )

                src.plot.plot_dataset_2D_real_data_and_sampled_data(
                    real_data=real_data.cpu().numpy(),
                    initial_sampled_data=initial_sampled_data.detach().cpu().numpy(),
                    final_sampled_data=final_sampled_data,
                    wandb_logger=pl_module.wandb_logger,
                    wandb_key=eval_name + "_real_data_and_sampled_data",
                )


class InContextLearningEnergyBasedModelSystem(pl.LightningModule):
    def __init__(self, wandb_config: Dict, wandb_logger):
        super().__init__()

        # Should save hyperparameters to checkpoint.
        # https://pytorch-lightning.readthedocs.io/en/stable/common/checkpointing_basic.html
        # self.save_hyperparameters()

        self.wandb_config = wandb_config
        self.wandb_logger = wandb_logger

        if self.wandb_config["mcmc_kwargs"]["replay_buffer"]:
            raise NotImplementedError("Replay buffer not implemented yet.")
        self.transformer_ebm = EnergyBasedTransformerModel(wandb_config=wandb_config)

    def configure_optimizers(self) -> Dict:
        # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers

        # TODO: Maybe add SWA
        # https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.StochasticWeightAveraging.html#pytorch_lightning.callbacks.StochasticWeightAveraging
        if self.wandb_config["optimizer"] == "adadelta":
            optimizer = torch.optim.Adadelta(
                self.parameters(),
                lr=self.wandb_config["learning_rate"],
                weight_decay=self.wandb_config["weight_decay"],
            )
        elif self.wandb_config["optimizer"] == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.wandb_config["learning_rate"],
                weight_decay=self.wandb_config["weight_decay"],
                eps=1e-4,  # https://stackoverflow.com/a/42420014/4570472
            )
        elif self.wandb_config["optimizer"] == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.wandb_config["learning_rate"],
                weight_decay=self.wandb_config["weight_decay"],
                eps=1e-4,  # https://stackoverflow.com/a/42420014/4570472
            )
        elif self.wandb_config["optimizer"] == "rmsprop":
            optimizer = torch.optim.RMSprop(
                self.parameters(),
                lr=self.wandb_config["learning_rate"],
                weight_decay=self.wandb_config["weight_decay"],
                momentum=0.9,
                eps=1e-4,
            )
        elif self.wandb_config["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.wandb_config["learning_rate"],
                weight_decay=self.wandb_config["weight_decay"],
                momentum=0.9,
            )
        else:
            # TODO: add adafactor https://pytorch-optimizer.readthedocs.io/en/latest/index.html
            raise NotImplementedError(f"{self.wandb_config['optimizer']}")

        optimizer_and_maybe_others_dict = {
            "optimizer": optimizer,
        }

        if self.wandb_config["learning_rate_scheduler"] is None:
            pass
        elif (
            self.wandb_config["learning_rate_scheduler"]
            == "cosine_annealing_warm_restarts"
        ):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=2,
            )
            optimizer_and_maybe_others_dict["lr_scheduler"] = scheduler

        elif (
            self.wandb_config["learning_rate_scheduler"]
            == "linear_warmup_cosine_annealing"
        ):
            from flash.core.optimizers import LinearWarmupCosineAnnealingLR

            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer=optimizer,
                warmup_epochs=1,
                max_epochs=self.wandb_config["n_epochs"],
            )

            optimizer_and_maybe_others_dict["lr_scheduler"] = scheduler

        elif self.wandb_config["learning_rate_scheduler"] == "reduce_lr_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                factor=0.95,
                optimizer=optimizer,
                patience=3,
            )
            optimizer_and_maybe_others_dict["lr_scheduler"] = scheduler
            optimizer_and_maybe_others_dict["monitor"] = "train/loss=total_loss"
        else:
            raise NotImplementedError(f"{self.wandb_config['learning_rate_scheduler']}")

        return optimizer_and_maybe_others_dict

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        # ############################################################
        # Compute energy on real data.
        # ############################################################
        # Shape: (batch size, n in-context examples, data shape...)
        real_data = batch["real_data"]
        forward_results_on_real_data = self.transformer_ebm.forward(data=real_data)
        # Shape: (batch size, num. in-context examples, 1)
        energy_on_real_data = forward_results_on_real_data["energy"]

        # ############################################################
        # Sample new data initialized using randomly sampled noise.
        # ############################################################
        # Shape: (batch size, ratio_of_confabulated_samples_to_real_samples, n in-context examples, data shape...)
        initial_sampled_data = batch["initial_sampled_data"]
        sampling_update_mask = batch["sampling_update_mask"]
        # Reshape to merge the first two dimensions.
        (
            batch_size,
            ratio_of_confabulated_samples_to_real_samples,
            max_seq_len,
            data_dim,
        ) = initial_sampled_data.shape

        if self.wandb_config["mcmc_kwargs"]["replay_buffer"]:
            # TODO: Take replay buffer from Yilun's code
            # https://github.com/yilundu/improved_contrastive_divergence/blob/master/train.py#L382-L392
            raise NotImplementedError("Replay buffer not implemented yet.")

        if self.wandb_config["mcmc_kwargs"]["algorithm"] == "langevin_mcmc":
            sampled_data_results_dict = self.sample_data_with_langevin_mcmc(
                real_data=real_data,
                initial_sampled_data=initial_sampled_data,
                sampling_update_mask=sampling_update_mask,
            )
        elif self.wandb_config["mcmc_kwargs"]["algorithm"] == "hamiltonian_mcmc":
            sampled_data_results_dict = self.sample_data_with_hamiltoinan_mcmc(
                real_data=real_data,
                initial_sampled_data=initial_sampled_data,
                sampling_update_mask=sampling_update_mask,
            )
        else:
            raise ValueError("Invalid MCMC algorithm.")

        # ############################################################
        # Compute energy using sampled data.
        # ############################################################
        # Shape: (batch size * ratio_of_confabulated_samples_to_real_samples, n in-context examples, data shape...)
        sampled_data = sampled_data_results_dict["final_sampled_data"].detach()
        energy_on_sampled_data = torch.zeros(
            size=(
                batch_size,
                ratio_of_confabulated_samples_to_real_samples,
                max_seq_len,
                1,
            ),
            device=self.device,
            requires_grad=True,
        )
        (
            batch_size,
            n_confabulated_samples,
            max_seq_len,
            data_dim,
        ) = sampled_data.shape  # equivalently, real_data.shape
        assert max_seq_len == real_data.shape[1]
        for seq_idx in range(max_seq_len):
            real_data_up_to_seq_idx = torch.clone(real_data[:, : seq_idx + 1, :])
            # TODO: Figure out way to vectorize this with transformer.
            for confab_idx in range(n_confabulated_samples):
                real_data_up_to_seq_idx[:, -1, :] = sampled_data[
                    :, confab_idx, seq_idx, :
                ]
                forward_results_on_sampled_data_up_to_seq_idx = (
                    self.transformer_ebm.forward(data=real_data_up_to_seq_idx)
                )
                # We need to do this clone-then-assign dance in order to avoid the error:
                # "a view of a leaf Variable that requires grad is being used in an in-place operation."
                updated_energy = torch.clone(energy_on_sampled_data)
                updated_energy[:, confab_idx, seq_idx, :] = (
                    energy_on_sampled_data[:, confab_idx, seq_idx, :]
                    + forward_results_on_sampled_data_up_to_seq_idx["energy"][:, -1, :]
                )
                energy_on_sampled_data = updated_energy

        # Recall, energy_on_real_data is shape: (batch size, num. in-context examples, 1)
        # Recall, energy_on_sampled_data is shape: (batch size * ratio_of_confabulated_samples_to_real_samples, num. in-context examples, 1)
        # Reshape to be able to take the element-wise difference.
        energy_on_real_data = energy_on_real_data.reshape(batch_size, 1, max_seq_len, 1)
        energy_on_sampled_data = energy_on_sampled_data.reshape(
            batch_size, ratio_of_confabulated_samples_to_real_samples, max_seq_len, 1
        )

        # Shape: (batch size, num confabulatory samples, num. in-context examples, 1)
        diff_of_energy = energy_on_real_data - energy_on_sampled_data

        # with torch.no_grad():
        #     loss_kl = self.model.forward(im_neg_kl)
        # https://github.com/yilundu/improved_contrastive_divergence/blob/master/train.py#L415C1-L415C95
        #             loss = loss  + (torch.pow(energy_pos, 2).mean() + torch.pow(energy_neg, 2).mean())

        # total_loss = (
        #     energy_diff.mean() + self.wandb_config["mcmc_kwargs"]["kl_coeff"] * loss_kl
        # )

        total_loss = torch.mean(diff_of_energy) + self.wandb_config[
            "energy_regularization"
        ] * (
            torch.mean(torch.square(energy_on_sampled_data))
            + torch.mean(torch.square(energy_on_real_data))
        )

        # Early stopping: If a model's loss plummets, end the run immediately.
        if torch.isnan(
            torch.tensor(total_loss.item())
        ):  # TODO: ideally, this should be some lower bound e.g. 25.0
            print("Loss is NaN. Ending run.")
            exit(0)

        loss_results = {
            "total_loss": total_loss,
            # "loss_kl": loss_kl,
            "energy_on_real_data_mean": energy_on_real_data.mean(),
            # "energy_on_real_data_var": energy_on_real_data.var(),
            # "conditional_energy_on_real_data_mean": conditional_energy_on_real_data.mean(),
            # "conditional_energy_on_real_data_var": conditional_energy_on_real_data.var(),
            "energy_on_sampled_data_mean": energy_on_sampled_data.mean(),
            # "energy_on_sampled_data_var": energy_on_sampled_data.var(),
            # "conditional_energy_on_sampled_data_mean": conditional_energy_on_sampled_data.mean(),
            # "conditional_energy_on_sampled_data_var": conditional_energy_on_sampled_data.var(),
        }

        # Compute and log average energy per time step.
        # for seq_idx in range(10):
        # loss_results[
        #     f"energy_on_real_data_mean_seq_idx={seq_idx}"
        # ] = energy_on_real_data[:, seq_idx, :].mean()
        # loss_results[
        #     f"energy_on_sampled_data_mean_seq_idx={seq_idx}"
        # ] = energy_on_sampled_data[:, seq_idx, :].mean()
        # loss_results[f"diff_of_energy_mean_seq_idx={seq_idx}"] = diff_of_energy[
        #     :, seq_idx, :
        # ].mean()

        for loss_str, loss_val in loss_results.items():
            self.log(
                f"train/{loss_str}",
                loss_val,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

        return total_loss

    def measure_energy_landscape(
        self,
        real_data: torch.Tensor,
        meshgrid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, data_dim = real_data.shape
        total_meshgrid_points, _ = meshgrid.shape
        assert data_dim == 2

        energy_landscape = torch.zeros(
            size=(batch_size, total_meshgrid_points, seq_len, 1),
            device=self.device,
        )
        neg_grad_energy_wrt_meshgrid = torch.zeros(
            size=(batch_size, total_meshgrid_points, seq_len, data_dim),
            device=self.device,
        )
        # Shape: (batch size, total meshgrid points, seq len, data dim)
        real_data_expanded = torch.tile(
            real_data.reshape(batch_size, 1, seq_len, data_dim),
            (1, total_meshgrid_points, 1, 1),
        )
        # with torch.no_grad():
        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                # Shape: (total meshgrid points, seq len, data dim)
                real_data_expanded_up_to_seq_idx = torch.clone(
                    real_data_expanded[batch_idx, :, : seq_idx + 1, :]
                )
                meshgrid_clone = torch.clone(meshgrid)
                meshgrid_clone.requires_grad_(requires_grad=True)
                real_data_expanded_up_to_seq_idx[:, -1, :] = meshgrid_clone
                forward_results = self.transformer_ebm.forward(
                    data=real_data_expanded_up_to_seq_idx
                )

                # Shape: (batch size, step idx + 1, 1)
                energy = forward_results["energy"]

                # Store the computed energy in the energy tensor.
                energy_landscape[batch_idx, :, seq_idx, :] = energy[:, -1, :]

                # Compute the gradient with respect to the single input datum we're sampling.
                # Shape: (batch size, max seq len, data_dim)
                grad_energy_wrt_meshgrid_at_seq_idx = torch.autograd.grad(
                    [energy[:, -1, :].sum()], [meshgrid_clone]
                )[0]

                # Store the computed gradient in the gradient tensor.
                neg_grad_energy_wrt_meshgrid[
                    batch_idx, :, seq_idx, :
                ] = -grad_energy_wrt_meshgrid_at_seq_idx.detach()

        energy_meshgrid_results_dict = {
            # Shape: (batch size, total meshgrid points, seq len, 1)
            "energy_landscape": energy_landscape.detach(),
            # Shape: (batch size, total meshgrid points, seq len, data dim)
            "neg_grad_energy_wrt_meshgrid": neg_grad_energy_wrt_meshgrid.detach(),
        }
        return energy_meshgrid_results_dict

    def sample_data_with_langevin_mcmc(
        self,
        real_data: torch.Tensor,
        initial_sampled_data: torch.Tensor,
        sampling_update_mask: torch.Tensor,
        noise_scale: float = None,
    ) -> Dict[str, torch.Tensor]:
        if noise_scale is None:
            noise_scale = self.wandb_config["mcmc_kwargs"]["noise_scale"]

        # batch_size, max_seq_len, data_dim = real_data.shape
        (
            batch_size,
            confabulation_ratio,
            max_seq_len,
            data_dim,
        ) = initial_sampled_data.shape

        # Create placeholder tensor to sample additive noise.
        additive_noise = torch.randn(
            (batch_size, 1, data_dim), device=self.device
        ).detach()

        # real_data_energy = self.model.forward(real_data)["energy"]
        # Shape: (batch size, ratio_of_confabulated_samples_to_real_samples, n in-context examples, data shape...)
        sampled_data = torch.clone(initial_sampled_data)
        # TODO: Figure out way to vectorize this with transformer.
        for confab_idx in range(confabulation_ratio):
            for seq_idx in range(max_seq_len):
                # Shape: (batch size, 1, data dim)
                sampled_datum = torch.clone(
                    sampled_data[:, confab_idx, seq_idx, np.newaxis, :]
                )
                sampled_datum.requires_grad_(requires_grad=True)

                # Take Langevin MCMC steps.
                for step_idx in range(self.wandb_config["mcmc_kwargs"]["n_mcmc_steps"]):
                    # Sample new additive Gaussian noise.
                    additive_noise.normal_()

                    # Add the additive Gaussian noise to the last element of the sequence.
                    sampled_datum = sampled_datum + noise_scale * additive_noise

                    # Shape: (batch size, step idx + 1, data dim)
                    real_data_up_to_seq_idx_followed_by_sampled_data = torch.cat(
                        [real_data[:, :seq_idx, :], sampled_datum], dim=1
                    )

                    # Shape: (batch size, step idx + 1, 1)
                    energy = self.transformer_ebm.forward(
                        real_data_up_to_seq_idx_followed_by_sampled_data
                    )["energy"]

                    # Compute the gradient with respect to the single input datum we're sampling.
                    # Shape: (batch size, max seq len, data_dim)
                    sampled_datum_grad = torch.autograd.grad(
                        [energy[:, seq_idx, :].sum()], [sampled_datum]
                    )[0]

                    # Clamp gradient to avoid exploding gradients.
                    # Shape: (batch size, 1, data dim)
                    sampled_datum_grad = torch.clamp(
                        sampled_datum_grad,
                        -self.wandb_config["mcmc_kwargs"]["gradient_clip_val"],
                        self.wandb_config["mcmc_kwargs"]["gradient_clip_val"],
                    )

                    # Update the sampled datum following: x   <-   x - nabla_x E(x, D).
                    sampled_datum = (
                        sampled_datum
                        - noise_scale  # step size has to be half of the noise scale
                        * sampled_datum_grad
                        * sampling_update_mask[:, confab_idx, seq_idx, :]
                        / 2.0
                    )

                sampled_data[:, confab_idx, seq_idx, :] = sampled_datum[
                    :, 0, :
                ].detach()

        results = {
            "final_sampled_data": sampled_data.detach(),
        }

        return results

    def sample_data_with_hamiltoinan_mcmc(
        self,
        real_data: torch.Tensor,
        initial_sampled_data: torch.Tensor,
        sampling_update_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # results = {
        #     "final_sampled_data": sampled_data,
        # }
        # return results

        raise NotImplementedError

    def hamiltonian(self, x, v, model, label):
        energy = (
            0.5 * torch.pow(v, 2).sum(dim=1).sum(dim=1).sum(dim=1)
            + model.forward(x, label).squeeze()
        )
        return energy

    def leapfrog_step(self, x, v, model, step_size, num_steps, label, sample=False):
        x.requires_grad_(requires_grad=True)
        energy = model.forward(x, label)
        im_grad = torch.autograd.grad([energy.sum()], [x])[0]
        v = v - 0.5 * step_size * im_grad
        im_negs = []

        for i in range(num_steps):
            x.requires_grad_(requires_grad=True)
            energy = model.forward(x, label)

            if i == num_steps - 1:
                im_grad = torch.autograd.grad([energy.sum()], [x], create_graph=True)[0]
                v = v - step_size * im_grad
                x = x + step_size * v
                v = v.detach()
            else:
                im_grad = torch.autograd.grad([energy.sum()], [x])[0]
                v = v - step_size * im_grad
                x = x + step_size * v
                x = x.detach()
                v = v.detach()

            if sample:
                im_negs.append(x)

            if i % 10 == 0:
                print(
                    i,
                    self.hamiltonian(torch.sigmoid(x), v, model, label).mean(),
                    torch.abs(im_grad).mean(),
                )

        if sample:
            return x, im_negs, v, im_grad
        else:
            return x, v, im_grad

    def gen_hmc_image(self, label, FLAGS, model, im_neg, num_steps, sample=False):
        step_size = self.wandb_config["mcmc_kwargs"]["step_size"]

        v = 0.001 * torch.randn_like(im_neg)

        if sample:
            im_neg, im_negs, v, im_grad = self.leapfrog_step(
                im_neg, v, model, step_size, num_steps, label, sample=sample
            )
            return im_neg, im_negs, im_grad, v
        else:
            im_neg, v, im_grad = self.leapfrog_step(
                im_neg, v, model, step_size, num_steps, label, sample=sample
            )
            return im_neg, im_grad, v

    def gen_image(self, label, FLAGS, model, im_neg, num_steps, sample=False):
        im_noise = torch.randn_like(im_neg).detach()

        im_negs_samples = []

        for i in range(num_steps):
            im_noise.normal_()

            if FLAGS.anneal:
                im_neg = im_neg + 0.001 * (num_steps - i - 1) / num_steps * im_noise
            else:
                im_neg = im_neg + 0.001 * im_noise

            im_neg.requires_grad_(requires_grad=True)
            energy = model.forward(im_neg, label)

            if FLAGS.all_step:
                im_grad = torch.autograd.grad(
                    [energy.sum()], [im_neg], create_graph=True
                )[0]
            else:
                im_grad = torch.autograd.grad([energy.sum()], [im_neg])[0]

            if i == num_steps - 1:
                im_neg_orig = im_neg
                im_neg = im_neg - FLAGS.step_lr * im_grad

                if FLAGS.dataset == "cifar10":
                    n = 128
                elif FLAGS.dataset == "celeba":
                    # Save space
                    n = 128
                elif FLAGS.dataset == "lsun":
                    # Save space
                    n = 32
                elif FLAGS.dataset == "object":
                    # Save space
                    n = 32
                elif FLAGS.dataset == "mnist":
                    n = 32
                elif FLAGS.dataset == "imagenet":
                    n = 32
                elif FLAGS.dataset == "stl":
                    n = 32

                im_neg_kl = im_neg_orig[:n]
                if sample:
                    pass
                else:
                    energy = model.forward(im_neg_kl, label)
                    im_grad = torch.autograd.grad(
                        [energy.sum()], [im_neg_kl], create_graph=True
                    )[0]

                im_neg_kl = im_neg_kl - FLAGS.step_lr * im_grad[:n]
                im_neg_kl = torch.clamp(im_neg_kl, 0, 1)
            else:
                im_neg = im_neg - FLAGS.step_lr * im_grad

            im_neg = im_neg.detach()

            if sample:
                im_negs_samples.append(im_neg)

            im_neg = torch.clamp(im_neg, 0, 1)

        if sample:
            return im_neg, im_neg_kl, im_negs_samples, im_grad
        else:
            return im_neg, im_neg_kl, im_grad


class EnergyBasedTransformerModel(pl.LightningModule):
    def __init__(self, wandb_config: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.wandb_config = wandb_config
        if self.wandb_config["which_transformer"] == "huggingface":
            self.transformer = TransformerModelHuggingface(
                wandb_config=wandb_config,
            )
        elif self.wandb_config["which_transformer"] == "pytorch":
            self.transformer = TransformerModelPytorch(
                wandb_config=wandb_config, *args, **kwargs
            )
        else:
            raise ValueError(
                f"Invalid transformer: {self.wandb_config['which_transformer']}"
            )

    def forward(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, data_dim = data.shape

        return self.transformer.forward(data)


# adapted from: https://github.com/dtsip/in-context-learning/blob/main/src/models.py
class TransformerModelHuggingface(pl.LightningModule):
    def __init__(self, wandb_config: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        super(TransformerModelHuggingface, self).__init__()
        self.wandb_config = wandb_config
        config = GPT2Config(
            n_positions=self.wandb_config["dataset_kwargs"]["max_n_samples_in_context"],
            n_embd=self.wandb_config["model_kwargs"]["d_embed"],
            n_layer=self.wandb_config["model_kwargs"]["n_layers"],
            n_head=self.wandb_config["model_kwargs"]["n_heads"],
            resid_pdrop=self.wandb_config["model_kwargs"]["resid_pdrop"],
            embd_pdrop=self.wandb_config["model_kwargs"]["embd_pdrop"],
            attn_pdrop=self.wandb_config["model_kwargs"]["attn_pdrop"],
            use_cache=False,
        )

        self.in_layer = torch.nn.Linear(
            in_features=self.wandb_config["dataset_kwargs"]["n_dimensions"],
            out_features=self.wandb_config["model_kwargs"]["d_embed"],
        )

        self.out_layer = torch.nn.Linear(
            in_features=self.wandb_config["model_kwargs"]["d_embed"],
            out_features=1,
        )
        self.transformer = GPT2Model(config=config)

    def forward(self, data: torch.Tensor):
        # Data has shape: Shape: (batch size, max seq len, data dim + potentially 1 if doing linear regression)

        # Shape: (batch size, max seq len, data dim)
        embeds = self.in_layer(data)
        # Shape: (batch size, max seq len, d_embed)
        output = self.transformer(inputs_embeds=embeds).last_hidden_state
        prediction = self.out_layer(output)
        return {"energy": prediction}  # TODO: change this?


class TransformerModelPytorch(pl.LightningModule):
    def __init__(self, wandb_config: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.wandb_config = wandb_config

        # https://discuss.pytorch.org/t/nn-transformerdecoderlayer-without-encoder-input/183990/2
        encoder_layer = torch.nn.TransformerEncoderLayer(
            activation=self.wandb_config["model_kwargs"]["activation"],
            d_model=self.wandb_config["model_kwargs"]["d_embed"],
            nhead=self.wandb_config["model_kwargs"]["n_heads"],
            dropout=self.wandb_config["model_kwargs"]["dropout"],
            batch_first=True,
            norm_first=True,
            bias=self.wandb_config["model_kwargs"]["bias"],
        )
        self.in_layer = torch.nn.Linear(
            in_features=self.wandb_config["dataset_kwargs"]["n_dimensions"],
            out_features=self.wandb_config["model_kwargs"]["d_embed"],
        )

        self.out_layer = torch.nn.Linear(
            in_features=self.wandb_config["model_kwargs"]["d_embed"],
            out_features=1,
        )

        self.causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            sz=self.wandb_config["dataset_kwargs"]["max_n_samples_in_context"],
        )

        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.wandb_config["model_kwargs"]["n_layers"],
        )

    def forward(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, data_dim = data.shape
        # Shape: (batch size, seq len, d_embed)
        in_layer_outputs = self.in_layer(data)
        # Shape: (seq len, batch size, d_embed)
        transformer_outputs = self.transformer(
            src=in_layer_outputs,
            mask=self.causal_mask[:seq_len, :seq_len],  # Take only the length we need.
            is_causal=True,
        )
        # Shape: (batch size, seq len, 1)
        out_layer_outputs = self.out_layer(transformer_outputs)
        forward_results = {
            "in_layer_outputs": in_layer_outputs,
            "transformer_output": transformer_outputs,
            "energy": out_layer_outputs,
        }
        return forward_results


class EnergyFunctionGaussian(pl.LightningModule):
    def __init__(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ):
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.prob_distribution = torch.distributions.MultivariateNormal(
            loc=self.mu, covariance_matrix=self.sigma
        )

    def forward(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        forward_results = {
            "energy": -self.prob_distribution.log_prob(data),
        }
        return forward_results


class EnergyFunctionMixtureOfGaussians(pl.LightningModule):
    def __init__(
        self,
        means: torch.Tensor,
        covariances: torch.Tensor,
        wandb_config,
    ):
        super().__init__()
        # TODO: Generalize to non-uniform mixture of Gaussians.
        assert len(means) == len(covariances)
        self.n_components = len(means)
        self.means = means
        self.covariances = covariances
        self.prob_distribution = (
            torch.distributions.mixture_same_family.MixtureSameFamily(
                mixture_distribution=torch.distributions.Categorical(
                    torch.ones(self.n_components).to(self.means.device)
                ),
                component_distribution=torch.distributions.MultivariateNormal(
                    loc=self.means,
                    covariance_matrix=self.covariances,
                ),
            )
        )
        self.wandb_config = wandb_config

    def forward(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        energy = -self.prob_distribution.log_prob(data)
        forward_results = {
            "energy": energy,
        }
        return forward_results

    def likelihood(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        likelihood = self.prob_distribution.log_prob(data)
        forward_results = {
            "likelihood": likelihood,
        }
        return forward_results

    def sample_data_with_langevin_mcmc(
        self, real_data: torch.Tensor, initial_sampled_data: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # batch_size, max_seq_len, data_dim = real_data.shape
        (
            batch_size,
            confabulation_ratio,
            max_seq_len,
            data_dim,
        ) = initial_sampled_data.shape

        # real_data_energy = self.model.forward(real_data)["energy"]
        # Shape: (batch size, ratio_of_confabulated_samples_to_real_samples, n in-context examples, data shape...)
        sampled_data = torch.clone(initial_sampled_data)
        sampled_data.requires_grad_(requires_grad=True)

        # Follow the negative gradient.
        for step_idx in range(self.wandb_config["mcmc_kwargs"]["n_mcmc_steps"]):
            # Shape: (batch size, step idx + 1, 1)
            energy = self.forward(sampled_data)["energy"]

            # Rescale energy to make it ~10. Otherwise we need to take step size into account.
            with torch.no_grad():
                # This needs to be an inplace operation, otherwise the next line breaks!
                energy *= 10.0 / torch.max(energy)

            # Compute the gradient with respect to the single input datum we're sampling.
            # Shape: (batch size, max seq len, data_dim)
            sampled_data_grad = torch.autograd.grad([energy.sum()], [sampled_data])[0]

            # Update the sampled datum following: x   <-   x - nabla_x E(x, D).
            sampled_data = (
                sampled_data
                - self.wandb_config["mcmc_kwargs"]["step_size"]
                * sampled_data_grad
                / 10.0
            )

        sampled_data = sampled_data.detach()
        results = {
            "final_sampled_data": sampled_data.detach(),
        }

        return results
