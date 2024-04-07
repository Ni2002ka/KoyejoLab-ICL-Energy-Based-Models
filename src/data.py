import ast
from functools import partial
import lightning.pytorch as pl
import numpy
import numpy as np
import torch
import torch.distributions
import torch.utils.data
import torchvision
import torchvision.transforms
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

from lightning.pytorch.utilities.types import EVAL_DATALOADERS

six_circles_centers = np.array(
    [
        [1.0, 0.0],
        [0.5, np.sqrt(3.0) / 2.0],
        [-0.5, np.sqrt(3.0) / 2.0],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3.0) / 2.0],
        [0.5, -np.sqrt(3.0) / 2.0],
    ]
)


class MixtureOf2DDatasets(torch.utils.data.Dataset):
    def __init__(self, wandb_config: Dict[str, Any], split: str = "train") -> None:
        self.wandb_config = wandb_config

        self.n_samples_per_dataset = self.wandb_config["dataset_kwargs"][
            "n_samples_per_dataset"
        ]
        self.n_samples_in_context = self.wandb_config["dataset_kwargs"][
            "max_n_samples_in_context"
        ]
        self.dataset_names = ast.literal_eval(
            self.wandb_config["dataset_kwargs"]["dataset_names"]
        )
        self.data_tensors_dict = {
            dataset_name: self.create_data(dataset_name=dataset_name)
            for dataset_name in self.dataset_names
        }

        # Compute the range of synthetic data.
        # self.noise_min = (
        #     1.2 * min([data.min() for data in self.data_tensors_dict.values()]).item()
        # )
        self.noise_min = -5.0
        # self.noise_max = (
        #     1.2 * max([data.max() for data in self.data_tensors_dict.values()]).item()
        # )
        self.noise_max = 5.0

        # Compute the "length" of this synthetic dataset.
        self.split = split
        if self.split == "train":
            self.length = (
                self.wandb_config["batch_size_train"]
                * self.wandb_config["n_batches_per_epoch"]
            )
        elif split == "val":
            self.length = self.wandb_config["batch_size_val"]
        elif split == "test":
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid split: {split}")

        self.ratio_of_confabulated_samples_to_real_samples = self.wandb_config[
            "mcmc_kwargs"
        ]["ratio_of_confabulated_samples_to_real_samples"]

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Sample a random dataset.
        dataset_name = self.dataset_names[
            torch.randint(low=0, high=len(self.dataset_names), size=(1,))
        ]
        # Sample in-context data from that dataset.
        # Shape: (n_samples_in_context, 2)
        in_context_data = self.data_tensors_dict[dataset_name][
            torch.randint(
                low=0,
                high=self.n_samples_per_dataset,
                size=(self.n_samples_in_context,),
            )
        ]
        # Draw noise from Uniform(noise_min, noise_max).
        initial_sampled_data = (self.noise_max - self.noise_min) * torch.rand(
            (self.ratio_of_confabulated_samples_to_real_samples,)
            + in_context_data.shape
        ) + self.noise_min
        return {
            "real_data": in_context_data,
            "initial_sampled_data": initial_sampled_data,
        }

    def create_data(
        self,
        dataset_name: str,
        n_samples_per_dataset: int = 1000,
    ) -> torch.Tensor:
        if dataset_name.startswith("anisotropic_blobs"):
            # Modified from: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html
            from sklearn.datasets import make_blobs

            dataset_name, random_state_str = dataset_name.split("_rs=")

            data = torch.tensor(
                make_blobs(
                    n_samples=n_samples_per_dataset,
                    centers=3,
                    cluster_std=0.5,
                    random_state=int(random_state_str),
                )[0]
            )
            transformation = torch.tensor([[0.6, -0.63], [-0.4, 0.85]]).double()
            data = data @ transformation
        elif dataset_name.startswith("blobs"):
            from sklearn.datasets import make_blobs

            dataset_name, random_state_str = dataset_name.split("_rs=")

            data = torch.tensor(
                make_blobs(
                    n_samples=n_samples_per_dataset,
                    centers=3,
                    cluster_std=0.5,
                    random_state=int(random_state_str),
                )[0]
            )
        elif dataset_name == "horizontal_rectangle":
            data = self.sample_uniformly_in_rectangle(
                n_samples=n_samples_per_dataset,
                min_x=-3.0,
                max_x=3.0,
                min_y=-1.0,
                max_y=1.0,
            )
        elif dataset_name == "moons":
            from sklearn.datasets import make_moons

            data = 2.0 * torch.tensor(
                make_moons(
                    n_samples=n_samples_per_dataset,
                    noise=0.05,
                )[0]
            )
        elif dataset_name == "one_circle":
            data = self.sample_uniformly_in_circle(
                n_samples=n_samples_per_dataset,
                center=(0.0, 0.0),
                radius=1.0,
            )
        elif dataset_name == "six_big_circles":
            data = torch.cat(
                [
                    self.sample_uniformly_in_circle(
                        n_samples=n_samples_per_dataset // 6 + 1,
                        center=3 * center,
                        radius=1.0,
                    )
                    for center in six_circles_centers
                ],
                dim=0,
            )
        elif dataset_name == "six_small_circles":
            data = torch.cat(
                [
                    self.sample_uniformly_in_circle(
                        n_samples=n_samples_per_dataset // 6 + 1,
                        center=center,
                        radius=0.5,
                    )
                    for center in six_circles_centers
                ],
                dim=0,
            )
        elif dataset_name == "three_odd_small_circles":
            data = torch.cat(
                [
                    self.sample_uniformly_in_circle(
                        n_samples=n_samples_per_dataset // 3 + 1,
                        center=3 * center,
                        radius=0.5,
                    )
                    for center in six_circles_centers[1::2]
                ],
                dim=0,
            )
        elif dataset_name == "three_even_small_circles":
            data = torch.cat(
                [
                    self.sample_uniformly_in_circle(
                        n_samples=n_samples_per_dataset // 3 + 1,
                        center=3 * center,
                        radius=0.5,
                    )
                    for center in six_circles_centers[::2]
                ],
                dim=0,
            )
        elif dataset_name == "swiss_roll":
            from sklearn.datasets import make_swiss_roll

            data = torch.tensor(
                make_swiss_roll(
                    n_samples=n_samples_per_dataset,
                    noise=0.05,
                )[
                    0
                ][:, [0, 2]]
            )
        elif dataset_name == "vertical_rectangle":
            data = self.sample_uniformly_in_rectangle(
                n_samples=n_samples_per_dataset,
                min_x=-1.0,
                max_x=1.0,
                min_y=-3.0,
                max_y=3.0,
            )
        else:
            raise ValueError(f"Invalid dataset_name: {dataset_name}")

        return data.float()

    @staticmethod
    def sample_uniformly_in_circle(
        n_samples: int = 1000,
        center: Tuple[float, float] = (0.0, 0.0),
        radius: float = 1.0,
    ) -> torch.Tensor:
        # https://stackoverflow.com/a/50746409/4570472
        data_radii = radius * torch.sqrt(torch.rand(n_samples))
        theta = 2.0 * np.pi * torch.rand(n_samples)
        return torch.stack(
            [
                data_radii * torch.cos(theta) + center[0],
                data_radii * torch.sin(theta) + center[1],
            ],
            dim=-1,
        )

    @staticmethod
    def sample_uniformly_in_rectangle(
        n_samples: int = 1000,
        min_x: float = -1.0,
        max_x: float = 1.0,
        min_y: float = -1.0,
        max_y: float = 1.0,
    ) -> torch.Tensor:
        return torch.stack(
            [
                (max_x - min_x) * torch.rand(n_samples) + min_x,
                (max_y - min_y) * torch.rand(n_samples) + min_y,
            ],
            dim=-1,
        )


class MixtureOfGaussiansDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        wandb_config: Dict[str, Any],
        split: str = "train",
    ) -> None:
        super().__init__()

        self.wandb_config = wandb_config
        # TODO: Generalize this to handle variable numbers of components
        assert isinstance(self.wandb_config["dataset_kwargs"]["n_components"], int)
        self.n_components = self.wandb_config["dataset_kwargs"]["n_components"]
        self.n_dimensions = self.wandb_config["dataset_kwargs"]["n_dimensions"]
        self.n_samples_per_dataset = self.wandb_config["dataset_kwargs"][
            "n_samples_per_dataset"
        ]
        self.n_unique_datasets = self.wandb_config["dataset_kwargs"][
            "n_unique_datasets"
        ]
        assert self.wandb_config["dataset_kwargs"]["prior"] == "gaussian"
        self.n_samples_per_dataset = self.wandb_config["dataset_kwargs"][
            "n_samples_per_dataset"
        ]

        self.n_samples_in_context = self.wandb_config["dataset_kwargs"][
            "max_n_samples_in_context"
        ]

        # Create the data. We have 3 cases:
        #   1. Finitely many unique pretraining datasets, each with finitely many samples.
        #   2. Finitely many unique pretraining datasets, each with infinitely many samples.
        #   3. Infinitely many unique pretraining datasets, with infinitely many samples.
        if self.n_unique_datasets < float("inf"):
            finitely_many_mixture_of_gaussians_list = [
                self.create_mixture_of_gaussians_distribution()
                for _ in range(self.n_unique_datasets)
            ]
            self.finitely_many_mixture_of_gaussians_list = (
                finitely_many_mixture_of_gaussians_list
            )

            # Finitely many unique pretraining datasets, each with finitely many samples.
            if self.n_samples_per_dataset < float("inf"):
                self.finite_mog_finite_samples_list = [
                    finitely_many_mixture_of_gaussians_list[dataset_idx].sample(
                        sample_shape=(self.n_samples_per_dataset,)
                    )
                    for dataset_idx in range(self.n_unique_datasets)
                ]
                # self.length = self.n_unique_datasets
                # self.length = (
                #     self.n_unique_datasets
                #     * self.n_samples_per_dataset
                #     // self.wandb_config["batch_size_train"]
                #     * self.wandb_config["n_batches_per_epoch"]
                # )
            # Finitely many unique pretraining datasets, each with infinitely many samples.
            else:
                self.finitely_many_mixture_of_gaussians_list = (
                    finitely_many_mixture_of_gaussians_list
                )
        else:
            # Infinitely many unique pretraining datasets, with infinitely many samples
            self.data_generating_object = self.create_mixture_of_gaussians_distribution
            # self.length = self.wandb_config["dataset_kwargs"]["dataset_length"]

        # Compute the range of synthetic data.
        self.noise_min = (
            self.wandb_config["dataset_kwargs"]["prior_kwargs"]["mean"]
            - 3.0 * self.wandb_config["dataset_kwargs"]["prior_kwargs"]["std_dev"]
        )
        self.noise_max = (
            self.wandb_config["dataset_kwargs"]["prior_kwargs"]["mean"]
            + 3.0 * self.wandb_config["dataset_kwargs"]["prior_kwargs"]["std_dev"]
        )

        self.split = split
        if self.split == "train":
            self.length = (
                self.wandb_config["batch_size_train"]
                * self.wandb_config["n_batches_per_epoch"]
            )
        elif self.split == "val":
            self.length = self.wandb_config["batch_size_val"]
        else:
            # TODO: Remove this hardcoding.
            self.length = 10000  # self.wandb_config["dataset_kwargs"]["dataset_length"]

        self.ratio_of_confabulated_samples_to_real_samples = self.wandb_config[
            "mcmc_kwargs"
        ]["ratio_of_confabulated_samples_to_real_samples"]

    def create_mixture_of_gaussians_distribution(
        self,
    ) -> torch.distributions.mixture_same_family.MixtureSameFamily:
        prior_cov = np.square(
            self.wandb_config["dataset_kwargs"]["prior_kwargs"]["std_dev"]
        ) * torch.eye(self.n_dimensions)
        mean_distribution = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=torch.zeros(self.n_dimensions),
            covariance_matrix=prior_cov,
        )
        means = mean_distribution.sample((self.n_components,))
        component_cov = np.square(
            self.wandb_config["dataset_kwargs"]["component_kwargs"]["std_dev"]
        ) * torch.eye(self.n_dimensions)
        mixture_of_gaussians = (
            torch.distributions.mixture_same_family.MixtureSameFamily(
                mixture_distribution=torch.distributions.Categorical(
                    torch.ones(self.n_components)
                ),
                component_distribution=torch.distributions.MultivariateNormal(
                    loc=means,
                    covariance_matrix=component_cov,
                ),
            )
        )

        return mixture_of_gaussians

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Sample in-context data from that dataset.
        if self.n_unique_datasets < float("inf"):
            # Finitely many unique pretraining datasets, each with finitely many samples.
            if self.n_samples_per_dataset < float("inf"):
                # Sample a random dataset.
                dataset_idx = torch.randint(
                    low=0, high=len(self.finite_mog_finite_samples_list), size=(1,)
                ).item()

                # Sample in-context data from that dataset.
                # Shape: (n_samples_in_context, n_dimensions)
                in_context_data = self.finite_mog_finite_samples_list[dataset_idx][
                    torch.randint(
                        low=0,
                        high=self.n_samples_per_dataset,
                        size=(self.n_samples_in_context,),
                    )
                ]

            # Finitely many unique pretraining datasets, each with infinitely many samples.
            else:
                dataset_idx = torch.randint(
                    low=0,
                    high=len(self.finitely_many_mixture_of_gaussians_list),
                    size=(1,),
                ).item()

                # Shape: (n_samples_in_context, n_dimensions)
                in_context_data = self.finitely_many_mixture_of_gaussians_list[
                    dataset_idx
                ].sample(sample_shape=(self.n_samples_in_context,))

            means = self.finitely_many_mixture_of_gaussians_list[
                dataset_idx
            ].component_distribution.mean
            covariances = self.finitely_many_mixture_of_gaussians_list[
                dataset_idx
            ].component_distribution.covariance_matrix

        else:
            # Infinitely many unique pretraining datasets, with infinitely many samples
            mixture_of_gaussians = self.create_mixture_of_gaussians_distribution()

            # Shape: (n_components, n_dimensions)
            means = mixture_of_gaussians.component_distribution.mean
            # Shape: (n_components, n_dimensions, n_dimensions)
            covariances = mixture_of_gaussians.component_distribution.covariance_matrix

            # Shape: (n_samples_in_context, n_dimensions)
            in_context_data = mixture_of_gaussians.sample(
                sample_shape=(self.n_samples_in_context,)
            )

        # Draw noise from Uniform(noise_min, noise_max).
        initial_sampled_data = (self.noise_max - self.noise_min) * torch.rand(
            (self.ratio_of_confabulated_samples_to_real_samples,)
            + in_context_data.shape
        ) + self.noise_min
        return {
            "real_data": in_context_data,
            "initial_sampled_data": initial_sampled_data,
            "means": means,
            "covariances": covariances,
        }

    def __len__(self):
        return self.length


class LinearDistribution(torch.utils.data.Dataset):
    def __init__(
        self, weight: torch.Tensor, n_dimensions, x_prior_mean, x_prior_std_dev
    ) -> None:
        self.n_dimensions = n_dimensions
        self.weight = weight
        self.x_prior_mean = x_prior_mean
        self.x_prior_std_dev = x_prior_std_dev

    def sample(self, sample_shape):
        # sample n-1 dimensional x vector
        sample_x = (
            torch.randn(self.n_dimensions - 1) * self.x_prior_std_dev
            + self.x_prior_mean
        )
        sample_y = torch.dot(self.weight, sample_x)

        # Append y entry to x
        sample = torch.cat((sample_x, sample_y), dim=0)
        return sample


class LinearRegressionsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        wandb_config: Dict[str, Any],
        split: str = "train",
    ) -> None:
        super().__init__()

        self.wandb_config = wandb_config
        # TODO: Generalize this to handle variable numbers of components
        assert isinstance(self.wandb_config["dataset_kwargs"]["n_components"], int)
        self.n_components = self.wandb_config["dataset_kwargs"]["n_components"]
        self.n_dimensions = self.wandb_config["dataset_kwargs"]["n_dimensions"]
        self.n_samples_per_dataset = self.wandb_config["dataset_kwargs"][
            "n_samples_per_dataset"
        ]
        self.n_unique_datasets = self.wandb_config["dataset_kwargs"][
            "n_unique_datasets"
        ]

        assert self.wandb_config["dataset_kwargs"]["prior"] == "linear_regression"
        self.n_samples_per_dataset = self.wandb_config["dataset_kwargs"][
            "n_samples_per_dataset"
        ]

        self.n_samples_in_context = self.wandb_config["dataset_kwargs"][
            "max_n_samples_in_context"
        ]

        if self.wandb_config["dataset_kwargs"]["w_prior"] == "isotropic_gaussian":
            self.w_prior_mean = self.wandb_config["dataset_kwargs"]["w_prior_kwargs"][
                "mean"
            ]
            self.w_prior_std_dev = self.wandb_config["dataset_kwargs"][
                "w_prior_kwargs"
            ]["std_dev"]

        if self.wandb_config["dataset_kwargs"]["x_prior"] == "isotropic_gaussian":
            self.x_prior_mean = self.wandb_config["dataset_kwargs"]["x_prior_kwargs"][
                "mean"
            ]
            self.x_prior_std_dev = self.wandb_config["dataset_kwargs"][
                "x_prior_kwargs"
            ]["std_dev"]

        # We have 3 cases:
        #   1. Finitely many unique pretraining datasets, each with finitely many samples.
        #   2. Finitely many unique pretraining datasets, each with infinitely many samples.
        #   3. Infinitely many unique pretraining datasets, with infinitely many samples. Constantly draw new weights
        if self.n_unique_datasets < float("inf"):
            finitely_many_mixture_of_linears_list = [
                self.create_linear_distribution() for _ in range(self.n_unique_datasets)
            ]
            self.finitely_many_linear_datasets_list = (
                finitely_many_mixture_of_linears_list
            )

            # Finitely many unique pretraining datasets, each with finitely many samples.
            if self.n_samples_per_dataset < float("inf"):
                self.finite_linear_datasets_finite_samples_list = [
                    finitely_many_mixture_of_linears_list[dataset_idx].sample(
                        sample_shape=(self.n_samples_per_dataset,)
                    )
                    for dataset_idx in range(self.n_unique_datasets)
                ]
            else:
                self.finitely_many_linear_datasets_list = (
                    finitely_many_mixture_of_linears_list
                )
        else:
            # Infinitely many unique pretraining datasets, with infinitely many samples
            self.data_generating_object = self.create_linear_distribution

        # Compute the range of synthetic data.
        self.noise_min = (
            self.wandb_config["dataset_kwargs"]["noise_prior_kwargs"]["mean"]
            - 3.0 * self.wandb_config["dataset_kwargs"]["noise_prior_kwargs"]["std_dev"]
        )
        self.noise_max = (
            self.wandb_config["dataset_kwargs"]["noise_prior_kwargs"]["mean"]
            + 3.0 * self.wandb_config["dataset_kwargs"]["noise_prior_kwargs"]["std_dev"]
        )

        self.split = split
        if self.split == "train":
            self.length = (
                self.wandb_config["batch_size_train"]
                * self.wandb_config["n_batches_per_epoch"]
            )
        elif self.split == "val":
            self.length = self.wandb_config["batch_size_val"]
        else:
            # TODO: Remove this hardcoding.
            self.length = 10000  # self.wandb_config["dataset_kwargs"]["dataset_length"]

        self.ratio_of_confabulated_samples_to_real_samples = self.wandb_config[
            "mcmc_kwargs"
        ]["ratio_of_confabulated_samples_to_real_samples"]

    def create_linear_distribution(
        self,
    ) -> LinearDistribution:
        weight = (
            torch.randn(self.n_dimensions - 1) * self.w_prior_std_dev
            + self.w_prior_mean
        )
        linear_dist = LinearDistribution(
            weight=weight,
            n_dimensions=self.n_dimensions,
            x_prior_mean=self.x_prior_mean,
            x_prior_std_dev=self.x_prior_std_dev,
        )
        return linear_dist

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Sample in-context data from that dataset.
        if self.n_unique_datasets < float("inf"):
            # Finitely many unique pretraining datasets, each with finitely many samples.
            if self.n_samples_per_dataset < float("inf"):
                # Sample a random dataset.
                dataset_idx = torch.randint(
                    low=0,
                    high=len(self.finite_linear_datasets_finite_samples_list),
                    size=(1,),
                ).item()

                # Sample in-context data from that dataset.
                # Shape: (n_samples_in_context, n_dimensions)
                in_context_data = self.finite_linear_datasets_finite_samples_list[
                    dataset_idx
                ][
                    torch.randint(
                        low=0,
                        high=self.n_samples_per_dataset,
                        size=(self.n_samples_in_context,),
                    )
                ]

            # Finitely many unique pretraining datasets, each with infinitely many samples.
            else:
                dataset_idx = torch.randint(
                    low=0,
                    high=len(self.finitely_many_linear_datasets_list),
                    size=(1,),
                ).item()

                # Shape: (n_samples_in_context, n_dimensions)
                in_context_data = self.finitely_many_linear_datasets_list[
                    dataset_idx
                ].sample(sample_shape=(self.n_samples_in_context,))

        else:
            # Infinitely many unique pretraining datasets, with infinitely many samples
            linear_distribution = self.create_linear_distribution()

            # Shape: (n_samples_in_context, n_dimensions)
            in_context_data = linear_distribution.sample(
                sample_shape=(self.n_samples_in_context,)
            )

        # Draw noise from Uniform(noise_min, noise_max).
        # TODO: change to gaussian noise
        initial_sampled_data = (self.noise_max - self.noise_min) * torch.rand(
            (self.ratio_of_confabulated_samples_to_real_samples,)
            + in_context_data.shape
        ) + self.noise_min
        return {
            "real_data": in_context_data,
            "initial_sampled_data": initial_sampled_data,
        }

    def __len__(self):
        return self.length


class InContextLearningEnergyBasedModelDataModule(pl.LightningDataModule):
    def __init__(
        self,
        wandb_config: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.wandb_config = wandb_config

        # Recommendation: https://pytorch-lightning.readthedocs.io/en/latest/guides/speed.html
        if "n_workers" not in self.wandb_config:
            # n_workers = max(4, os.cpu_count() // 4)  # heuristic
            self.n_workers = 1
        else:
            self.n_workers = self.wandb_config["n_workers"]
        self.train_dataset = None
        self.val_dataset = None

        # Allegedly pinning memory saves time. Not sure what it does though.
        # self.pin_memory = torch.cuda.is_available()
        self.pin_memory = False

    def setup(self, stage: str) -> None:
        self.train_dataset = create_dataset(
            wandb_config=self.wandb_config,
            split="train",
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.wandb_config["batch_size_train"],
            num_workers=self.n_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            worker_init_fn=partial(self.worker_init_fn, split_seed=0),
        )

    @staticmethod
    def worker_init_fn(id, split_seed: int):
        # Recommended by NumPy Rng Author: https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
        # Another good resource: https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
        process_seed = torch.initial_seed()
        # Back out the base_seed so we can use all the bits.
        base_seed = process_seed - id
        # TODO: split_seed seems to have no impact.
        ss = np.random.SeedSequence(
            [id, base_seed, split_seed]
        )  # Rylan added split seed.
        # More than 128 bits (4 32-bit words) would be overkill.
        np.random.seed(ss.generate_state(4))

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished.
        print(f"TrajectoryDataModule.teardown(stage={stage}) called.")


def create_dataset(
    wandb_config: Dict[str, Any],
    split: str = "train",
) -> torch.utils.data.Dataset:
    if wandb_config["dataset_kwargs"]["dataset"] == "mixture_of_2D_datasets":
        dataset = MixtureOf2DDatasets(
            wandb_config=wandb_config,
            split=split,
        )
    elif wandb_config["dataset_kwargs"]["dataset"] == "mixture_of_gaussians":
        dataset = MixtureOfGaussiansDataset(
            wandb_config=wandb_config,
            split=split,
        )
    # TODO: MSE error
    # TODO: a parent class for linear and gaussian etc.
    elif wandb_config["dataset_kwargs"]["dataset"] == "linear_regression":
        dataset = LinearRegressionsDataset(wandb_config=wandb_config, split=split)
    else:
        raise NotImplementedError
    return dataset
