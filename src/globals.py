default_config = {
    "accumulate_grad_batches": 2,
    "batch_size_train": 8,
    "batch_size_val": 1,
    "check_val_every_n_epoch": 5,
    "compile_model": False,
    # "dataset_kwargs": {
    #     "dataset": "mixture_of_gaussians",
    #     # "dataset_length": 10000,
    #     "max_n_samples_in_context": 35,
    #     "variable_n_components_in_train": True,
    #     "n_components": 3,
    #     "n_dimensions": 2,
    #     # "n_samples_per_dataset": float("inf"),
    #     "n_samples_per_dataset": 1000,
    #     # "n_unique_datasets": float("inf"),
    #     "n_unique_datasets": 10,
    #     "prior": "gaussian",
    #     "prior_kwargs": {
    #         "mean": 0.0,
    #         "std_dev": 5.0,
    #     },
    #     "component": "gaussian",
    #     "component_kwargs": {
    #         "std_dev": 0.33,
    #     },
    # },
    "dataset_kwargs": {
        "dataset": "linear_regression",
        "max_n_samples_in_context": 31,
        "n_dimensions": 2,
        # "n_samples_per_dataset": float("inf"),
        "n_samples_per_dataset": 1000,
        "n_unique_datasets": float("inf"),
        # "n_unique_datasets": 7,
        "w_prior": "isotropic_gaussian",
        "w_prior_kwargs": {
            "mean": 0.0,
            "std_dev": 2.0,
        },
        # "x_prior": "isotropic_gaussian",
        # "x_prior_kwargs": {
        #     "mean": 0.0,
        #     "std_dev": 100.0,
        # },
        "x_prior": "uniform",
        "x_prior_kwargs": {
            "low": -15.0,
            "high": 15.0,
        },
        "data_std_dev": .5,
        "noise_prior": "isotropic_gaussian",
        "noise_prior_kwargs": {
            "mean": 0.0,
            "std_dev": 5.0,
        },
    },
    "eval_datasets": {
        "standard_eval_dataset_kwargs": {
            "dataset": "linear_regression",
            # "dataset_length": 10000,
            "max_n_samples_in_context": 35,
            "n_dimensions": 2,
            # "n_samples_per_dataset": float("inf"),
            "n_samples_per_dataset": 1000,
            # "n_unique_datasets": float("inf"),
            "n_unique_datasets": 10,
            "w_prior": "isotropic_gaussian",
            "w_prior_kwargs": {
                "mean": 0.0,
                "std_dev": 5.0,
            },
            # "x_prior": "isotropic_gaussian",
            # "x_prior_kwargs": {
            #     "mean": 0.0,
            #     "std_dev": 10.0,
            # },
            "x_prior": "uniform",
            "x_prior_kwargs": {
                "low": -10.0,
                "high": 10.0,
            },
            "data_std_dev": .5,
            "noise_prior": "isotropic_gaussian",
            "noise_prior_kwargs": {
                "mean": 0.0,
                "std_dev": 5.0,
            },
        },
        # "num_gaussian_eval_dataset_kwargs": {
        #     "dataset": "mixture_of_gaussians",
        #     # "dataset_length": 10000,
        #     "max_n_samples_in_context": 35,
        #     "n_components": 5,
        #     "n_dimensions": 2,
        #     # "n_samples_per_dataset": float("inf"),
        #     "n_samples_per_dataset": 1000,
        #     # "n_unique_datasets": float("inf"),
        #     "n_unique_datasets": 10,
        #     "prior": "gaussian",
        #     "prior_kwargs": {
        #         "mean": 0.0,
        #         "std_dev": 4.5,
        #     },
        #     "component": "gaussian",
        #     "component_kwargs": {
        #         "std_dev": 0.13,
        #     },
    },
    #     "inf_eval_dataset": {
    #         "dataset": "mixture_of_gaussians",
    #         # "dataset_length": 10000,
    #         "max_n_samples_in_context": 35,
    #         "n_components": 3,
    #         "n_dimensions": 2,
    #         "n_samples_per_dataset": float("inf"),
    #         "n_unique_datasets": float("inf"),
    #         "prior": "gaussian",
    #         "prior_kwargs": {
    #             "mean": 0.0,
    #             "std_dev": 4.5,
    #         },
    #         "component": "gaussian",
    #         "component_kwargs": {
    #             "std_dev": 0.33,
    #         },
    #     },
    # },
    "eval_kwargs": {
        "n_meshgrid_points_in_1D_slice": 13,
    },
    "energy_regularization": .001,
    "gradient_clip_val": 0.1,
    "learning_rate": 0.001,
    "learning_rate_scheduler": "None",
    "log_every_n_steps": 1,
    "mcmc_kwargs": {
        "algorithm": "langevin_mcmc",
        "all_steps": False,
        "n_mcmc_steps": 10,
        "gradient_clip_val": 1.0,
        "step_size": 1.0,  # Yilun says large step sizes e.g., 100.0, 10. are good.
        "kl_coeff": 0.0,
        "ratio_of_confabulated_samples_to_real_samples": 3,
        "replay_buffer": False,  # Use MCMC chains initialized from a replay buffer.
        "replay_buffer_size": 10000,
        "resampling_rate": 0.001,
        "noise_scale": 2.0,
    },
    "model_kwargs": {
        "activation": "gelu",
        "d_model": 64,
        # "dropout": 0.1,
        "dropout": 0.0,
        "n_heads": 8,
        "n_layers": 3,
    },
    "n_batches_per_epoch": 10,
    "n_epochs": 200,
    "n_workers": 8,
    "optimizer": "adam",
    # "precision": 32,
    # "precision": "16-mixed",
    "precision": "bf16-mixed",  # Supported on CPU. 16-mixed is not.
    "seed": 0,
    "weight_decay": 1e-5,
}
