import os

import src.utils

# For debugging, choose 1 GPU.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# Rok asked us to include the following specifications in our code to prevent CPUs from spinning idly:
n_threads_str = "4"
os.environ["OMP_NUM_THREADS"] = n_threads_str
os.environ["OPENBLAS_NUM_THREADS"] = n_threads_str
os.environ["MKL_NUM_THREADS"] = n_threads_str
os.environ["VECLIB_MAXIMUM_THREADS"] = n_threads_str
os.environ["NUMEXPR_NUM_THREADS"] = n_threads_str


import json
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
import pprint
import torch
from typing import Any, Dict, List, Union
import wandb


from src.data import InContextLearningEnergyBasedModelDataModule
from src.globals import default_config
from src.run import set_seed
from src.systems import (
    InContextLearningEnergyBasedModelEvaluationCallback,
    InContextLearningEnergyBasedModelSystem,
)


def main(wandb_config: Dict[str, Any]):
    set_seed(seed=wandb_config["seed"])

    wandb_logger = WandbLogger(experiment=run)

    icl_ebm_system = InContextLearningEnergyBasedModelSystem(
        wandb_config=wandb_config, wandb_logger=wandb_logger
    )

    datamodule = InContextLearningEnergyBasedModelDataModule(
        wandb_config=wandb_config,
    )

    # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.trainer.Trainer.html
    trainer = pl.Trainer(
        accumulate_grad_batches=wandb_config["accumulate_grad_batches"],
        callbacks=[
            ModelCheckpoint(
                monitor="train/total_loss",
                save_top_k=1,
                mode="min",
                save_on_train_epoch_end=True,
                dirpath=run_checkpoint_dir,
            ),
            # DeviceStatsMonitor(),
            # Stop if training loss diverges.
            # EarlyStopping(
            #     monitor="train/total_loss", patience=int(1e12), check_finite=True
            # ),
            LearningRateMonitor(logging_interval="step", log_momentum=True),
            InContextLearningEnergyBasedModelEvaluationCallback(
                wandb_config=wandb_config,
            ),
        ],
        # check_val_every_n_epoch=wandb_config["check_val_every_n_epoch"],
        default_root_dir=run_checkpoint_dir,
        deterministic=True,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        # devices="4",
        # strategy='ddp',
        # fast_dev_run=True,
        # fast_dev_run=False,
        logger=wandb_logger,
        log_every_n_steps=wandb_config["log_every_n_steps"],
        # overfit_batches=1,  # useful for debugging
        gradient_clip_val=wandb_config["gradient_clip_val"],
        max_epochs=wandb_config["n_epochs"],
        num_sanity_val_steps=0,  # -1 means runs all of validation before starting to train.
        # limit_train_batches=0.01,
        profiler="simple",  # Simplest profiler.
        # profiler="advanced",  # More advanced profiler.
        # profiler=PyTorchProfiler(filename=),  # PyTorch specific profiler
        precision=wandb_config["precision"],
        strategy="ddp_find_unused_parameters_true",
        sync_batchnorm=True,
    )

    if wandb_config["compile_model"] and hasattr(torch, "compile"):
        # Compile model if PyTorch supports it.
        print("Milestone: Compiling pretrain system...")
        icl_ebm_system = torch.compile(icl_ebm_system)
        print("Milestone: Compiled pretrain system.")

    # Pretrain.
    trainer.fit(model=icl_ebm_system, datamodule=datamodule)

    print("Milestone: Finished training.")


# .fit() needs to be called below for multiprocessing.
# See: https://github.com/Lightning-AI/lightning/issues/13039
# See: https://github.com/Lightning-AI/lightning/discussions/9201
# See: https://github.com/Lightning-AI/lightning/discussions/151
if __name__ == "__main__":
    wandb_username = src.utils.get_wandb_username()
    run = wandb.init(
        project="icl-ebm",
        config=default_config,
        entity=wandb_username,
    )
    wandb_config = dict(wandb.config)

    # Convert "None" (type: str) to None (type: NoneType)
    for key in [
        "accumulate_grad_batches",
        "gradient_clip_val",
        "learning_rate_scheduler",
    ]:
        if isinstance(wandb_config[key], str):
            if wandb_config[key] == "None":
                wandb_config[key] = None

    # Create checkpoint directory for this run, and save the config to the directory.
    run_checkpoint_dir = os.path.join("lightning_logs", wandb.run.id)
    os.makedirs(run_checkpoint_dir)
    wandb_config["run_checkpoint_dir"] = run_checkpoint_dir
    with open(os.path.join(run_checkpoint_dir, "wandb_config.json"), "w") as fp:
        json.dump(obj=wandb_config, fp=fp)

    pp = pprint.PrettyPrinter(indent=4)
    print("W&B Config:")
    pp.pprint(wandb_config)

    # Pretrain and evaluate!
    main(wandb_config=wandb_config)

    print("Done!")