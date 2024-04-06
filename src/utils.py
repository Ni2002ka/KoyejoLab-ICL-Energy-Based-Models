import os


def get_wandb_username() -> str:
    if "WANDB_USERNAME" in os.environ:
        wandb_username = os.environ["WANDB_USERNAME"]
    else:
        system_username = os.environ.get("USER")
        if system_username == "mikail":
            wandb_username = "mikailkhona"
        elif system_username == "rschaef":
            wandb_username = "rylan"
        elif system_username == "nzahedi":
            wandb_username = "nzahedi"
        else:
            raise ValueError(
                f"Could not find W&B username for system user: {system_username}."
            )

    return wandb_username
