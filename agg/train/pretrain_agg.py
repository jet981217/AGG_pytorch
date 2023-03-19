"""Copyright 2023 by @jet981217. All rights reserved."""
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
from transformers import AdamW
from transformers import logging as transformers_logging

from agg.data.prepare_dataloader import prepare_mlm_dataloader
from agg.models import map_mlm_model
from agg.utils.utils import set_seed
from agg.engine.pretrain.trainer import TrainerMLM
from agg.utils.check_conditions import check_conditions

logger = logging.getLogger(__name__)

transformers_logging.set_verbosity_error()

# pyright: reportGeneralTypeIssues=false
# pylint: disable=logging-fstring-interpolation, too-many-branches


def main(train_config: Dict, train_log_root: Union[Path, str]) -> None:
    """Main function to initiate trainin
    g
        Args:
            train_config (Dict): Training config dict
            train_log_root (Union[Path, str]):
                Save root of checkpoint & log of the model
    """
    # Read from train_config file and make args
    set_seed(train_config["train_params"])

    # GPU or CPU
    train_config["device"] = (
        "cuda"
        if torch.cuda.is_available()
        and not train_config["train_params"]["no_cuda"]
        else "cpu"
    )
    model = map_mlm_model(train_config)

    model.to(train_config["device"])

    dataloaders = prepare_mlm_dataloader(train_config)

    TrainerMLM(
        train_config=train_config,
        model=model,
        dataloaders=dataloaders,
        log_root=train_log_root,
        pretrain_type="agg",
        token_ids_to_use=list(range(999,30522))
    )


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser()

    cli_parser.add_argument(
        "--experiment_configs", type=str, nargs="+", required=True
    )

    cli_args = cli_parser.parse_args()

    for experiment_config in cli_args.experiment_configs:
        assert (
            "." not in experiment_config
        ), "Your experiment config name cannot have file extension inside!"

        with open(
            f"configs/pretrain_configs/" f"{experiment_config}.json",
            "r",
        ) as train_config_file:
            train_config_dict = json.load(train_config_file)
        train_config_dict["config_file_path"] = experiment_config

        print(f"Pretraining by MLM {experiment_config}...")

        main(
            train_config=train_config_dict,
            train_log_root=check_conditions(
                train_config=train_config_dict, train_mode="mlm"
            ),
        )
