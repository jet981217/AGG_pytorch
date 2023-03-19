"""Module to check conditions of experiment"""
import os
from typing import Dict

from agg.utils.utils import MODEL_ALIAS, TASKS

AVAILABLE_TASKS = sum((value for _, value in TASKS.items()), start=[])
AVAILABLE_MODELS = sum((value for _, value in MODEL_ALIAS.items()), start=[])

# pyright: reportGeneralTypeIssues=false


def check_conditions(train_config: Dict, train_mode: str) -> str:
    """Method to map check condition method of train config

    Args:
        train_config (Dict):
            Dict of train config
        train_mode (str):
            Mode of train that is currently used.
            It could only be finetune or mlm

    Returns:
        str: Root path to save checkpoint and logs
    """
    assert train_mode in [
        "finetune",
        "mlm",
    ], "mode for finetune and pretrain is only available!"

    if train_mode == "finetune":
        return check_finetune_conditions(
            train_config=train_config, train_mode=train_mode
        )
    return mlm_pretrain_condition(
        train_config=train_config, train_mode="pretrain"
    )


def check_finetune_conditions(train_config: Dict, train_mode: str) -> str:
    """Method to assert every condition of training config dict when fintuning

    Args:
        train_config (Dict):
            Dict of train config
        train_mode (str):
            Mode of train that is currently used.
            It could only be finetune or mlm

    Returns:
        str: Root path to save checkpoint and logs
    """
    _ = train_mode
    assert (
        train_config["task"] in AVAILABLE_TASKS
    ), f"Task should only be one of {AVAILABLE_TASKS}"
    assert (
        train_config["model"] in AVAILABLE_MODELS
    ), f"Model type should only be one of {AVAILABLE_MODELS}"

    if train_config["custom_tokenizer_path"] != "":
        assert train_config["task"] not in [
            "lp",
            "linearprobe",
            "LinearProbe",
            "wef",
            "wordembeddingfreeze",
            "WordEmbeddingFreeze",
        ], (
            f"Your task cannnot be {train_config['task']}"
            f" when you use a custom tokenizer!!!"
        )
    assert (
        train_config["PLM_version"][0] == "v"
    ), "The name of PLM_version should start with 'v'"
    assert (
        train_config["dataset_version"][0] == "v"
    ), "The name of dataset_version should start with 'v'"

    assert (
        os.path.exists(train_config["experiment_root"])
        and len(os.listdir(train_config["experiment_root"])) > 0
    ), f"Trial {train_config['experiment_root']} is wrong!!"

    if train_config["optimizer_path"] != "":
        assert os.path.exists(train_config["optimizer_path"]), (
            f"You have given optimizer status as a path"
            f" {train_config['optimizer_path']} "
            f"but it does not exist."
        )
    if train_config["scheduler_path"] != "":
        assert os.path.exists(train_config["optimizer_path"]), (
            f"You have given scheduler status as a path"
            f" {train_config['scheduler_path']} "
            f"but it does not exist."
        )

    assert train_config["train_params"]["pooling_method"] in [
        "cls",
        "mean",
    ], "Wrong pooling method!!!"

    for mode in ["train", "val", "test"]:
        assert (
            train_config["batch_size"][mode] > 0
        ), f"Batch size of {mode} must be bigger than 0"

    for param_option in [
        key
        for key in list(train_config["train_params"].keys())
        if key
        not in ["max_steps", "no_cuda", "pooling_method", "save_optimizer"]
    ]:
        assert (
            train_config["train_params"][param_option] >= 0
        ), f"{param_option} can not be a negative value!"

    if not os.path.exists(
        f"{train_config['output_root']}/{train_config['experiment_root']}/"
        f"{train_config['train_run_name']}"
    ):
        os.makedirs(
            f"{train_config['output_root']}/{train_config['experiment_root']}/"
            f"{train_config['train_run_name']}"
        )
    else:
        # Checks if the log dir of this current experiment trial exists
        assert not os.listdir(
            f"{train_config['output_root']}/{train_config['experiment_root']}/"
            f"{train_config['train_run_name']}"
        ), "Trial already exists!!!"

    # Returns the path of output log directory
    return (
        f"{train_config['output_root']}/{train_config['experiment_root']}/"
        f"{train_config['train_run_name']}"
    )


def mlm_pretrain_condition(train_config: Dict, train_mode: str) -> str:
    """Method to assert every condition of training config dict for mlm

    Args:
        train_config (Dict):
            Dict of train config
        train_mode (str):
            Mode of train that is currently used.
            It could only be finetune or mlm

    Returns:
        str: Root path to save checkpoint and logs
    """
    assert (
        train_config["model"] in AVAILABLE_MODELS
    ), f"Model type should only be one of {AVAILABLE_MODELS}"

    assert os.path.exists(
        f"configs/{train_mode}_configs/"
        f"{train_config['config_file_path']}.json"
    ), f"Trial {train_config['config_file_path']} is wrong!!"

    if train_config["optimizer_path"] != "":
        assert os.path.exists(train_config["optimizer_path"]), (
            f"You have given optimizer status as a path"
            f" {train_config['optimizer_path']} "
            f"but it does not exist."
        )
    if train_config["scheduler_path"] != "":
        assert os.path.exists(train_config["optimizer_path"]), (
            f"You have given scheduler status as a path"
            f" {train_config['scheduler_path']} "
            f"but it does not exist."
        )

    for mode in ["train", "val", "test"]:
        assert (
            train_config["batch_size"][mode] > 0
        ), f"Batch size of {mode} must be bigger than 0"

    for param_option in [
        key
        for key in list(train_config["train_params"].keys())
        if key
        not in ["max_steps", "no_cuda", "pooling_method", "save_optimizer"]
    ]:
        assert (
            train_config["train_params"][param_option] >= 0
        ), f"{param_option} can not be a negative value!"

    if not os.path.exists(
        f"{train_config['output_root']}/{train_config['config_file_path']}"
    ):
        os.makedirs(
            f"{train_config['output_root']}/{train_config['config_file_path']}"
        )
    else:
        # Checks if the log dir of this current experiment trial exists
        assert not os.listdir(
            f"{train_config['output_root']}/{train_config['config_file_path']}"
        ), "Trial already exists!!!"

    # Returns the path of output log directory
    return f"{train_config['output_root']}/{train_config['config_file_path']}"
