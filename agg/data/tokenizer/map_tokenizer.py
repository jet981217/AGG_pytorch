"""Module to map tokenizer with your config file"""
from typing import Dict

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
)

from agg.utils.utils import MODEL_ALIAS

#tokenizer = AutoTokenizer.from_pretrained("vanilla-transformer")
def map_tokenizer(
    train_config: Dict,
) -> PreTrainedTokenizer:
    """Map a tokenizer with your config

    Args:
        train_config (Dict): Config file of your training

    Raises:
        Exception: When your model choice does not fit UT.

    Returns:
        PreTrainedTokenizer: Output tokenizer.
    """
    model_type = train_config["model"]
    custom_tokenizer = train_config["custom_tokenizer_path"]

    if model_type in MODEL_ALIAS["vanilla-transformer-base"]:
        return AutoTokenizer.from_pretrained(
            "bert-base-uncased"
            if custom_tokenizer == ""
            else custom_tokenizer
        )
    raise Exception(
        f"Your model choice is not available right now. "
        f"Try to choose a model in {MODEL_ALIAS.keys()}"
    )
