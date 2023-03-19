"""Copyright 2023 by @jet981217. All rights reserved."""
import logging
from typing import Dict, Tuple

import torch
from transformers import logging as transformers_logging

# pyright: reportGeneralTypeIssues=false
# pylint: disable=logging-fstring-interpolation, too-many-branches


def EvalerMLM(
    model: torch.nn.parallel.DataParallel,
    dataloader: torch.utils.data.DataLoader,
    train_config: Dict,
) -> Tuple[float, float, float]:
    """Evaluate function for validating

    Args:
        model (torch.nn.parallel.DataParallel): Model to eval
        dataloader (torch.utils.data.DataLoader):
            A single dataloader to use for evaluating
        train_config (Dict):
            Training config dict

    Returns:
        Tuple[float, float, Dict]: Eval loss
    """
    # Eval!

    eval_loss = 0
    num_sample = 0

    right = 0
    total = 0

    right_token = 0
    total_token = 0

    for batch in dataloader:
        model.eval()

        with torch.no_grad():
            batch_id, batch_mask, labels, masked_idxs = batch
            outputs = []

            inputs = {
                "input_ids": batch_id.to(train_config["device"]),
                "attention_mask": batch_mask.to(train_config["device"]),
            }
            is_masked = int(len(masked_idxs) > 0).to(train_config["device"])

            outputs = model(**inputs)

            criterion = torch.nn.CrossEntropyLoss()

            loss = criterion(
                outputs.reshape(-1, outputs.shape[-1]),
                labels.to(train_config["device"]).reshape(-1),
            )

            diff = torch.sum(
                (
                    torch.abs(
                        torch.argmax(outputs, dim=-1)
                        - labels.to(train_config["device"])
                    )
                ),
                axis=-1,
            )

            if is_masked.count_nonzero().item() > 0:
                total += is_masked.count_nonzero().item()
                right += (
                    len(diff)
                    - torch.count_nonzero(diff * is_masked)
                    - len(is_masked)
                    + is_masked.count_nonzero().item()
                )

            eval_loss += loss.item()
            num_sample += len(batch_id)

            for batch_idx, batch_match in enumerate(
                batch_id.to(train_config["device"])
                != labels.to(train_config["device"])
            ):
                for token_idx in batch_match.nonzero(as_tuple=True)[0]:
                    if (
                        labels[batch_idx][token_idx.item()]
                        in outputs[batch_idx][token_idx.item()].topk(5).indices
                    ):
                        right_token += 1
                    total_token += 1

    return (
        eval_loss / num_sample,
        100 * right / total if total > 0 else -1,
        100 * right_token / total_token if total_token > 0 else -1,
    )
