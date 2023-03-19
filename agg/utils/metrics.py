"""Copyright 2023 by @jet981217. All rights reserved."""
import torch


def isotropy(
    word_embedding_matrix: torch.Tensor,
    batch_process: bool
) -> torch.Tensor:
    """Calculate isotropy of word embedding matrix

    Args:
        word_embedding_matrix (torch.Tensor):
            Word embedding matrix of a model
        batch_process (bool):
            Process in batch or not

    Returns:
        torch.Tensor:
            Isotropy value
    """
    unit_vectors = torch.nn.functional.normalize(
        input = word_embedding_matrix,
        dim = -1
    )

    if batch_process:
        Z_a = torch.sum(
            torch.exp(word_embedding_matrix.matmul(unit_vectors.T)),
            dim=0
        )
        min_value, _ = torch.min(Z_a, dim=-1)
        max_value, _ = torch.max(Z_a, dim=-1)

        return min_value/max_value
    
    min_val = torch.Tensor([float("inf")])
    max_val = torch.Tensor([0.])
    
    for unit_vector in unit_vectors:
        Z = torch.sum(
            torch.exp(word_embedding_matrix.matmul(unit_vector.T)),
            dim=-1
        )
        if Z < min_val[0]:
            min_val[0] = Z
        if Z >= max_val[0]:
            max_val[0] = Z
    return min_val[0]/max_val[0]
