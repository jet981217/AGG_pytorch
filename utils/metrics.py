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
            torch.exp(unit_vectors.T.matmul(word_embedding_matrix)),
            dim=-1
        )
        return torch.min(Z_a, dim=-1)/torch.max(Z_a, dim=-1)
    
    min_val = torch.Tensor(float("inf"))
    max_val = torch.Tensor(0.)
    
    for unit_vector in unit_vectors:
        Z = torch.sum(
            torch.exp(word_embedding_matrix.matmul(unit_vector.T)),
            dim=-1
        )
        if Z < min_val:
            min_val = Z
        if Z >= max_val:
            max_val = Z
    return min_val/max_val
