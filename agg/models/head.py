"""Module of classification head for every Ugly Classifier models"""

import numpy as np
import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.activations import get_activation

# pyright: reportIncompatibleMethodOverride=false
# pyright: reportGeneralTypeIssues=false
# pylint: disable=invalid-name


class ClassificationHead(nn.Module):
    """ClassificationHead class for every LM"""

    def __init__(
        self,
        model_config: PretrainedConfig,
        hidden_size: int,
        num_labels: int,
        pooling_method: str,
    ) -> None:
        """Classification head used for every UT models

        Args:
            model_config (PretrainedConfig):
                Config of LM
            hidden_size (int):
                Size of hidden layer output
            num_labels (int):
                Number of categories
            pooling_method (str):
                Polling method of hidden state.
                Can only be "cls" or "mean"
        """
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        classifier_dropout = (
            model_config.classifier_dropout
            if model_config.classifier_dropout is not None
            else model_config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)
        self.pooling_method = pooling_method

    @staticmethod
    def pool_output(
        embedding: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_method: str,
    ) -> torch.Tensor:
        """Pooling function for hidden state

        Args:
            embedding (torch.Tensor):
                Output embedding of hidden layer
            attention_mask (torch.Tensor):
                Attention mask of input
            pooling_method (str):
                Polling method of hidden state.
                Can only be "cls" or "mean"

        Returns:
            torch.Tensor: _description_
        """
        if pooling_method == "cls":
            return embedding[:, 0, :]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(embedding.size()).float()
        )
        sum_embedding = torch.sum(embedding * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_embedding / sum_mask

    def forward(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward function of the head

        Args:
            features (torch.Tensor):
                Output embeddings of LM
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor:
                Output of Classification head
        """
        x = ClassificationHead.pool_output(
            embedding=features,
            attention_mask=kwargs["attention_mask"],
            pooling_method=self.pooling_method,
        )
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = get_activation("gelu")(x)
        # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class MLMHead(nn.Module):
    """ClassificationHead class for every LM when doing MLM"""

    def __init__(
        self,
        model_config: PretrainedConfig,
        embedding_layer: torch.nn.Embedding,
        weight_tying: bool,
    ) -> None:
        """Classification head used for every Transformer models

        Args:
            model_config (PretrainedConfig):
                Config of LM
            embedding_layer (torch.nn.Embedding):
               Embedding layer to use weight tying
               with Linear layer on MLM head
            weight_tying (bool):
                Apply weight tying or not
        """
        super().__init__()
        classifier_dropout = (
            model_config.classifier_dropout
            if model_config.classifier_dropout is not None
            else model_config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = torch.nn.Linear(
            *np.array(list(embedding_layer.weight.shape))[::-1]
        )
        if weight_tying:
            self.out_proj.weight = embedding_layer.weight

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward function of the head

        Args:
            features (torch.Tensor):
                Output embeddings of LM

        Returns:
            torch.Tensor:
                Output of Classification head
        """
        x = self.dropout(features)
        return self.out_proj(x)
