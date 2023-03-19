"""Copyright 2023 by @jet981217. All rights reserved."""
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

from agg.models.head import MLMHead


class TransformerEncoder(nn.Module):
    """
    Class for implementing a Transformer Encoder.

    Args:
        num_layers (int): The number of layers in the encoder.
        hidden_size (int): The number of expected features in the input.
        num_heads (int): The number of heads in the multiheadattention models.
        intermediate_size (int): The size of the feedforward network model.

    Attributes:
        encoder (nn.TransformerEncoder): The Transformer encoder.

    Methods:
        forward(src: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            Performs forward propagation on Transformer Encoder.
    """
    def __init__(self, num_layers: int, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        # Stack Transformer EncoderLayer num_layers times.
        encoder_layer = nn.TransformerEncoderLayer(hidden_size, num_heads, intermediate_size)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs forward propagation on Transformer Encoder.

        Args:
            src (torch.Tensor): The input sequence to the encoder.
            mask (torch.Tensor, optional): The mask tensor used to hide input elements.
            src_key_padding_mask (torch.Tensor, optional): The mask tensor used to hide input padding.

        Returns:
            output (torch.Tensor): The output tensor of the encoder.
        """
        output = self.encoder(src, mask=mask, src_key_padding_mask=src_key_padding_mask)
        return output


class TransformerLM(PreTrainedModel):
    """
    Transformer-based language model.

    Args:
        config: A PretrainedConfig object containing the model configuration.
    """
    def __init__(self, model_config):
        super().__init__(model_config)
        self.embeddings = nn.Embedding(model_config.vocab_size, model_config.hidden_size, padding_idx=model_config.pad_token_id)
        self.encoder = TransformerEncoder(model_config.num_hidden_layers, model_config.hidden_size, model_config.num_attention_heads, model_config.intermediate_size)
        self.pooler = nn.Linear(model_config.hidden_size, 1)
        self.activation = nn.Tanh()
        self.pool_type = model_config.pool_type
        self.init_weights()
        self.__config = model_config

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Perform a forward pass on the model.

        Args:
            input_ids: A tensor of shape (batch_size, sequence_length) containing the input token IDs.
            attention_mask: A tensor of shape (batch_size, sequence_length) containing the attention masks for the input tokens.
            token_type_ids: A tensor of shape (batch_size, sequence_length) containing the token type IDs.
            position_ids: A tensor of shape (batch_size, sequence_length) containing the position IDs.
            head_mask: A tensor of shape (num_heads,) containing the head masks for the attention layer.
            inputs_embeds: A tensor of shape (batch_size, sequence_length, hidden_size) containing the input embeddings.
            output_attentions: A bool indicating whether to return attention weights.
            output_hidden_states: A bool indicating whether to return all hidden states.
            return_dict: A bool indicating whether to return a dictionary of outputs or a tuple.

        Returns:
            If `return_dict=False`, returns a tuple of two tensors:
                1. A tensor of shape (batch_size, sequence_length, hidden_size) containing the last hidden state.
                2. A tensor of shape (batch_size, hidden_size) containing the pooled output.

            If `return_dict=True`, returns a dictionary containing:
                1. 'last_hidden_state': A tensor of shape (batch_size, sequence_length, hidden_size) containing the last hidden state.
                2. 'pooler_output': A tensor of shape (batch_size, hidden_size) containing the pooled output.
        """
        return_dict = return_dict if return_dict is not None else self.__config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        
        # Embed the input sequence at each position.
        inputs_embeds = self.embeddings(input_ids)
        # Add embeddings representing position information of the input sequence.
        position_ids = torch.arange(input_shape[-1], dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)
        position_embeddings = self.embeddings(position_ids)
        inputs_embeds += position_embeddings
        # Add embeddings representing token type information of the input sequence.
        token_type_embeddings = self.embeddings(token_type_ids)
        inputs_embeds += token_type_embeddings
        
        # Encode input embeddings through Transformer Encoder.
        encoded_layers = self.encoder(
            inputs_embeds.transpose(0,1), src_key_padding_mask=attention_mask.float()
        )
        sequence_output = encoded_layers.transpose(0,1)

        # Generate final output through pooler layer.
        pooled_output = self.pooler(
            sequence_output[:, 0] if self.pool_type == "cls"
            else torch.mean(sequence_output, dim=1)
        )
        pooled_output = self.activation(pooled_output)
        
        if not return_dict:
            return (sequence_output, pooled_output)
        
        return {
            "last_hidden_state": sequence_output,
            "pooler_output": pooled_output,
        }


class TransformerMLM(nn.Module):
    """Transformer Module for MLM"""

    def __init__(
        self,
        LM: PreTrainedModel,
        model_config: PretrainedConfig
    ) -> None:
        """KoElcetra model implementation for mlm

        Args:
            model_config (PretrainedConfig):
                Config of LM
            LM (PreTrainedModel):
                Pretrained LM
            weight_tying (bool):
                Apply weight tying or not
        """
        super().__init__()
        self.hidden_size = model_config.hidden_size

        self.LM = LM

        self.__head = MLMHead(
            model_config=model_config,
            embedding_layer=LM.embeddings,
            weight_tying=True,
        )
        self.__config = model_config

        # Initialize weights and apply final processing
        self.post_init()


    def post_init(self):
        # Initialize weights of LM's embedding layer
        with torch.no_grad():
            self.LM.embeddings.weight.normal_(mean=0.0, std=self.LM.config.initializer_range)
            if self.LM.config.pad_token_id is not None:
                self.LM.embeddings.weight[self.LM.config.pad_token_id].zero_()


    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.Tensor:
        """Forward

        Args:
            input_ids (torch.Tensor): Input tensor.
            attention_mask (torch.Tensor): Attention mask for input.
            token_type_ids (Optional[torch.Tensor], optional): Defaults to None.
            position_ids (Optional[torch.Tensor], optional): Defaults to None.
            head_mask (Optional[torch.Tensor], optional): Defaults to None.
            inputs_embeds (Optional[torch.Tensor], optional): Defaults to None.
            output_attentions (Optional[bool], optional): Defaults to None.
            output_hidden_states (Optional[bool], optional): Defaults to None.
            return_dict (Optional[bool], optional): Defaults to None.

        Returns:
            torch.Tensor:
                Output logits of the model
        """
        return_dict = (
            return_dict
            if return_dict is not None
            else self.__config.use_return_dict
        )

        discriminator_hidden_states = self.LM(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return self.__head(discriminator_hidden_states["last_hidden_state"])
