from __future__ import annotations

import os
import pickle
import sys

import torch
from torch import nn
from transformers import BartConfig

sys.path.insert(0, os.getcwd() + '/train/')  # to import modules in train
import train.train_autoencoder
from models.localattn import LoBART


class DocDictionary:

    def __init__(self, model_file_path: str, model_config_path: str, device: torch.device | str):
        with open(model_config_path, "rb") as config_file:
            config = pickle.load(config_file)

        max_length: int = config.max_position_embeddings
        vocab_length: int = config.max_position_embeddings

        # TODO: run in GPU
        device = "cpu"
        self.dict_tensor = torch.rand((max_length, vocab_length), requires_grad=True).to(device)
        self.autoencoder = _load_autoencoder(model_file_path, config, device)
        self.device = device

    def get_dict_loss(self, input_ids, attention_mask, encoder_logits) -> torch.Tensor:
        # TODO: figure out how to iterate over the 0th dimension
        assert encoder_logits.shape[0] == 1, "Use batches of 1 for now"
        loss = self._dict_loss(input_ids, attention_mask, encoder_logits[0] + self._ortho_loss())
        return loss

    def _dict_loss(self, input_ids, attention_mask, encoder_logits) -> torch.Tensor:
        """
        Calculates the ||h-Ae|| cost, where h the base model's logits, A the dict matrix and
        e the pretrained autoencoder's logits.
        @param encoder_logits: the base model's encoder's logits
        @return: the dictionary loss
        """
        h = self._pass_through_encoder(input_ids, attention_mask)
        diff = h - torch.mm(encoder_logits.to(self.device), self.dict_tensor)
        return torch.norm(diff, p=2)

    def _ortho_loss(self) -> torch.Tensor:
        """
        Calculates the ||A A^T - I|| cost, where A the dict matrix.
        @return: the orthogonality loss
        """
        a = self.dict_tensor
        diff = torch.mm(a, torch.t(a)) - torch.eye(a.size()[0], a.size()[1]).to(self.device)
        return torch.norm(diff, p=2)

    def _pass_through_encoder(self, input_ids, attention_mask) -> torch.Tensor:
        """
        Get the pretrained model's encoder's logits.
        @return: the logits
        """
        # BART forward
        x = self.autoencoder(input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device))
        # x[0] # decoder output
        # x[1] # encoder output
        lm_logits = x[1]
        return lm_logits


def _load_autoencoder(model_file_path: str, config: BartConfig, device: str | torch.device) -> nn.Module:
    model = LoBART(config)
    state_dict = torch.load(model_file_path)
    model.load_state_dict(state_dict)
    model.eval()

    if device == 'cuda':
        model.cuda()

    return model


def main():
    model_path: str = os.path.join(train.train_autoencoder.MODEL_DIR, train.train_autoencoder.MODEL_FILE_NAME)
    config_path = os.path.join(train.train_autoencoder.MODEL_DIR, train.train_autoencoder.MODEL_CONFIG_NAME)
    if torch.cuda.is_available():
        print("Using GPU")
        torch_device = 'cuda'
    else:
        print("Using CPU")
        torch_device = 'cpu'

    dictionary = DocDictionary(model_path, config_path, torch_device)


if __name__ == "__main__":
    main()
