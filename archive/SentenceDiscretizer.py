from torch import nn

from torch import Tensor

from torch.nn.functional import pad
from torch.nn.functional import gumbel_softmax

from models.shelgon2.GenerativeFactorDiscretizer import GenerativeFactorDiscretizer

import torch


class SentenceDiscretizer(nn.Module):
    def __init__(
        self,
        word_embedding_size: int,
        sentence_length: int
    ):
        super(SentenceDiscretizer, self).__init__()

        self.NUM_LATENT_GENERATIVE_FACTORS = 8
        self.MAX_SUPPORT_SIZE_LATENT_GENERATIVE_FACTORS = 3

        self.sentence_to_latent_factors = nn.Linear(
            in_features=word_embedding_size, 
            out_features=self.NUM_LATENT_GENERATIVE_FACTORS
        )

        self.latent_factors_expand = nn.Linear(
            in_features=1, 
            out_features=self.MAX_SUPPORT_SIZE_LATENT_GENERATIVE_FACTORS,
        )

        self.expand_word_embedding_dim = nn.Linear(
            in_features=self.MAX_SUPPORT_SIZE_LATENT_GENERATIVE_FACTORS, 
            out_features=word_embedding_size
        )

        self.expand_sentence_length_dim = nn.Conv1d(
            in_channels=self.NUM_LATENT_GENERATIVE_FACTORS,
            out_channels=sentence_length,
            kernel_size=1
        )

    def forward(self, embedded_sentences: Tensor):

        # print(f"embedded_sentences.shape: {embedded_sentences.shape}")

        latent_factors = self.sentence_to_latent_factors(embedded_sentences)

        # print(f"latent_factors.shape: {latent_factors.shape}")

        latent_factors = torch.unsqueeze(latent_factors, -1)

        # print(f"latent_factors.shape: {latent_factors.shape}")

        latent_factors = self.latent_factors_expand(latent_factors)

        # print(f"latent_factors.shape: {latent_factors.shape}")

        latent_factors = gumbel_softmax(logits=latent_factors, dim=-1)

        # print(f"latent_factors.shape: {latent_factors.shape}")

        latent_sentences = self.expand_word_embedding_dim(latent_factors)

        # print(f"latent_sentences.shape: {latent_sentences.shape}")

        latent_sentences = self.expand_sentence_length_dim(latent_sentences)

        # print(f"latent_sentences.shape: {latent_sentences.shape}")

        return latent_sentences, latent_factors, latent_factors.argmax(-1)