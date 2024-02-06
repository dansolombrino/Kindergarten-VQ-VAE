from torch import nn

from torch.nn.functional import gumbel_softmax

from torch import Tensor

import torch

class GenerativeFactorDiscretizer(nn.Module):
    def __init__(
        self,
        word_emb_size: int,
        gen_factor_num_values: int
    ):
        
        super(GenerativeFactorDiscretizer, self).__init__()

        self.proj_in = nn.Linear(in_features=word_emb_size, out_features=gen_factor_num_values)

        self.proj_out = nn.Linear(in_features=gen_factor_num_values, out_features=word_emb_size)
        

    def forward(self, embedded_sentences: Tensor) -> (Tensor, Tensor):

        gen_factor_logits = self.proj_in(embedded_sentences)

        gen_factor_logits = gumbel_softmax(logits=gen_factor_logits, dim=-1)

        with torch.no_grad():
            gen_factor_label = torch.argmax(input=gen_factor_logits, dim=-1)

        embedded_sentences = self.proj_out(gen_factor_logits)

        return embedded_sentences, gen_factor_logits, gen_factor_label