# original code from https://github.com/karpathy/deep-vector-quantization and https://github.com/SerezD/vqvae-vqgan-pytorch-lightning
# adapted to match text sequences, as original is for images

from rich import print

from torch import nn

from torch import Tensor

import torch

from torch import einsum

from torch.nn import functional as F

class GumbelQuantizer(nn.Module):
    """
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """
    def __init__(
        self, 
        # num_hiddens, 
        enc_out_size, 
        n_embed, embedding_dim, 
        straight_through=False
    ):
        super().__init__()

        # self.embedding_dim = embedding_dim
        self.e_dim = embedding_dim
        self.n_embed = n_embed

        self.straight_through = straight_through
        self.temperature = 1.0
        self.kld_scale = 5e-4

        # self.proj = nn.Conv2d(num_hiddens, n_embed, 1)
        self.proj = nn.Conv1d(enc_out_size, n_embed, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)

    def forward(self, z: Tensor, is_training: bool):

        # print(f"[GumbelQuantizer] z.shape     : {z.shape}")
        # z.shape --> (batch_size, sequence_length, embedding_size)

        # (b, s, e) --> (b, e, s) to comply w/ Gumbel-Softmax vector quantization
        z = z.permute((0, 2, 1))
        # print(f"[GumbelQuantizer] z.shape     : {z.shape}")

        # force hard = True when we are in eval mode, as we must quantize
        hard = self.straight_through if is_training else True

        # mapping embedding_size to n_embed to comply w/ Gumbel-Softmax vector quantization
        logits = self.proj(z)
        # print(f"[GumbelQuantizer] logits.shape: {logits.shape}")
        
        soft_one_hot = F.gumbel_softmax(logits, tau=self.temperature, dim=1, hard=hard)
        # print(f"[GumbelQuantizer] logits.shape: {logits.shape}")

        # z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight)
        # replacing putting the corresponding latent vector embedding according 
        # to the soft_one_hot indexes computed before
        # z_q.shape --> (batch_size, latent_vector_size, sequence_lenth)
        z_q = einsum('b n s, n d -> b d s', soft_one_hot, self.embed.weight)
        # print(f"[GumbelQuantizer] z_q.shape   : {z_q.shape}")

        # + kl divergence to the prior loss (prevents index collapse)
        qy = F.softmax(logits, dim=1)
        # print(f"[GumbelQuantizer] qy.shape    : {qy.shape}")

        diff = self.kld_scale * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()
        # print(f"[GumbelQuantizer] diff.shape  : {diff.shape}")

        ind = soft_one_hot.argmax(dim=1)
        # print(f"[GumbelQuantizer] ind.shape   : {ind.shape}")

        # permuting z_q back to og shape needed by decoder
        # (batch_size, latent_vector_size, sequence_lenth) --> (batch_size, seq_len, latent_vector_size)
        z_q = z_q.permute((0, 2, 1))

        return z_q, diff, ind