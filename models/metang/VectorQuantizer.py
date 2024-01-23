from rich import print

import torch
import torch.nn as nn


class VectorQuantizer(nn.Module):

    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z: torch.Tensor, device):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width) --> original paper
        z.shape = (batch, channel, seq_len)       --> Kindergarten-VQ-VAE
        channel corresponds --> embedding_dim
        height, width or seq_len --> number of data features (i.e. spatial or temporal dimension)

        quantization pipeline:

            1. get encoder input ((B,C,H,W) in original paper, (B, S, C) in Kindergarten-VQ-VAE)
            2. flatten input to (B*H*W,C) in original, (X, X, X) in Kindergarten-VQ-VAE

        """

        # print(f"[VectorQuantizer] z.shape: {z.shape}\n")

        # flatten input: (B, S, C) --> (B*S, C)
        z_flattened = z.view((-1, self.e_dim))
        # print(f"[VectorQuantizer] z_flattened.shape: {z_flattened.shape}\n")

        # distances from z to embeddings e_j --> (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())
        # print(f"[VectorQuantizer] d.shape: {d.shape}\n")

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1).to(device)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach() - z) ** 2) + \
            self.beta * torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape --> NOT needed in Kindergarten-VQ-VAE
        # z_q = z_q.permute(0, 3, 1, 2).contiguous()

        # print(f"[VectorQuantizer] z_q.shape: {z_q.shape}\n")

        return loss, z_q, perplexity, min_encodings, min_encoding_indices




def main():

    batch_size = 16
    seq_len = 7
    enc_out_dim = 768

    n_e = 10
    e_dim = enc_out_dim
    beta = 0.69
    
    vq = VectorQuantizer(n_e=n_e, e_dim=e_dim, beta=beta)

    z_e = torch.rand((batch_size, seq_len, enc_out_dim))

    pre_quantization_map = nn.Identity() if enc_out_dim == e_dim else nn.Linear(in_features=enc_out_dim, out_features=e_dim)
    z_e = pre_quantization_map(z_e)

    loss, z_q, perplexity, min_encodings, min_encoding_indices = vq.forward(z_e)












if __name__ == "__main__":
    main()