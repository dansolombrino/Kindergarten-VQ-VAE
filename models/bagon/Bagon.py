from rich import print

import torch.nn as nn
import torch
from torch import argmax
from torch.nn.functional import softmax

from transformers import EncoderDecoderModel

from VectorQuantizer import VectorQuantizer



class Bagon(nn.Module):
    def __init__(
        self, encoder_model_name: str, 
        vq_n_e: int, vq_e_dim: int, vq_beta: float,
        decoder_model_name: str
    ):
        super(Bagon, self).__init__()

        encoder_decoder_model: EncoderDecoderModel = EncoderDecoderModel.from_encoder_decoder_pretrained(encoder_model_name, decoder_model_name)

        self.encoder = encoder_decoder_model.encoder
        
        self.vector_quantizer = VectorQuantizer(n_e=vq_n_e, e_dim=vq_e_dim, beta=vq_beta)
        
        self.decoder = encoder_decoder_model.decoder

        

    def forward(self, input_ids, device):
        
        embeds = self.encoder(input_ids).last_hidden_state

        assert embeds.shape[-1] == self.vector_quantizer.e_dim, "embedding dim of encoder output must match e_dim (for now)!"

        vq_loss, z_q, perplexity, min_encodings, min_encoding_indices = self.vector_quantizer(embeds, device)

        reconstructed_logits = self.decoder(inputs_embeds=z_q).logits

        return vq_loss, reconstructed_logits
    
    def get_num_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_num_not_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    




def main():
    from rich import print

    from transformers.utils import logging
    logging.set_verbosity(40)

    from transformers import BertTokenizer

    encoder_model_name = "bert-base-uncased"
    vq_n_e = 10
    vq_e_dim = 768
    vq_beta = 0.69
    decoder_model_name = "bert-base-uncased"

    model = Bagon(encoder_model_name, vq_n_e, vq_e_dim, vq_beta, decoder_model_name)

    tokenizer_name = "bert-base-uncased"
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    text = [
        "Today is an amazing day",
        "Tomorrow is the best day",
        "I am totally ready for this great adventure"
    ]
    
    text_input_ids = tokenizer(text, return_tensors="pt", padding=True).input_ids

    text_reconstructed_ids = model.forward(text_input_ids)

    for i, (a, b, c) in enumerate(zip(text_input_ids, text_reconstructed_ids, text)):
        print(f"sentence id: {i}")
        print(f"input_ids: {a}")
        print(f"reconstructed_ids: {b}")

        print(f"text: {c}")
        print(f"text_reconstructed: {' '.join(tokenizer.convert_ids_to_tokens(b.squeeze(0)))}")
        
        print("\n\n")












if __name__ == "__main__":
    main()