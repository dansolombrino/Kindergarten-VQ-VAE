from rich import print

import torch.nn as nn
import torch
from torch import argmax
from torch.nn.functional import softmax

from transformers import EncoderDecoderModel

from consts import *

from models.utils import *

SUPPORTED_MODEL_MODES = ["full", "dec-head-ft"]


class Bagon(nn.Module):
    def __init__(
        self, 
        encoder_model_name: str, 
        decoder_model_name: str
    ):
        super(Bagon, self).__init__()

        encoder_decoder_model: EncoderDecoderModel = EncoderDecoderModel.from_encoder_decoder_pretrained(encoder_model_name, decoder_model_name)

        self.encoder = encoder_decoder_model.encoder
    
        self.decoder = encoder_decoder_model.decoder

        

    def forward(self, input_ids, attention_mask):
        
        embeds = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state

        # reconstructed_logits = self.decoder(inputs_embeds=embeds).logits
        reconstructed_logits = self.decoder(encoder_hidden_states=embeds, input_ids=input_ids, attention_mask=attention_mask).logits

        return reconstructed_logits
    
    def model_params_summary_dict(self):
        return {
            "encoder": {
                "n_trainable_params": n_trainable_params(self.encoder),
                "n_not_trainable_params": n_not_trainable_params(self.encoder),
                "n_params": n_params(self.encoder)
            },
            
            "decoder": {
                "n_trainable_params": n_trainable_params(self.decoder),
                "n_not_trainable_params": n_not_trainable_params(self.decoder),
                "n_params": n_params(self.decoder)
            }
        }

    
    def model_params_summary_print(self):

        print_module_params_summary(
            self.encoder, "Encoder", COLOR_TRAIN, COLOR_FROZEN, COLOR_TOT
        )
        
        print_module_params_summary(
            self.decoder, "Decoder", COLOR_TRAIN, COLOR_FROZEN, COLOR_TOT
        )
        
        return
    
    def set_mode(self, model_mode: str):
        if model_mode == "full":
            return
        
        if model_mode == "dec-head-ft":
            # Layers composing BERT classification head, in Huggingface implementation:
            # - decoder.cls.predictions.transform.dense 
            # - decoder.cls.predictions.decoder

            # NOT possible to freeze every param except the ones in the desired layers
            # So, we first freeze the BERT encoder and the BERT decoder
            for param in self.encoder.parameters():
                param.requires_grad = False
            
            for param in self.decoder.parameters():
                param.requires_grad = False

            # (Vector Quantizer parameters are kept trainable!)
            
            # Then, we unfreeze the parameters of the layers we're interested in
            for p in self.decoder.cls.predictions.transform.dense.parameters():
                p.requires_grad = True

            for p in self.decoder.cls.predictions.decoder.parameters():
                p.requires_grad = True

            return
        
        raise ValueError(f"Invalid model mode {model_mode}, please use one of the following: {', '.join(SUPPORTED_MODEL_MODES)}")

    




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