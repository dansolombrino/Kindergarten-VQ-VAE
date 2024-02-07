from rich import print

import torch.nn as nn
import torch

from transformers import EncoderDecoderModel

from common.consts import *

from common.model_utils import *

SUPPORTED_MODEL_MODES = ["full", "dec-head-ft"]


class Bagon(nn.Module):
    def __init__(
        self, 
        encoder_model_name: str, 
        decoder_model_name: str
    ):
        super(Bagon, self).__init__()

        enc_dec: EncoderDecoderModel
        enc_dec = EncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_model_name, decoder_model_name
        )

        self.encoder = enc_dec.encoder
    
        self.decoder = enc_dec.decoder


    def forward(
        self, 
        encoder_input_ids, encoder_attention_mask,
        decoder_input_ids, decoder_attention_mask
    ):
        
        encoder_output = self.encoder(
            encoder_input_ids, attention_mask=encoder_attention_mask
        ).last_hidden_state

        reconstructed_logits = self.decoder(
            encoder_hidden_states=encoder_output, 
            input_ids=decoder_input_ids, attention_mask=decoder_attention_mask
        ).logits

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

    from transformers import BertTokenizerFast

    from torch import argmax
    from torch.nn.functional import softmax

    encoder_model_name = "bert-base-uncased"
    decoder_model_name = "bert-base-uncased"

    model = Bagon(encoder_model_name, decoder_model_name)

    tokenizer_name = "bert-base-uncased"
    tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(tokenizer_name)

    text = [
        "Today is an amazing day",
        "Tomorrow is the best day",
        "I am totally ready for this great adventure"
    ]
    
    tokenized = tokenizer(text, return_tensors="pt", padding=True)
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask

    recon_logits = model.forward(
        encoder_input_ids=input_ids, encoder_attention_mask=attention_mask,
        decoder_input_ids=input_ids, decoder_attention_mask=attention_mask
    )

    recon_ids = argmax(softmax(recon_logits, dim=-1), dim=-1)

    for i, (a, b, c) in enumerate(zip(input_ids, recon_ids, text)):
        print(f"sentence id: {i}")
        print(f"input_ids: {a}")
        print(f"recon_ids: {b}")

        print(f"text: {c}")
        print(f"text_reconstructed: {' '.join(tokenizer.decode(b))}")
        
        print("\n\n")












if __name__ == "__main__":
    main()