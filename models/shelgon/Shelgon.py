from rich import print

import torch.nn as nn
import torch

from transformers import EncoderDecoderModel

from common.consts import *

from common.model_utils import *

SUPPORTED_MODEL_MODES = ["full", "dec-head-ft", "enc-head-ft-dec-head-ft"]


class Shelgon(nn.Module):
    def __init__(
        self, 
        encoder_model_name: str, 
        emb_size: int, seq_len: int,
        num_latent_classes: int, num_labels_per_class: int, 
        decoder_model_name: str
    ):
        super(Shelgon, self).__init__()

        enc_dec: EncoderDecoderModel
        enc_dec = EncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_model_name, decoder_model_name
        )

        self.encoder = enc_dec.encoder
    
        self.decoder = enc_dec.decoder

        self.encoder_model_name = encoder_model_name

        self.decoder_model_name = decoder_model_name

        self.proj_in = nn.Sequential(
            nn.Linear(in_features=emb_size, out_features=num_labels_per_class),
            nn.Conv1d(in_channels=seq_len, out_channels=num_latent_classes, kernel_size=1)
        )

        self.proj_out = nn.Sequential(
            nn.Conv1d(in_channels=num_latent_classes, out_channels=seq_len, kernel_size=1),
            nn.Linear(in_features=num_labels_per_class, out_features=emb_size)
        )


    def forward(
        self, 
        encoder_input_ids, encoder_attention_mask,
        decoder_input_ids, decoder_attention_mask
    ):
        
        encoder_output = self.encoder(
            encoder_input_ids, attention_mask=encoder_attention_mask
        ).last_hidden_state

        pred_latent_classes = self.proj_in(encoder_output)

        decoder_input_conditioning = self.proj_out(pred_latent_classes)

        reconstructed_logits = self.decoder(
            encoder_hidden_states=decoder_input_conditioning, 
            input_ids=decoder_input_ids, attention_mask=decoder_attention_mask
        ).logits

        return reconstructed_logits, pred_latent_classes
    

    def model_params_summary_dict(self):
        return {
            "encoder": {
                "n_trainable_params": n_trainable_params(self.encoder),
                "n_not_trainable_params": n_not_trainable_params(self.encoder),
                "n_params": n_params(self.encoder)
            },
            
            "enc to latent labels": {
                "n_trainable_params": n_trainable_params(self.proj_in),
                "n_not_trainable_params": n_not_trainable_params(self.proj_in),
                "n_params": n_params(self.proj_in)
            },
            
            "latent labels to dec": {
                "n_trainable_params": n_trainable_params(self.proj_out),
                "n_not_trainable_params": n_not_trainable_params(self.proj_out),
                "n_params": n_params(self.proj_out)
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
            self.proj_in, "Encoder to latent labels", COLOR_TRAIN, COLOR_FROZEN, COLOR_TOT
        )
        
        print_module_params_summary(
            self.proj_out, "Latent labels to decoder", COLOR_TRAIN, COLOR_FROZEN, COLOR_TOT
        )
        
        print_module_params_summary(
            self.decoder, "Decoder", COLOR_TRAIN, COLOR_FROZEN, COLOR_TOT
        )
        
        return
    

    def _module_make_trainable(self, module, module_requires_grad: bool):
        for param in module.parameters():
            param.requires_grad = module_requires_grad
    
    
    def _encoder_make_trainable(self, encoder_requires_grad: bool):
        self._module_make_trainable(self.encoder, encoder_requires_grad)

    
    def _decoder_make_trainable(self, decoder_requires_grad: bool):
        self._module_make_trainable(self.decoder, decoder_requires_grad)

    
    def _decoder_lm_head_make_trainable(self, encoder_lm_head_requires_grad: bool):

        if "bert" in self.decoder_model_name:
            # Layers composing BERT classification head, in Huggingface implementation:
            # - decoder.cls.predictions.transform.dense 
            # - decoder.cls.predictions.decoder
            self._module_make_trainable(self.decoder.cls.predictions.transform.dense, encoder_lm_head_requires_grad)
            self._module_make_trainable(self.decoder.cls.predictions.decoder        , encoder_lm_head_requires_grad)

        elif "gpt" in self.decoder_model_name:
            self._module_make_trainable(self.decoder.lm_head, encoder_lm_head_requires_grad)
    
    
    def _decoder_cross_attn_make_trainable(self, decoder_cross_attn_requires_grad: bool):

        if "bert" in self.decoder_model_name:
            for layer in self.decoder.bert.encoder.layer:

                self._module_make_trainable(layer.crossattention, decoder_cross_attn_requires_grad)

    def _set_mode_dec_head_ft(self):
        self.model_mode = "dec-head-ft"

        # NOT possible to freeze every param except the ones in the desired layers
        # So, we first freeze the BERT encoder and the BERT decoder
        self._encoder_make_trainable(False)
        self._decoder_make_trainable(False)

        # Vector Quantizer still trainable!
        
        # Then, we unfreeze the parameters of the layers we're interested in
        self._decoder_lm_head_make_trainable(True)

        self._decoder_cross_attn_make_trainable(True)

    
    def _set_mode_enc_head_dec_head_ft(self):
        self.model_mode = "enc-dec-head-ft"

        # First we set model to fine-tune just the decoder head
        self._set_mode_dec_head_ft()
        # Then we make the encoder last layer trainable as well
        self._module_make_trainable(self.encoder.encoder.layer[-1], True)
        self._module_make_trainable(self.encoder.pooler, True)

    
    def set_mode(self, model_mode: str):

        if model_mode == "full":
            self.model_mode = "full"
            return
        
        if model_mode == "dec-head-ft":
            self._set_mode_dec_head_ft()

            return

        if model_mode == "enc-head-ft-dec-head-ft":
            self._set_mode_enc_head_dec_head_ft()

            return
        
        if model_mode == "vq-ft":
            self.model_mode = "vq-ft"
            # we want to just fine-tune the vector quantization NN --> gotta freeze encoder and decoder
            for param in self.encoder.parameters():
                param.requires_grad = False
            
            for param in self.decoder.parameters():
                param.requires_grad = False

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
    emb_size = 768
    seq_len = 14
    num_latent_classes = 5
    num_labels_per_class = 3
    decoder_model_name = "bert-base-uncased"

    model = Shelgon(
        encoder_model_name, 
        emb_size, seq_len,
        num_latent_classes, num_labels_per_class,
        decoder_model_name
    )

    tokenizer_name = "bert-base-uncased"
    tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(tokenizer_name)

    text = [
        "Today is an amazing day",
        "Tomorrow is the best day",
        "I am totally ready for this great adventure"
    ]
    
    tokenized = tokenizer(
        text, return_tensors="pt", padding="max_length", max_length=seq_len
    )
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask

    recon_logits, pred_labels = model.forward(
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