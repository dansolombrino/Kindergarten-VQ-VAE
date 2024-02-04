from rich import print

import torch.nn as nn
import torch
from torch import argmax
from torch.nn.functional import softmax

from transformers import EncoderDecoderModel

from models.shelgon.VectorQuantizer import VectorQuantizer
from models.shelgon.GumbelQuantizer import GumbelQuantizer

from common.consts import *

from common.model_utils import *

from models.bagon.Bagon import Bagon

from typing import Union

from transformers import PreTrainedModel

SUPPORTED_MODEL_MODES = ["full", "dec-head-ft", "enc-head-ft-dec-head-ft", "vq-ft"]


class Shelgon(Bagon):
    def __init__(
        self, encoder_model_name: str, 
        vector_quantizer: Union[VectorQuantizer, GumbelQuantizer],
        decoder_model_name: str,
        from_pretrained_bagon: Union[str, None]
    ):
        # self.encoder and self.decoder are inherited from Bagon!
        super(Shelgon, self).__init__(
            encoder_model_name=encoder_model_name, 
            decoder_model_name=decoder_model_name
        )
        
        self.vector_quantizer: Union[VectorQuantizer, GumbelQuantizer] = vector_quantizer
        
        if from_pretrained_bagon is not None:
            bagon_checkpoint = torch.load(from_pretrained_bagon)

            self.encoder.load_state_dict(bagon_checkpoint["encoder_state_dict"])
            self.decoder.load_state_dict(bagon_checkpoint["decoder_state_dict"])

        self.model_mode = "full"
        

    def forward(self, input_ids: Tensor, attention_mask: Tensor, device, is_training: bool):
        
        embeds = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state

        assert embeds.shape[-1] == self.vector_quantizer.e_dim, "embedding dim of encoder output must match e_dim (for now)!"

        # NOTE ugly AF, TODO uniform inputs and outputs of the two classes to avoid using the if statement
        if type(self.vector_quantizer).__name__ == "VectorQuantizer":
            vq_loss, z_q, perplexity, min_encodings, min_encoding_indices = self.vector_quantizer.forward(embeds, device)
        
        elif type(self.vector_quantizer).__name__ == "GumbelQuantizer":
            z_q, vq_loss, min_encoding_indices = self.vector_quantizer.forward(embeds, is_training)

            # NOTE not the actual perplexity computation, but still informative
            # according to https://stats.stackexchange.com/questions/600948/codebook-perplexity-in-vq-vae
            perplexity = torch.numel(torch.unique(min_encoding_indices.cpu()))
        
        else:
            raise ValueError(f"{type(self.vector_quantizer).__name__} vector quantizer mode NOT supported. Supported modalities: {', '.join(SUPPORTED_VQ_MODES)}")

        # reconstructed_logits = self.decoder(inputs_embeds=z_q).logits
        reconstructed_logits = self.decoder(encoder_hidden_states=z_q, input_ids=input_ids, attention_mask=attention_mask).logits

        return vq_loss, perplexity, min_encoding_indices, reconstructed_logits
    
    def model_params_summary_dict(self):
        return {
            "encoder": {
                "n_trainable_params": n_trainable_params(self.encoder),
                "n_not_trainable_params": n_not_trainable_params(self.encoder),
                "n_params": n_params(self.encoder)
            },
            
            "vector_quantizer": {
                "n_trainable_params": n_trainable_params(self.vector_quantizer),
                "n_not_trainable_params": n_not_trainable_params(self.vector_quantizer),
                "n_params": n_params(self.vector_quantizer)
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
            self.vector_quantizer, "Vector Quantizer", COLOR_TRAIN, COLOR_FROZEN, COLOR_TOT
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
        # Layers composing BERT classification head, in Huggingface implementation:
        # - decoder.cls.predictions.transform.dense 
        # - decoder.cls.predictions.decoder
        self._module_make_trainable(self.decoder.cls.predictions.transform.dense, encoder_lm_head_requires_grad)
        self._module_make_trainable(self.decoder.cls.predictions.decoder        , encoder_lm_head_requires_grad)
    
    def _set_mode_dec_head_ft(self):
        self.model_mode = "dec-head-ft"

        # NOT possible to freeze every param except the ones in the desired layers
        # So, we first freeze the BERT encoder and the BERT decoder
        self._encoder_make_trainable(False)
        self._decoder_make_trainable(False)

        # Vector Quantizer still trainable!
        
        # Then, we unfreeze the parameters of the layers we're interested in
        self._decoder_lm_head_make_trainable(True)

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

    from transformers import BertTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder_model_name = "bert-base-uncased"
    vq_n_e = 10
    vq_e_dim = 768
    vq_beta = 0.69
    decoder_model_name = "bert-base-uncased"

    model = Shelgon(
        encoder_model_name, 
        vq_n_e, vq_e_dim, vq_beta, 
        decoder_model_name,
        from_pretrained_bagon="./runs/Bagon/2024_01_27_18_31_27/bagon_ckpt_loss_recon_val_best.pth"
    ).to(device)

    tokenizer_name = "bert-base-uncased"
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    text = [
        "Today is an amazing day",
        "Tomorrow is the best day",
        "I am totally ready for this great adventure"
    ]
    
    tokenized = tokenizer(text, return_tensors="pt", padding=True, add_special_tokens=False)
    input_ids: Tensor = tokenized.input_ids.to(device)
    attention_mask: Tensor = tokenized.attention_mask.to(device)

    vq_loss, perplexity, min_encoding_indices, recon_ids = model.forward(input_ids, attention_mask, device)

    print(recon_ids, recon_ids.shape)
    print()
    print(vq_loss)












if __name__ == "__main__":
    main()