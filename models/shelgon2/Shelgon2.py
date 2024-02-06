from rich import print

import torch.nn as nn
import torch

from common.consts import *

from common.model_utils import *

from models.bagon.Bagon import Bagon

from models.shelgon2.SentenceDiscretizer import SentenceDiscretizer

SUPPORTED_MODEL_MODES = ["full", "dec-head-ft", "enc-head-ft-dec-head-ft", "vq-ft"]


class Shelgon2(Bagon):
    def __init__(
        self, 
        encoder_model_name: str, 
        sentence_discretizer: SentenceDiscretizer,
        decoder_model_name: str
    ):
        # self.encoder and self.decoder are inherited from Bagon!
        super(Shelgon2, self).__init__(
            encoder_model_name=encoder_model_name, 
            decoder_model_name=decoder_model_name
        )

        self.sentence_discretizer: SentenceDiscretizer = sentence_discretizer

        self.model_mode = "full"
        

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        
        enc_out = self.encoder(input_ids, attention_mask=attention_mask)
        
        embedded_words = enc_out.last_hidden_state
        embedded_sentences = enc_out.pooler_output

        discretized_embedded_sentences, latent_factors_logits, latent_factors_labels = self.sentence_discretizer.forward(embedded_sentences)

        reconstructed_logits = self.decoder(
            encoder_hidden_states=discretized_embedded_sentences, 
            input_ids=input_ids, 
            attention_mask=attention_mask
        ).logits

        return reconstructed_logits, latent_factors_logits, latent_factors_labels
    
    def model_params_summary_dict(self):
        return {
            "encoder": {
                "n_trainable_params": n_trainable_params(self.encoder),
                "n_not_trainable_params": n_not_trainable_params(self.encoder),
                "n_params": n_params(self.encoder)
            },
            
            "sentence_discretizer": {
                "n_trainable_params": n_trainable_params(self.sentence_discretizer),
                "n_not_trainable_params": n_not_trainable_params(self.sentence_discretizer),
                "n_params": n_params(self.sentence_discretizer)
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
            self.sentence_discretizer, "Sentence Discretizer", COLOR_TRAIN, COLOR_FROZEN, COLOR_TOT
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

    sentence_discretizer = SentenceDiscretizer(
        word_embedding_size=768, sentence_length=14
    )

    decoder_model_name = "bert-base-uncased"

    model = Shelgon2(
        encoder_model_name, 
        sentence_discretizer,
        decoder_model_name,
    ).to(device)

    tokenizer_name = "bert-base-uncased"
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    text = [
        "Today is an amazing day",
        "Tomorrow is the best day",
        "I am totally ready for this great adventure"
    ]
    
    tokenized = tokenizer(text, return_tensors="pt", padding="max_length", max_length=14, add_special_tokens=True)
    input_ids: Tensor = tokenized.input_ids.to(device)
    attention_mask: Tensor = tokenized.attention_mask.to(device)

    reconstructed_logits, latent_factors_logits, latent_factors_labels = model.forward(input_ids, attention_mask)

    print(f"reconstructed_logits.shape: {reconstructed_logits.shape}")












if __name__ == "__main__":
    main()