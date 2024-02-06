from torch import nn

from torch import Tensor

from torch.nn.functional import pad
from torch.nn.functional import gumbel_softmax

from models.shelgon2.GenerativeFactorDiscretizer import GenerativeFactorDiscretizer

import torch


class SentenceDiscretizer(nn.Module):
    def __init__(
        self,
        word_emb_size: int,
        sentence_len: int,
        num_latent_gen_factors: int
    ):
        super(SentenceDiscretizer, self).__init__()

        self.num_latent_gen_factors = num_latent_gen_factors

        # singular, plural --> "car", "cars"
        self.gram_num_obj_discretizer = GenerativeFactorDiscretizer(
            word_emb_size=word_emb_size, gen_factor_num_values=3
        )

        # interrogative, affirmative --> sentence is a question, sentence is an affirmation
        self.sentence_type_discretizer = GenerativeFactorDiscretizer(
            word_emb_size=word_emb_size, gen_factor_num_values=3
        )

        self.gender_discretizer = GenerativeFactorDiscretizer(
            word_emb_size=word_emb_size, gen_factor_num_values=3
        )
        
        # singular, plural --> "I", "we"
        self.gram_num_subject_discretizer = GenerativeFactorDiscretizer(
            word_emb_size=word_emb_size, gen_factor_num_values=3
        )

        # 1st, 2nd or 3rd person
        self.gram_num_person_discretizer = GenerativeFactorDiscretizer(
            word_emb_size=word_emb_size, gen_factor_num_values=3
        )
        
        # affirmative, negative
        self.sentence_neg_discretizer = GenerativeFactorDiscretizer(
            word_emb_size=word_emb_size, gen_factor_num_values=3
        )

        # past, present, future
        self.tense_discretizer = GenerativeFactorDiscretizer(
            word_emb_size=word_emb_size, gen_factor_num_values=3
        )
        
        # progressive, not progressie --> "eat", "eating"
        self.style_discretizer = GenerativeFactorDiscretizer(
            word_emb_size=word_emb_size, gen_factor_num_values=3
        )

        self.latent_factors_contract = nn.Conv1d(
            in_channels=self.num_latent_gen_factors, 
            out_channels=sentence_len, kernel_size=1
        )

    def forward(self, embedded_sentences: Tensor):

        gram_num_obj_emb, gram_num_obj_logits, gram_num_obj_label = self.gram_num_obj_discretizer.forward(embedded_sentences)
        sentence_type_emb, sentence_type_logits, sentence_type_label = self.sentence_type_discretizer.forward(embedded_sentences)
        gender_emb, gender_logits, gender_label = self.gender_discretizer.forward(embedded_sentences)
        gram_num_subject_emb, gram_num_subject_logits, gram_num_subject_label = self.gram_num_subject_discretizer.forward(embedded_sentences)
        gram_num_person_emb, gram_num_person_logits, gram_num_person_label = self.gram_num_person_discretizer.forward(embedded_sentences)
        sentence_neg_emb, sentence_neg_logits, sentence_neg_label = self.sentence_neg_discretizer.forward(embedded_sentences)
        tense_emb, tense_logits, tense_label = self.tense_discretizer.forward(embedded_sentences)
        style_emb, style_logits, style_label = self.style_discretizer.forward(embedded_sentences)

        discretized_sentence = torch.stack(
            [
                gram_num_obj_emb,
                sentence_type_emb,
                gender_emb,
                gram_num_subject_emb,
                gram_num_person_emb,
                sentence_neg_emb,
                tense_emb,
                style_emb
            ], 
            dim=1
        )
        # print(f"discretized_sentence.shape: {discretized_sentence.shape}")

        discretized_sentence = self.latent_factors_contract.forward(discretized_sentence)
        # print(f"encoder_hidden_states.shape: {encoder_hidden_states.shape}")

        gen_factors_logits = torch.stack(
            [
                gram_num_obj_logits,
                sentence_type_logits,
                gender_logits,
                gram_num_subject_logits,
                gram_num_person_logits,
                sentence_neg_logits,
                tense_logits,
                style_logits
            ],
            dim=-1
        )

        gen_factors_labels = torch.stack(
            [
                gram_num_obj_label,
                sentence_type_label,
                gender_label,
                gram_num_subject_label,
                gram_num_person_label,
                sentence_neg_label,
                tense_label,
                style_label
            ],
            dim=-1
        )

        return discretized_sentence, gen_factors_logits, gen_factors_labels