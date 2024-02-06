from torch import Tensor, IntTensor
from torch.utils.data import Dataset

import numpy as np

from typing import Union

from torch.nn.functional import one_hot

from torch import as_tensor


class dSentencesDataset(Dataset): 
    def __init__(
        self, 
        sentences_path: str, 
        latent_classes_labels_path: str = None,
        latent_classes_one_hot_path: str = None,
    ):
        self.sentences = [sentence.decode() for sentence in np.load(sentences_path)]

        self.latent_classes_labels = None
        self.latent_classes_one_hot = None
        
        if latent_classes_labels_path is not None and latent_classes_one_hot_path is not None:
        
            self.latent_classes_labels: Tensor = as_tensor(np.load(latent_classes_labels_path)).long()
            self.latent_classes_one_hot: Tensor = as_tensor(np.load(latent_classes_one_hot_path))

            if len(self.sentences) != self.latent_classes_labels.shape[0]:
                raise AssertionError(
                    f"Provided {len(self.sentences)} sentences but only {self.latent_classes_labels.shape[0]} latent classes labels.\n" + \
                    f"Please provide latent classes labels for each sentence!"
                )
            
            if len(self.sentences) != self.latent_classes_one_hot.shape[0]:
                raise AssertionError(
                    f"Provided {len(self.sentences)} sentences but only {self.latent_classes_one_hot.shape[0]} latent classes one-hot labels.\n" + \
                    f"Please provide latent classes one-hot labels for each sentence!"
                )
            
            if self.latent_classes_labels.shape[0] != self.latent_classes_one_hot.shape[0]:
                raise AssertionError(
                    f"Number of latent classes labels ({self.latent_classes_labels.shape[0]}) should match number of latent classes one-hot labels ({self.latent_classes_one_hot.shape[0]})"
                )

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx) -> Union[str, dict]:

        if self.latent_classes_labels is None:
            return self.sentences[idx]
        else:

            return {
                "sentence": self.sentences[idx],
                "latent_classes_labels": self.latent_classes_labels[idx][1:], # excluding obj-verb relation latent generative factor
                "latent_classes_one_hot": self.latent_classes_one_hot[idx]
            }
    


if __name__ == "__main__":

    from rich import print

    dataset = dSentencesDataset(
        "./data/dSentences/dSentences_sentences.npy",
        "./data/dSentences/dSentences_latent_classes_labels.npy"
    )

    print(dataset[0])
    print(dataset[-1])