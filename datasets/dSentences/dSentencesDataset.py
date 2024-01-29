from torch import Tensor
from torch.utils.data import Dataset

import numpy as np

from typing import Union


class dSentencesDataset(Dataset): 
    def __init__(self, sentences_path: str, latent_classes_path: str = None):
        self.sentences = [sentence.decode() for sentence in np.load(sentences_path)]

        self.latent_classes = None
        
        if latent_classes_path is not None:
        
            self.latent_classes: np.ndarray = np.load(latent_classes_path)

            if len(self.sentences) != self.latent_classes.shape[0]:
                raise AssertionError(
                    f"Provided {len(self.sentences)} sentences but only {self.latent_classes.shape[0]} latent classes labels.\n" + \
                    f"Please provide latent classes labels for each sentence!"
                )

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx) -> Union[str, dict]:

        if self.latent_classes is None:
            return self.sentences[idx]
        else:
            return {
                "sentence": self.sentences[idx],
                "latent_classes_labels": self.latent_classes[idx],
            }
    


if __name__ == "__main__":

    from rich import print

    dataset = dSentencesDataset(
        "./data/dSentences/dSentences_sentences.npy",
        "./data/dSentences/dSentences_latent_classes_labels.npy"
    )

    print(dataset[0])
    print(dataset[-1])