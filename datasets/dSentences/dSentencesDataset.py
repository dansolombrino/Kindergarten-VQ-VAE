from torch import Tensor
from torch.utils.data import Dataset

import numpy as np


class dSentencesDataset(Dataset):
    def __init__(self, corpus_path: str):
        self.corpus = [sentence.decode() for sentence in np.load(corpus_path)]

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        return self.corpus[idx]
    


if __name__ == "__main__":

    from rich import print

    dataset = dSentencesDataset("./data/dSentences/dSentences_sentences.npy")

    print(dataset[0])
    print(dataset[-1])