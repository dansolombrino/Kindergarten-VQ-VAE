import numpy as np

from tqdm import tqdm

import pandas as pd

from collections import Counter

SENTENCES_PATH = "./data/dSentences/dSentences_sentences.npy"
CLASS_LABELS_PATH = "./data/dSentences/dSentences_latent_classes_labels.npy"
CLASS_ONE_HOT_PATH = "./data/dSentences/dSentences_latent_classes_one_hot.npy"

sentences = [sentence.decode() for sentence in np.load(SENTENCES_PATH)]
latent_classes_labels = np.load(CLASS_LABELS_PATH)
latent_classes_one_hot = np.load(CLASS_ONE_HOT_PATH)

sentences_clean = []
sentences_clean_set = set()
latent_classes_labels_clean = []
latent_classes_one_hot_clean = []

for idx, (s, label, one_hot) in tqdm(
    enumerate(zip(sentences, latent_classes_labels, latent_classes_one_hot)), 
    total=len(sentences)
):

    if s not in sentences_clean_set:
        sentences_clean.append(s)
        sentences_clean_set.add(s)

        label = label[[2, 5, 6, 7, 8]]
        latent_classes_labels_clean.append(label)

        one_hot = np.concatenate((np.asarray([[-1, -1, -1]]), one_hot), axis=0)
        one_hot = one_hot[[2, 5, 6, 7, 8], :]

        latent_classes_one_hot_clean.append(one_hot)

np.save(
    SENTENCES_PATH.replace(".npy", "_clean.npy"),
    np.asarray(sentences_clean)
)

np.save(
    CLASS_LABELS_PATH.replace(".npy", "_clean.npy"),
    np.asarray(latent_classes_labels_clean)
)

np.save(
    CLASS_ONE_HOT_PATH.replace(".npy", "_clean.npy"),
    np.asarray(latent_classes_one_hot_clean)
)

print(len(sentences_clean))
print(len(sentences_clean_set))
print(len(latent_classes_labels_clean))
print(len(latent_classes_one_hot_clean))


