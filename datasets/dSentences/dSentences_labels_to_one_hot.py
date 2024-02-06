from rich import print

import numpy as np

import torch

from torch.nn.functional import one_hot

CORPUS_PATH = "./data/dSentences/dSentences_latent_classes_labels.npy" 

latent_classes_labels = np.load(CORPUS_PATH)

latent_classes_labels = latent_classes_labels[:, 1:] # excluding verb-obj intreraction latent factor

print(latent_classes_labels.shape)

latent_classes_labels = torch.as_tensor(latent_classes_labels)

LATENT_GEN_FACTOR_MAX_SUPPORT_SIZE = 3

latent_classes_one_hot = one_hot(latent_classes_labels, LATENT_GEN_FACTOR_MAX_SUPPORT_SIZE)

latent_classes_one_hot = latent_classes_one_hot.numpy()

print(latent_classes_one_hot.shape)

np.save("./data/dSentences/dSentences_latent_classes_one_hot.npy", latent_classes_one_hot)





