import torch

from tqdm import tqdm

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

MODEL_NAME = "Shelgon"

BASE_DIR = f"sentences_latent_reps/{MODEL_NAME}"

sentences_latent_reps = torch.load(f"{BASE_DIR}/sentence_latent_reps.pth")
sentences_latent_reps = torch.tensor(torch.cat(sentences_latent_reps))

sentences_latent_reps_classes = torch.load(f"{BASE_DIR}/sentence_latent_reps_classes.pth")
sentences_latent_reps_classes = torch.tensor(torch.cat(sentences_latent_reps_classes))

classes_to_plot = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 1],
    [0, 0, 1, 1, 0],
    [0, 0, 1, 1, 1],
]

classes_to_labels_for_plot = {
    str(l): i for i, l in enumerate(classes_to_plot)
}

sentences_filtered = []
sentences_filtered_classes = []
sentences_filtered_classes_for_plot = []

for sentence, classes in tqdm(zip(sentences_latent_reps, sentences_latent_reps_classes), total=len(sentences_latent_reps_classes)):
    if classes.tolist() in classes_to_plot:
        sentences_filtered.append(sentence)
        sentences_filtered_classes.append(classes)
        sentences_filtered_classes_for_plot.append(classes_to_labels_for_plot[str(classes.tolist())])

sentences_filtered = torch.stack(sentences_filtered)
sentences_filtered_classes_for_plot = torch.tensor(sentences_filtered_classes_for_plot)

x_tensor = sentences_filtered[:, 0].cpu()
y_tensor = sentences_filtered[:, 1].cpu()
label_tensor = sentences_filtered_classes_for_plot.cpu()

# Min-max scaling for x and y coordinates
x_scaled = (x_tensor - x_tensor.min()) / (x_tensor.max() - x_tensor.min())
y_scaled = (y_tensor - y_tensor.min()) / (y_tensor.max() - y_tensor.min())

# Convert scaled data to pandas DataFrame
data = pd.DataFrame({
    'x': x_scaled.numpy(),
    'y': y_scaled.numpy(),
    'label': label_tensor.numpy()
})

# Plotting
scatter_plot = sns.scatterplot(data=data, x='x', y='y', hue='label', palette='tab10')

# Set x and y axes limits to be between 0 and 1
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title(f"{MODEL_NAME}_1 Latent Space")
plt.xlabel("Latent dimension 1")
plt.ylabel("Latent dimension 2")

plt.savefig(f"{BASE_DIR}/latent_space_plot.png", dpi=1200)