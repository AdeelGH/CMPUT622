import MLDP
import numpy as np
from gensim.models import KeyedVectors
import os
import pickle

from CLMLDP.CollocationExtractor import CollocationExtractor
from datasets import load_dataset


def save_processed(dataset, task_name, split):
    save_dir = "./processed_datasets"
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{task_name}_{split}.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(dataset, f)


def load_processed(task_name, split):
    file_path = f"./processed_datasets/{task_name}_{split}.pkl"
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    return None


def get_split_sentences(dataset, split, task_name):
    if task_name in ["cola", "sst2"]:
        return dataset[split]["sentence"]
    elif task_name in ["mrpc", "rte"]:
        return dataset[split]["sentence1"], dataset[split]["sentence2"]
    else:
        raise ValueError(f"Task {task_name} not supported.")


def perturb_sentences(sentences, mechanism):
    perturbed_sentences = []
    for sentence in sentences:
        words = sentence.split()
        perturbed_words = [mechanism.replace_word(word, 1) for word in words]
        perturbed_sentences.append(" ".join(perturbed_words))
    return perturbed_sentences


def extract_collocations(sentences, extractor):
    collocations = []
    for sentence in sentences:
        collocations.append(extractor.parse(sentence))
    return collocations


# Load specific GLUE tasks
cola = load_dataset("glue", "cola")
mrpc = load_dataset("glue", "mrpc")
rte = load_dataset("glue", "rte")
sst2 = load_dataset("glue", "sst2")
print(cola)

# SST2 Random Sample
sst2_sample = sst2["train"].shuffle(seed=42).select(range(10000))

np_vectors = np.load('/Users/gajia/PycharmProjects/CMPUT622/phrase_max.wordvectors.vectors.npy')
orig_sentence = 'I do not know what to do but I like to dance'

GST = True
# GST Collocation Generation
if GST:
    gensim_vectors = KeyedVectors.load('/Users/gajia/PycharmProjects/CMPUT622/phrase.wordvectors')
    extractor = CollocationExtractor()
    parsed_text = extractor.parse(orig_sentence)
    print(parsed_text)
# MST Collocation Generation
else:
    gensim_vectors = KeyedVectors.load('/Users/gajia/PycharmProjects/CMPUT622/phrase_max.wordvectors')
    extractor = CollocationExtractor()
    parsed_text = extractor.parse_max(orig_sentence)
    print(parsed_text)

mechanism = MLDP.MultivariateCalibrated(embedding_matrix=gensim_vectors, use_faiss=True)
orig_word = ['I', 'do', 'not', 'know', 'what', 'to', 'do', 'but', 'I', 'like', 'to', 'dance']
perturbed_word = []
for word in orig_word:
    perturbed_word.append(mechanism.replace_word(word, 1))
print(perturbed_word)


perturbed_datasets = {}
for task_name, dataset in [("cola", cola), ("mrpc", mrpc), ("rte", rte), ("sst2", sst2)]:
    perturbed_datasets[task_name] = {}
    # Iterate over train, validation, test splits
    for split in dataset.keys():
        print(f"Processing {task_name} - {split} split...")

        if split == "test":
            perturbed_datasets[task_name][split] = dataset[split]
            continue

        if task_name in ["mrpc", "rte"]:
            sentences1, sentences2 = get_split_sentences(dataset, split, task_name)

            # Extract collocations for both sentences
            collocations1 = extract_collocations(sentences1, extractor)
            collocations2 = extract_collocations(sentences2, extractor)

            # Perturb only the first sentence
            perturbed1 = perturb_sentences(sentences1, mechanism)

            # Save both sets of perturbed sentences and collocations
            perturbed_datasets[task_name][split] = {
                "perturbed_sentence1": perturbed1,
                "collocations1": collocations1,
                "sentence2": sentences2,
                "collocations2": collocations2,
                "labels": dataset[split]["label"],
            }
        else:
            sentences = get_split_sentences(dataset, split, task_name)
            collocations = extract_collocations(sentences, extractor)
            perturbed = perturb_sentences(sentences, mechanism)

            # Save perturbed sentences and collocations
            perturbed_datasets[task_name][split] = {
                "sentence": perturbed,
                "collocations": collocations,
                "labels": dataset[split]["label"],
            }
        save_processed(perturbed_datasets[task_name][split], task_name, split)
        print(f"Saved {task_name} - {split} split.")
