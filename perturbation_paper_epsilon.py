import MLDP
import numpy as np
from gensim.models import KeyedVectors
import os
import pickle
import nltk
nltk.download("punkt_tab", quiet=True)
from CollocationExtractor import CollocationExtractor
from datasets import load_dataset

# Save processed dataset
def save_processed(dataset, task_name, split, epsilon):
    save_dir = f"./datasets_MVC_epsilon_{epsilon}"
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{task_name}_{split}.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(dataset, f)


# Load processed dataset
def load_processed(task_name, split, epsilon):
    file_path = f"./datasets_MVC_epsilon_{epsilon}/{task_name}_{split}.pkl"
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    return None


# Get sentences from dataset
def get_split_sentences(dataset, split, task_name):
    if task_name in ["cola", "sst2"]:
        return dataset[split]["sentence"]
    elif task_name in ["mrpc", "rte"]:
        return dataset[split]["sentence1"], dataset[split]["sentence2"]
    else:
        raise ValueError(f"Task {task_name} not supported.")


# Calculate modified epsilon
def calculate_modified_epsilon(collocated_sentences, epsilon, task_name):
    total_words = 0
    total_word_length = 0

    for sentence in collocated_sentences:
        words = sentence.split()
        total_words += len(words)
        total_word_length += sum(len(word) for word in words)

    avg_word_len = total_word_length / total_words if total_words > 0 else 1
    modified_epsilon = epsilon * avg_word_len
    print(f"[{task_name}] Average Word Length after Collocation: {avg_word_len:.2f}, Modified Epsilon: {modified_epsilon:.2f}")
    return modified_epsilon


# Extract collocations and calculate epsilon
def extract_collocations_and_calculate_epsilon(sentences, extractor, epsilon, task_name):
    collocated_sentences = []
    for sentence in sentences:
        tokens, _ = extractor.parse(sentence)
        collocated_sentences.append(" ".join(tokens))

    modified_epsilon = calculate_modified_epsilon(collocated_sentences, epsilon, task_name)
    return collocated_sentences, modified_epsilon


# Perturb sentences with sentence-specific epsilon
def perturb_sentences(sentences, mechanism, epsilon):
    perturbed_sentences = []
    for sentence in sentences:
        words = sentence.split()
        sentence_specific_epsilon = epsilon / len(words) if len(words) > 0 else epsilon  # Avoid division by zero
        perturbed_words = [mechanism.replace_word(word, sentence_specific_epsilon) for word in words]
        perturbed_sentences.append(" ".join(perturbed_words))
    return perturbed_sentences


# Main workflow
epsilon_values = [1]  # List of epsilon values to process
datasets_to_process = [("rte", load_dataset("glue", "rte"))]

# Load vectors and embeddings
np_vectors = np.load('vectors/phrase_max.wordvectors.vectors.npy')
if GST := True:  # Toggle for GST or MST
    gensim_vectors = KeyedVectors.load('vectors/phrase.wordvectors')
else:
    gensim_vectors = KeyedVectors.load('vectors/phrase_max.wordvectors')

extractor = CollocationExtractor()
mechanism = MLDP.MultivariateCalibrated(embedding_matrix=gensim_vectors, use_faiss=True)

for epsilon in epsilon_values:
    print(f"\nProcessing for epsilon = {epsilon}")
    for task_name, dataset in datasets_to_process:
        print(f"\nProcessing Task: {task_name}")
        for split in dataset.keys():
            if split == "test":
                print(f"Skipping {task_name} - {split} split (test set).")
                continue

            print(f"\nProcessing {task_name.upper()} - {split} split")

            if task_name in ["cola", "sst2"]:
                sentences = dataset[split]["sentence"]
            elif task_name in ["mrpc", "rte"]:
                sentences1, sentences2 = dataset[split]["sentence1"], dataset[split]["sentence2"]
                sentences = sentences1 + sentences2

            collocated_sentences, modified_epsilon = extract_collocations_and_calculate_epsilon(sentences, extractor, epsilon, task_name)
            perturbed_sentences = perturb_sentences(collocated_sentences, mechanism, epsilon)

            processed_data = {
                "sentence": perturbed_sentences,
                "collocations": collocated_sentences,
                "label": dataset[split]["label"],
            }
            save_processed(processed_data, task_name, split, epsilon)
            print(f"Saved {task_name} - {split} split for epsilon = {epsilon}.")
