# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

tokenizer = None
model = None


def encode_text(text):
    # Load pre-trained BERT model and tokenizer
    global tokenizer, model
    if not tokenizer:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    if not model:
        model = BertModel.from_pretrained("bert-base-uncased")
    input_encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**input_encoded)
    encoded_input_text = output.last_hidden_state[:, 0, :].squeeze().numpy()
    return encoded_input_text


def cosine_similarity(a, b):
    """
    Compute cosine similarity between a matrix `a` and a vector `b`.

    Parameters:
        a (numpy.ndarray): Matrix of shape (m, n).
        b (numpy.ndarray): Vector of shape (n,).

    Returns:
        numpy.ndarray: Array of cosine similarity scores of shape (m,).
    """
    # Compute dot product between `a` and `b`
    dot_product = np.dot(a, b)

    # Compute magnitudes
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b)

    # Compute cosine similarity
    cosine_sim = dot_product / (a_norm * b_norm)

    return cosine_sim


def top_k_most_similar(test_input, examples, example_embeddings, k):
    example_inputs, example_outputs = zip(*examples)
    # Encode test input
    test_input_embedding = encode_text(test_input)

    # Calculate cosine similarity between test input and each example input
    similarities = cosine_similarity(example_embeddings, test_input_embedding)

    # Find the indices of the top k most similar example inputs
    most_similar_indices = np.argsort(similarities)[-k:]

    # Select the top k most similar example inputs
    top_k_similar_examples_input_output = [
        (example_inputs[i], example_outputs[i]) for i in most_similar_indices
    ]

    return top_k_similar_examples_input_output


class ExampleSelector:
    def __init__(self, candidate_examples):
        self.candidate_examples = candidate_examples
        example_inputs, example_outputs = zip(*self.candidate_examples)
        self.example_embeddings = np.array(
            [
                encode_text(inp)
                for inp in tqdm(example_inputs, desc="Embedding candidates")
            ]
        )

    def select_k_example(self, k, test_input):
        selected_examples = top_k_most_similar(
            self.format_test_input(test_input),
            self.candidate_examples,
            self.example_embeddings,
            k,
        )
        return selected_examples

    def format_test_input(self, test_input):
        return test_input
