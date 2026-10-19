"""Author classifier in PyTorch.

A rebuild of my Java Neural Network. It reads two text files, turns
each one into short chunks, and trains a small network to tell which file a
chunk came from.

Network: 100 inputs -> 10 hidden units (ReLU) -> 1 output (logistic).

Usage:
    python author_classifier.py data/shakespeare.txt data/other_author.txt
"""

import random
import re
import sys
from collections import Counter

import torch
from torch import nn

SEED = 42
VOCAB_SIZE = 100    # number of input features (the most common words)
CHUNK_WORDS = 40    # words per training example
HIDDEN = 10         # hidden units
EPOCHS = 300
LEARNING_RATE = 0.01
TEST_FRACTION = 0.2


def clean(text):
    """Lowercase the text and keep only letters and apostrophes."""
    text = text.lower()
    text = re.sub(r"[^a-z'\s]", " ", text)
    return text.split()


def make_chunks(path, label):
    """Split one file into chunks of CHUNK_WORDS words. Each chunk gets a label."""
    with open(path, encoding="utf-8") as f:
        words = clean(f.read())
    chunks = []
    for start in range(0, len(words) - CHUNK_WORDS + 1, CHUNK_WORDS):
        chunks.append((words[start:start + CHUNK_WORDS], label))
    return chunks


def build_vocab(chunks):
    """Pick the VOCAB_SIZE most common words. Use training data only."""
    counts = Counter(word for words, _ in chunks for word in words)
    return [word for word, _ in counts.most_common(VOCAB_SIZE)]


def vectorize(words, vocab):
    """Turn a chunk into a vector: how often each vocab word appears."""
    counts = Counter(words)
    return [counts[word] / len(words) for word in vocab]


def to_tensors(chunks, vocab):
    x = torch.tensor([vectorize(words, vocab) for words, _ in chunks], dtype=torch.float32)
    y = torch.tensor([[label] for _, label in chunks], dtype=torch.float32)
    return x, y


def accuracy(model, x, y):
    """Share of chunks the model labels correctly."""
    with torch.no_grad():
        predictions = (torch.sigmoid(model(x)) >= 0.5).float()
    return (predictions == y).float().mean().item()


def main():
    if len(sys.argv) != 3:
        print("Usage: python author_classifier.py <author_0.txt> <author_1.txt>")
        sys.exit(1)

    random.seed(SEED)
    torch.manual_seed(SEED)

    # 1. Load the data. Label 0 for the first file, label 1 for the second.
    chunks = make_chunks(sys.argv[1], 0) + make_chunks(sys.argv[2], 1)
    random.shuffle(chunks)

    # 2. Split into training and test sets before building the vocabulary,
    #    so the test set stays unseen.
    split = int(len(chunks) * (1 - TEST_FRACTION))
    train_chunks, test_chunks = chunks[:split], chunks[split:]
    vocab = build_vocab(train_chunks)
    x_train, y_train = to_tensors(train_chunks, vocab)
    x_test, y_test = to_tensors(test_chunks, vocab)

    # 3. Define the network: 100 -> 10 -> 1.
    model = nn.Sequential(
        nn.Linear(VOCAB_SIZE, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, 1),
    )
    # BCEWithLogitsLoss applies the logistic (sigmoid) function inside the loss.
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Train. Each epoch is one forward pass, one loss, one backward pass,
    #    and one weight update on the full training set.
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"epoch {epoch:3d}  loss {loss.item():.4f}")

    # 5. Evaluate on chunks the model has never seen, and compare to a baseline
    #    that always guesses the more common label.
    baseline = max(y_test.mean().item(), 1 - y_test.mean().item())
    print(f"\ntrain accuracy: {accuracy(model, x_train, y_train):.3f}")
    print(f"test accuracy:  {accuracy(model, x_test, y_test):.3f}")
    print(f"baseline (always guess the bigger class): {baseline:.3f}")

    # 6. Save the trained weights, like the binary serialization in the Java version.
    torch.save(model.state_dict(), "author_classifier.pt")
    print("saved weights to author_classifier.pt")


if __name__ == "__main__":
    main()
