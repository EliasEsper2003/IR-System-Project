import pickle
import numpy as np


def save_pickle(obj, path):

    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):

    with open(path, "rb") as f:
        return pickle.load(f)


def save_embeddings(embeddings, path):

    np.save(path, embeddings)


def load_embeddings(path):

    return np.load(path)

