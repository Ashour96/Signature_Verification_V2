import torch


def squared_euclidean_distance(embedding1, embedding2):
    """
    Calculate squared Euclidean distance between two embeddings.
    """
    return torch.sum((embedding1 - embedding2) ** 2).item()