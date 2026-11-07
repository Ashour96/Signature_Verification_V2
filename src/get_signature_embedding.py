
import torch
from src.image_transform import load_signature, load_signature_cropped_image


@torch.no_grad()
def get_signature_embedding(model, image_path, device):
    """
    Generate a signature embedding from an image.

    Returns:
        A 1D embedding tensor.
    """
    model.eval()

    image_tensor = load_signature(image_path).to(device)

    output = model(image_tensor)

    # Final concatenated embedding
    embedding = output[0]

    # Remove batch dimension: (1, embedding_dim) -> (embedding_dim,)
    embedding = embedding.squeeze(0)

    return embedding