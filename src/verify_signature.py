import torch
from src.get_signature_embedding import get_signature_embedding
from src.squared_euclidean_distance import squared_euclidean_distance
from src.get_signature_embedding_cropped_image import get_signature_embedding_cropped_image

# threshold_path = r"F:\03 - Projects\8_Signature_Verification_V2\model\threshold.pt"
threshold_path = r"F:\03 - Projects\8_Signature_Verification\Models\V3\NBE_Output\NBE_MS_SigNet_Project\checkpoints\threshold.pt"


threshold_data = torch.load(
    threshold_path,
    map_location="cpu",
    weights_only=False
)



@torch.no_grad()
def verify_signature(
    model,
    reference_embedding,
    questioned_embedding,
    device,
    threshold= threshold_data['distance_threshold']
):
    """
    Compare two signatures using NBE-MS-SigNet embeddings.

    Returns:
        Dictionary containing the distance, threshold, and decision.
    """

    model.eval()

    # Ensure embeddings have compatible shapes
    if reference_embedding.shape != questioned_embedding.shape:
        raise ValueError(
            f"Embedding shape mismatch: "
            f"{reference_embedding.shape} vs "
            f"{questioned_embedding.shape}"
        )

    # Calculate distance
    distance = squared_euclidean_distance(
        reference_embedding,
        questioned_embedding
    )

    # Apply the trained model's verification threshold
    decision = (
        "GENUINE"
        if distance <= threshold
        else "FORGERY"
    )

    return {
        "decision": decision,
        "distance": distance,
        "threshold": threshold
    }