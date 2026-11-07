import torch
from src.image_transform import load_signature_cropped_image






preprocessing_configs = [
    # 1. Original baseline
    {
        "name": "original",
        "brightness": 1.00,
        "contrast": 1.00,
        "threshold_offset": 30
    },

    # 2. Slightly brighter
    {
        "name": "bright",
        "brightness": 1.05,
        "contrast": 1.00,
        "threshold_offset": 30
    },

    # 3. Slightly darker
    {
        "name": "dark",
        "brightness": 0.95,
        "contrast": 1.00,
        "threshold_offset": 30
    },

    # 4. Higher contrast
    {
        "name": "high_contrast",
        "brightness": 1.00,
        "contrast": 1.15,
        "threshold_offset": 30
    },

    # 5. Lower contrast
    {
        "name": "low_contrast",
        "brightness": 1.00,
        "contrast": 0.85,
        "threshold_offset": 30
    },

    # 6. Stronger threshold
    {
        "name": "strong_threshold",
        "brightness": 1.00,
        "contrast": 1.00,
        "threshold_offset": 45
    },

    # 7. Weaker threshold
    {
        "name": "weak_threshold",
        "brightness": 1.00,
        "contrast": 1.00,
        "threshold_offset": 15
    },

    # 8. Bright background adjustment
    {
        "name": "bright_background",
        "brightness": 1.05,
        "contrast": 1.10,
        "threshold_offset": 40
    },

    # 9. Faded signature adjustment
    {
        "name": "faded_signature",
        "brightness": 0.95,
        "contrast": 1.15,
        "threshold_offset": 35
    },

    # 10. Dark signature adjustment
    {
        "name": "dark_signature",
        "brightness": 1.05,
        "contrast": 0.90,
        "threshold_offset": 20
    }
]

@torch.no_grad()
def get_signature_embedding_cropped_image(model,
                                           image,
                                            device,
                                            preprocessing_configs = preprocessing_configs):
    """
    Generate a signature embedding from a cropped PIL image.

    Args:
        model: Loaded signature verification model.
        image: Cropped PIL image.
        device: CPU or CUDA device.

    Returns:
        A 1D embedding tensor.
    """
    model.eval()
    embeddings = {}

    for config in preprocessing_configs:

        tensor = load_signature_cropped_image(
            image,
            brightness=config["brightness"],
            contrast=config["contrast"],
            threshold_offset=config["threshold_offset"]
        ).to(device)

        output = model(tensor)

        embedding = output[0]
        embedding = embedding.squeeze(0)
        embeddings[config["name"]] = embedding
    
    
    
    
    
    
    # Preprocess the cropped image
    # image_tensor = load_signature_cropped_image(image,
    #                                             brightness=1.05,
    #                                             contrast=1.0,
    #                                             threshold_offset=30).to(device)

    # Model inference
    # output = model(image_tensor)

    # Final concatenated embedding
    # embedding = output[0]

    # Remove batch dimension
    # embedding = embedding.squeeze(0)

    return embeddings
