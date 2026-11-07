from pathlib import Path
from datetime import datetime
import numpy as np


id = np.random.randint(1,100)
default_upload_path = Path(r"F:\03 - Projects\8_Signature_Verification_V2\database\uploaded")


def save_signature(uploaded_file, customer_id = id, save_dir=default_upload_path):
    if uploaded_file is None:
        return None

    # Create the directory if it doesn't exist
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Generate a unique filename: CustomerID_Date_Time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{customer_id}_{timestamp}{Path(uploaded_file.name).suffix.lower()}"

    # Save the uploaded image
    image_path = save_dir / filename
    image_path.write_bytes(uploaded_file.getvalue())

    # Return the full path
    return str(image_path.resolve())