import matplotlib.pyplot as plt
import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image, ImageEnhance
import cv2
import io
import torch
import numpy as np
from torchvision import transforms

import sys
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add the project root to Python's module search path
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_load import get_model, device
from src.get_signature_embedding import get_signature_embedding
from src.get_signature_embedding_cropped_image import get_signature_embedding_cropped_image
from src.squared_euclidean_distance import squared_euclidean_distance
from src.save_image import save_signature
from src.verify_signature import verify_signature
from src.database import get_client_signature
from src.get_final_signature_decision import get_final_signature_decision
from collections import Counter

IMAGE_HEIGHT = 150
IMAGE_WIDTH = 220


st.set_page_config(
    page_title="Image Cropper",
    page_icon="✂️",
    layout="wide"
)

st.title("✂️ Image Cropper")
st.write("Upload an image and select the area you want to crop.")


# --------------------------------------------------
# 1. Upload Image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)


if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(uploaded_file)

    # Convert to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    st.subheader("Select Crop Area")

    # --------------------------------------------------
    # 3. Crop Image and Display Results
    # --------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        # st.subheader("Original Image")
        # st.image(
        #     image,
        #     use_container_width=True
        # )
        cropped_image = st_cropper(
                image,
                realtime_update=True,
                box_color="#00FF00",
                aspect_ratio=None,
                return_type="image"
            )

    with col2:
        st.subheader("Cropped Image")
        st.image(
            cropped_image,
            use_container_width=True
        )

    # --------------------------------------------------
    # 4. Convert Cropped Image to Bytes
    # --------------------------------------------------

    image_bytes = io.BytesIO()

    cropped_image.save(
        image_bytes,
        format="PNG"
    )

    image_bytes.seek(0)

    # --------------------------------------------------
    # 5. Download Button
    # --------------------------------------------------
    


    st.download_button(
        label="⬇️ Download Cropped Image",
        data=image_bytes,
        file_name="cropped_image.png",
        mime="image/png"
    )


model = get_model()

if st.button("Covert to embedding"):
    emb = get_signature_embedding_cropped_image(model, cropped_image, device)
    st.write(emb)
    st.success("embedding Generated Successfully")
    


# #####################################
# # Select Mode 2 use the model
# #####################################
with st.container(border=True):
    option = st.segmented_control(
        "Image Type",
        ["Get Signature From Database", "Upload Image"],
        default=None,
        label_visibility="collapsed",
        width="stretch"
    )


# #####################################
# # Option 1 convert a sig. in your PC
# #####################################
    if option == None:
        st.subheader("Welcome 👋🏻")
        st.info("Select your preferred method for signature authentication.")
    
    elif option == "Upload Image":
        st.subheader("Upload an image 💻")
        # 1- upload the signature
        orignal_signature = st.file_uploader( "Upload Signature Image", type=["jpg", "jpeg", "png", "bmp", "webp"])
        image_path = save_signature(orignal_signature)
        if orignal_signature is not None:
            if st.button("Verify The signatures"):
                # ##############################################
                # # Convert The Images to embedding
                # ##############################################
                crop_emb_dict = get_signature_embedding_cropped_image(model, cropped_image, device)
                org_emb = get_signature_embedding(model, image_path, device)
                # ##############################################
                # # show the images
                # ##############################################
                col1, col2 = st.columns(2)
                with col1:
                     st.subheader("Original Signature")
                     st.image(
                         orignal_signature,
                         use_container_width=True)
                with col2:
                    st.subheader("Cropped Signature")
                    st.image(
                        cropped_image,
                        use_container_width=True)
                # ##############################################
                # # make Decision
                # ##############################################
                results = []
                for img_name, crop_emb in crop_emb_dict.items():
                    result = verify_signature(model, crop_emb, org_emb, device)
                    result['Name'] = img_name
                    results.append(result)
                
                
                # most_frequent = Counter(decision).most_common(1)[0][0]
                final_result = get_final_signature_decision(results)
                st.write(final_result)
                # st.write(np.mean(distance))
                
        else:
            st.error("Falied to Upload Signature (Please Try again!)")
    


    ###############################################
    # # Updated By Eslam
    ###############################################
    elif(option == "Get Signature From Database"):
        st.subheader("Get Signature From Database 🏛️")
        # ##############################################
        # # 1- Enter The id of the customer
        # ##############################################
        id  = st.number_input("Enter Customer id")
        # ##############################################
        # # 2- Check and Retrive the signature if found
        # ##############################################
        image_path = get_client_signature(int(id))
        st.write(image_path.values[0])
        image_path = image_path.values[0]
        # ##############################################
        # # 3- show the signatures
        # ##############################################


        # ##############################################
        # # 4- convert 2 embedding
        # ##############################################
        crop_emb = get_signature_embedding_cropped_image(model, cropped_image, device)
        org_emb = get_signature_embedding(model, image_path, device)

        # ##############################################
        # # 5- show decision
        # ##############################################
        result = verify_signature(model, crop_emb, org_emb, device)
        st.write(result)