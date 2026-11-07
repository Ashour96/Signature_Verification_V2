# Signature Verification V2 – AI-Powered Signature Authentication System

An end-to-end AI-powered signature verification system designed to authenticate handwritten signatures and detect potential forgeries using deep learning and computer vision.

The project leverages a deep learning feature-extraction architecture (M-SigNet) to generate 1024-dimensional signature embeddings, enabling accurate comparison between registered signatures and newly submitted samples.

### Key Features
- **Image Preprocessing:** Standardizes signature images through grayscale conversion, resizing, background normalization, and image enhancement.
- **Test-Time Augmentation (TTA):** Generates multiple variations of input signatures using brightness, contrast, and threshold adjustments to improve robustness against variations in image acquisition conditions.
- **Deep Learning Embeddings:** Extracts 1024-dimensional feature vectors using a pretrained M-SigNet architecture.
- **Signature Similarity:** Compares signature embeddings using distance-based similarity measurement and a predefined verification threshold.
- **Decision Aggregation:** Applies multiple verification predictions and majority voting to produce a final authentication decision.
- **PostgreSQL Integration:** Stores user information, signature images, and their corresponding feature embeddings for future verification.
- **Interactive Interface:** Provides a Streamlit-based application for uploading and verifying signature samples.

### Technology Stack
Python | PyTorch | OpenCV | NumPy | Pandas | Streamlit | PostgreSQL | Deep Learning | Computer Vision

### Project Objective
To develop a reliable and scalable signature authentication solution that reduces manual verification efforts and helps detect forged signatures in document authentication and financial applications.
