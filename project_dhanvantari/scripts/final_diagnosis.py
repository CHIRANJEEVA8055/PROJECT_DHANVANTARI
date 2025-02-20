import numpy as np
import joblib
import os

# Define paths to models
CNN_MODEL_PATH = "C:/Users/INDUSTRY 4.0/PycharmProjects/project_dhanvantari/models/cnn/cnn_model.h5"
NLP_MODEL_PATH = "C:/Users/INDUSTRY 4.0/PycharmProjects/project_dhanvantari/models/nlp/nlp_model.pkl"
FUSION_MODEL_PATH = "C:/Users/INDUSTRY 4.0/PycharmProjects/project_dhanvantari/models/fusion/fusion_weights.npy"

def load_cnn_model():
    """Loads the trained CNN model if it exists."""
    if os.path.exists(CNN_MODEL_PATH):
        print("CNN model path exists. Load it in your main script using TensorFlow/Keras.")
        return CNN_MODEL_PATH  # Returning path instead of loading directly
    else:
        print("CNN model not found.")
        return None

def load_nlp_model():
    """Loads the trained NLP model if it exists."""
    if os.path.exists(NLP_MODEL_PATH):
        return joblib.load(NLP_MODEL_PATH)
    else:
        print("NLP model not found.")
        return None

def load_fusion_model():
    """Loads the fusion weights if they exist."""
    if os.path.exists(FUSION_MODEL_PATH):
        return np.load(FUSION_MODEL_PATH)
    else:
        print("Fusion weights not found.")
        return None

if __name__ == "__main__":
    cnn_model = load_cnn_model()
    nlp_model = load_nlp_model()
    fusion_weights = load_fusion_model()

    print("CNN Model Path:", cnn_model)
    print("NLP Model:", "Loaded" if nlp_model else "Not Loaded")
    print("Fusion Weights:", fusion_weights if fusion_weights is not None else "Not Loaded")
