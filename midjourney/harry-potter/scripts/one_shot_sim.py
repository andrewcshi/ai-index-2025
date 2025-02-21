import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# For pixel-based similarity (SSIM)
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# Keras application modules for CNN models
from tensorflow.keras.applications import (
    vgg16,
    resnet50,
    inception_v3,
    xception
)

##############################################################################
# 1. Configuration of multiple "models" (including pixel-based)
##############################################################################
MODELS_TO_TRY = {
    # Pixel-based approach (SSIM)
    "PixelBased": {
        "type": "pixel",
        "target_size": (224, 224)  # Make sure it's at least 7x7 for SSIM
    },
    # CNN approaches
    "VGG16": {
        "type": "cnn",
        "model_fn": vgg16.VGG16,
        "preprocess_fn": vgg16.preprocess_input,
        "feature_layer": "fc2",  # penultimate dense layer
        "target_size": (224, 224)
    },
    "ResNet50": {
        "type": "cnn",
        "model_fn": resnet50.ResNet50,
        "preprocess_fn": resnet50.preprocess_input,
        "feature_layer": "avg_pool",  # global average pooling layer
        "target_size": (224, 224)
    },
    "InceptionV3": {
        "type": "cnn",
        "model_fn": inception_v3.InceptionV3,
        "preprocess_fn": inception_v3.preprocess_input,
        "feature_layer": "avg_pool",
        "target_size": (299, 299)
    },
    "Xception": {
        "type": "cnn",
        "model_fn": xception.Xception,
        "preprocess_fn": xception.preprocess_input,
        "feature_layer": "avg_pool",
        "target_size": (299, 299)
    }
}

##############################################################################
# 2. Pixel-based similarity
##############################################################################
def pixel_based_similarity_score(img_path_1, img_path_2, target_size=(224, 224)):
    """
    Computes a pixel-based similarity using SSIM (Structural Similarity Index).
    1) Load both images in RGB.
    2) Resize them to `target_size` so they're the same shape.
    3) Compute SSIM on the [H, W, 3] arrays in [0..1].
    4) Return a value typically in [0..1], where 1 => identical.
    """
    img1 = Image.open(img_path_1).convert("RGB")
    img2 = Image.open(img_path_2).convert("RGB")

    # Resize using Pillow's Resampling
    img1 = img1.resize(target_size, Image.Resampling.LANCZOS)
    img2 = img2.resize(target_size, Image.Resampling.LANCZOS)

    # Convert to NumPy float arrays [H, W, 3] in [0..1]
    arr1 = np.array(img1, dtype=np.float32) / 255.0
    arr2 = np.array(img2, dtype=np.float32) / 255.0

    # Use channel_axis=-1 for color images in recent scikit-image
    ssim_val = ssim(arr1, arr2, data_range=1.0, channel_axis=-1, win_size=7)

    # Ensure the value is >= 0 in case of small floating underflow
    return max(ssim_val, 0.0)

##############################################################################
# 3. CNN-based similarity
##############################################################################
def load_and_preprocess_image_cnn(img_path, preprocess_fn, target_size=(224, 224)):
    """
    Loads an image, resizes it, and applies a CNN's preprocess function.
    """
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_fn(x)  # e.g., vgg16.preprocess_input
    return x

def get_cnn_feature_vector(model, img_path, preprocess_fn, target_size):
    """
    Given a model, an image path, a preprocessing function, 
    and a target size, returns a flattened feature vector for the image.
    """
    x = load_and_preprocess_image_cnn(img_path, preprocess_fn, target_size)
    features = model.predict(x)
    return features.flatten()

def cosine_similarity(vec1, vec2):
    """
    Computes the cosine similarity between two vectors in [-1, 1].
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot_product / (norm1 * norm2)

def cnn_similarity_score(model, img_path_1, img_path_2, preprocess_fn, target_size):
    """
    Computes a CNN-based similarity score (0..1) using cosine similarity
    of the extracted feature vectors.
    """
    fv1 = get_cnn_feature_vector(model, img_path_1, preprocess_fn, target_size)
    fv2 = get_cnn_feature_vector(model, img_path_2, preprocess_fn, target_size)
    sim = cosine_similarity(fv1, fv2)
    return (sim + 1) / 2  # map [-1..1] to [0..1]

##############################################################################
# 4. Main Script (One-Shot)
##############################################################################
if __name__ == "__main__":
    # 1) Path to the reference image
    ref_path = os.path.join("midjourney", "harry-potter", "one-shot", "ref-one-shot.png")
    
    # 2) Define subfolders to search
    version_folders = ["v1", "v2", "v3", "v4", "v5", "v5.1", "v5.2", "v6", "v6.1"]
    parent_folder = os.path.join("midjourney", "harry-potter", "one-shot")
    
    # 3) We'll store results in a list of (Method, Version, ImageName, Similarity)
    all_results = []
    
    # 4) Loop over each "model" entry in MODELS_TO_TRY
    for method_name, config in MODELS_TO_TRY.items():
        print(f"\n=== Processing with {method_name} ===")
        
        method_type = config["type"]
        target_size = config["target_size"]
        
        # If it's a CNN, build the feature-extractor model
        if method_type == "cnn":
            model_constructor = config["model_fn"]
            preprocess_fn = config["preprocess_fn"]
            layer_name = config["feature_layer"]
            
            # Build the base model (with include_top=True, then extract the desired layer)
            base_keras_model = model_constructor(weights="imagenet", include_top=True)
            feature_extractor = Model(
                inputs=base_keras_model.input,
                outputs=base_keras_model.get_layer(layer_name).output
            )
        elif method_type == "pixel":
            # Nothing to build for pixel-based
            pass
        else:
            raise ValueError(f"Unknown method type: {method_type}")
        
        # 5) Iterate through each version folder
        for vfolder in version_folders:
            folder_path = os.path.join(parent_folder, vfolder)
            
            if not os.path.isdir(folder_path):
                print(f"Folder not found: {folder_path}")
                continue
            
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(".png"):
                    img_path = os.path.join(folder_path, filename)
                    
                    # 6) Compute similarity with the one-shot reference
                    if method_type == "cnn":
                        preprocess_fn = config["preprocess_fn"]
                        score = cnn_similarity_score(
                            feature_extractor,
                            ref_path,
                            img_path,
                            preprocess_fn,
                            target_size
                        )
                    elif method_type == "pixel":
                        score = pixel_based_similarity_score(
                            ref_path,
                            img_path,
                            target_size=target_size
                        )
                    
                    # 7) Store result
                    all_results.append((method_name, vfolder, filename, score))
    
    # 8) Convert all results to a DataFrame
    df = pd.DataFrame(all_results, columns=["Method", "Version", "Image", "Similarity"])
    
    # 9) For each (Method, Version), compute the average similarity if multiple images exist
    df_summary = df.groupby(["Method", "Version"], as_index=False)["Similarity"].mean()
    
    # 10) Plot: x=Version, y=Similarity, hue=Method => grouped bar plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))

    ax = sns.barplot(
        data=df_summary,
        x="Version",
        y="Similarity",
        hue="Method",
        palette="Blues_d"
    )

    # Optional: add text labels on top of each bar
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", label_type="edge", padding=2)

    plt.title("One-Shot Similarity per Version Folder, Across Multiple Methods")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=30)

    # Move the legend so it does not interfere with the main plot
    plt.legend(
        title="Method",
        loc="upper left",
        bbox_to_anchor=(1.05, 1)
    )

    plt.tight_layout()
    plt.show()
    
    # 11) Print a detailed table
    print("\nDetailed Similarity Scores (per image):")
    print(df.to_string(index=False))
