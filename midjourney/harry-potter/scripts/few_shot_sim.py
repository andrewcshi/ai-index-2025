import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

# Keras application modules for CNN models
from tensorflow.keras.applications import (
    vgg16,
    resnet50,
    inception_v3,
    xception
)

# For pixel-based similarity (SSIM)
from PIL import Image
from skimage.metrics import structural_similarity as ssim

###############################################################################
# 1. Model Configuration
###############################################################################
# Order matters here: to process PixelBased first, list it first.
MODELS_TO_TRY = {
    "PixelBased": {
        "type": "pixel",
        "target_size": (224, 224)  # Make sure it's >= 7x7 for SSIM
    },
    "VGG16": {
        "type": "cnn",
        "model_fn": vgg16.VGG16,
        "preprocess_fn": vgg16.preprocess_input,
        "feature_layer": "fc2",  # penultimate dense layer in VGG16
        "target_size": (224, 224)
    },
    "ResNet50": {
        "type": "cnn",
        "model_fn": resnet50.ResNet50,
        "preprocess_fn": resnet50.preprocess_input,
        "feature_layer": "avg_pool",
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

###############################################################################
# 2. Utility Functions
###############################################################################

def pixel_based_similarity_score(img_path_1, img_path_2, target_size=(224,224)):
    """
    Computes pixel-based similarity using SSIM (Structural Similarity Index).
    1) Load both images in RGB.
    2) Resize to `target_size` so they're the same shape (>= 7x7).
    3) Compute SSIM on the [H, W, 3] arrays in [0..1].
    4) Return a value typically in [0..1], where 1 => identical.
    """
    # Load & convert to RGB
    img1 = Image.open(img_path_1).convert("RGB")
    img2 = Image.open(img_path_2).convert("RGB")

    # Resize (using modern Pillow's Resampling)
    img1 = img1.resize(target_size, Image.Resampling.LANCZOS)
    img2 = img2.resize(target_size, Image.Resampling.LANCZOS)

    # Convert to NumPy float arrays [H, W, 3] in [0..1]
    arr1 = np.array(img1, dtype=np.float32) / 255.0
    arr2 = np.array(img2, dtype=np.float32) / 255.0

    # For newer versions of scikit-image:
    #  - `multichannel` is deprecated.
    #  - Use `channel_axis=-1` to indicate channels-last.
    # Also, pass a smaller or equal "win_size" so it doesn't exceed image size.
    ssim_val = ssim(arr1, arr2, data_range=1.0, channel_axis=-1, win_size=7)

    # Ensure the value is >= 0 in case of floating underflow
    if ssim_val < 0.0:
        ssim_val = 0.0

    return ssim_val

def load_and_preprocess_image_cnn(img_path, preprocess_fn, target_size):
    """
    Loads an image from disk, resizes it to `target_size`, 
    and applies the appropriate CNN preprocess function.
    """
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_fn(x)
    return x

def get_cnn_feature_vector(base_model, img_path, preprocess_fn, target_size):
    """
    For CNN-based similarity: returns a flattened feature vector for the image.
    """
    x = load_and_preprocess_image_cnn(img_path, preprocess_fn, target_size)
    features = base_model.predict(x)
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
    For CNN-based approach: returns a similarity score (0..1) 
    from the cosine similarity of features.
    """
    fv1 = get_cnn_feature_vector(model, img_path_1, preprocess_fn, target_size)
    fv2 = get_cnn_feature_vector(model, img_path_2, preprocess_fn, target_size)
    sim = cosine_similarity(fv1, fv2)
    # Map [-1..1] to [0..1]
    return (sim + 1) / 2

###############################################################################
# 3. Main: Compare "few-shot" references to images in each folder, 
#          for multiple CNN models + Pixel-based method
###############################################################################

if __name__ == "__main__":
    # A) Define the references for few-shot
    ref_paths = [
        os.path.join("midjourney", "harry-potter", "few-shot", f"ref-few-shot-{i}.png")
        for i in range(1, 6)
    ]

    print(ref_paths)
    
    # B) Define subfolders to search
    version_folders = ["v1", "v2", "v3", "v4", "v5", "v5.1", "v5.2", "v6", "v6.1"]
    parent_folder = os.path.join("midjourney", "harry-potter", "few-shot")
    
    # C) We'll store results for all methods (CNN + Pixel).
    #    Each entry: (MethodName, Version, ImageName, Similarity)
    all_results = []
    
    # D) Loop over each method in MODELS_TO_TRY 
    #    => PixelBased will come first due to dictionary order
    for method_name, config in MODELS_TO_TRY.items():
        print(f"\n\n=== Processing with {method_name} ===")
        
        method_type = config["type"]
        target_size = config["target_size"]
        
        # 1) If it's a CNN, build the feature extractor
        if method_type == "cnn":
            model_constructor = config["model_fn"]
            preprocess_fn = config["preprocess_fn"]
            layer_name = config["feature_layer"]
            
            # Build the full model
            base_keras_model = model_constructor(weights="imagenet", include_top=True)
            # Extract from specified layer
            feature_extractor = Model(
                inputs=base_keras_model.input,
                outputs=base_keras_model.get_layer(layer_name).output
            )
        
        # 2) Iterate over folders & images
        for vfolder in version_folders:
            folder_path = os.path.join(parent_folder, vfolder)
            if not os.path.isdir(folder_path):
                print(f"Folder not found: {folder_path}")
                continue
            
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(".png"):
                    img_path = os.path.join(folder_path, filename)
                    
                    # E) Compute average similarity across reference images
                    per_ref_scores = []
                    
                    for ref_path in ref_paths:
                        if method_type == "cnn":
                            # CNN-based approach
                            preprocess_fn = config["preprocess_fn"]
                            score = cnn_similarity_score(
                                feature_extractor,
                                ref_path,
                                img_path,
                                preprocess_fn,
                                target_size
                            )
                        elif method_type == "pixel":
                            # Pixel-based approach
                            score = pixel_based_similarity_score(
                                ref_path,
                                img_path,
                                target_size=target_size
                            )
                        else:
                            raise ValueError(f"Unknown method type: {method_type}")
                        
                        per_ref_scores.append(score)
                    
                    # Average across references => final score for this image
                    avg_score = np.mean(per_ref_scores)
                    
                    # Store in all_results
                    all_results.append((method_name, vfolder, filename, avg_score))
    
    # F) Convert all results to a DataFrame for analysis
    df = pd.DataFrame(all_results, columns=["Model", "Version", "Image", "Similarity"])
    
    # G) Aggregate by (Model, Version) => average across images in that folder
    df_summary = df.groupby(["Model", "Version"], as_index=False)["Similarity"].mean()
    
    # H) Plot the aggregated results as a grouped bar chart
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))

    ax = sns.barplot(
        data=df_summary,
        x="Version",
        y="Similarity",
        hue="Model",
        palette="Blues_d"
    )

    # Optional: label each bar with numeric value
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", label_type="edge", padding=2)

    plt.title("Few-Shot Similarity by Version Folder, Across Multiple Methods")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=30)

    # Move legend to the upper-left corner (outside) so it doesn't overlap the plot
    plt.legend(
        title="Method",
        loc="upper left",        # position inside the figure area
        bbox_to_anchor=(1.05, 1) # place it just outside the axes on the right
    )

    plt.tight_layout()
    plt.show()

    
    # I) Print a detailed table
    print("\nDetailed Results (per image):")
    print(df.to_string(index=False))
