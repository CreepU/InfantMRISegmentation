import numpy as np
from skimage import feature
from skimage import exposure
from skimage import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# normalize each itensity to unit l2 norm 

def normalize_l2(image): 
    # Reshape the 3D image to a 1D array
    flattened_image = image.flatten()

    # Calculate the L2 norm
    l2_norm = np.linalg.norm(flattened_image, ord=2)

    # Normalize the image by dividing each voxel by the L2 norm
    normalized_image = image / l2_norm

    return normalized_image


def mean_intensity_difference(image, patch_size):
    result = np.zeros_like(image, dtype=np.float32)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                # Define cubical region R1 around voxel (i, j, k)
                start_i1, start_j1, start_k1 = max(0, i - patch_size // 2), max(0, j - patch_size // 2), max(0, k - patch_size // 2)
                end_i1, end_j1, end_k1 = min(image.shape[0], i + patch_size // 2 + 1), min(image.shape[1], j + patch_size // 2 + 1), min(image.shape[2], k + patch_size // 2 + 1)

                # Define cubical region R2 (asymmetric to R1)
                start_i2, start_j2, start_k2 = max(0, i - patch_size // 2), max(0, j - patch_size // 2), max(0, k - patch_size // 2)
                end_i2, end_j2, end_k2 = min(image.shape[0], i + patch_size // 2, i + patch_size), min(image.shape[1], j + patch_size // 2, j + patch_size), min(image.shape[2], k + patch_size // 2, k + patch_size)

                # Calculate mean intensity difference over R1 and R2
                mean_diff = np.mean(image[start_i1:end_i1, start_j1:end_j1, start_k1:end_k1]) - np.mean(image[start_i2:end_i2, start_j2:end_j2, start_k2:end_k2])

                # Assign the mean intensity difference to the result
                result[i, j, k] = mean_diff

    return result

def ExtractFeature(image,patch_size,b=0):
    feature_image = np.zeros_like(image,dtype=np.float32)
    
    return 