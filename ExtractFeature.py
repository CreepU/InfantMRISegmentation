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


# input data: patch_size_R is centered at the same 
def mean_intensity_difference(image, patch_size_R, patch_size_R1,b=0):
    len_R, len_R1 = (patch_size_R-1)/2, (patch_size_R1-1)/2
    len_random = len_R-len_R1
    result = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                # Define cubical region R1 around voxel (i, j, k)
                center_i1 = np.random.randint(max(0, i - len_random), min(image.shape[0] - patch_size_R + 1, i + len_random + 1))
                center_j1 = np.random.randint(j-len_random,j+len_random+1)
                center_k1 = np.random.randint(k-len_random,k+len_random+1)
                
                # Define cubical region R2 (asymmetric to R1)
                center_i2 = np.random.randint(i-len_random,i+len_random+1)
                center_j2 = np.random.randint(j-len_random,j+len_random+1)
                center_k2 = np.random.randint(k-len_random,k+len_random+1)

                # Calculate mean intensity difference over R1 and R2
                mean_diff = np.mean(image[center_i1-len_R1:center_i1+len_R1+1, center_j1-len_R1:center_j1+len_R1+1, 
                                        center_k1-len_R1:center_k1+len_R1+1]) 
                - b*np.mean(image[center_i2-len_R1:center_i2+len_R1+1, center_j2-len_R1:center_j2+len_R1+1, 
                                        center_k2-len_R1:center_k2+len_R1+1])

                # Assign the mean intensity difference to the result
                result[i, j, k] = mean_diff
                
    return result
