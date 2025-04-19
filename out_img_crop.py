import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

def output_img_concat_and_crop(patches, size_info, patch_size=256):
    """
    Concatenate list of patches into one big image, then crop back based on original size ratio.
    
    Args:
        patches (List[np.ndarray]): List of (patch_size, patch_size, C) patches.
        size_info (dict): Dictionary with 'original_size' and 'padded_size' info.
        patch_size (int): Size of patches (default 256).
    
    Returns:
        final_image (np.ndarray): Cropped image matching original size ratio.
    """
    original_h, original_w = size_info["original_size"]
    padded_h, padded_w = size_info["padded_size"]

    num_cols = math.ceil(padded_w / patch_size) # number of img along width
    num_rows = math.ceil(padded_h / patch_size) # number of img along height

    # 1. Concatenate patches
    rows = []
    for r in range(num_rows):
        row_patches = patches[r * num_cols : (r + 1) * num_cols]
        row = np.concatenate(row_patches, axis=1)  # horizontal concat
        rows.append(row)

    full_image = np.concatenate(rows, axis=0)  # vertical concat

    # 2. Crop based on original size ratio
    scale_h = padded_h / original_h
    scale_w = padded_w / original_w

    final_h = int(full_image.shape[0] / scale_h)
    final_w = int(full_image.shape[1] / scale_w)

    final_image = full_image[:final_h, :final_w]

    return final_image



def input_img_pad_and_cut(img, patch_size=256):
    """
    Pad the input image so that both dimensions are multiples of patch_size,
    then cut into non-overlapping patches of size (patch_size x patch_size).
    
    Args:
        img (np.ndarray): Input image as a NumPy array (H, W, C) or (H, W).
        patch_size (int): Size of each patch (default 256).
        
    Returns:
        List of patches (each as a np.ndarray of shape (patch_size, patch_size, C))
    """
    h, w = img.shape[:2]

    # Calculate padding sizes
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size

    padded_h = h + pad_h
    padded_w = w + pad_w

    size_info = {
        "original_size": (h, w),
        "padded_size": (padded_h, padded_w)
    }

    # Pad the image with zeros 
    padded_img = np.pad(
        img,
        ((0, pad_h), (0, pad_w), (0, 0)) if img.ndim == 3 else ((0, pad_h), (0, pad_w)),
        mode='constant',
        constant_values=0
    )

    new_h, new_w = padded_img.shape[:2]
    
    patches = []
    
    # Cut into 256*256 patches
    for i in range(0, new_h, patch_size):
        for j in range(0, new_w, patch_size):
            patch = padded_img[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    
    return patches, size_info



# ==== Full testing code ====

# 1. Load your image
img = cv2.imread('test2.png')  # Replace with your actual image path
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. Step 1: Pad and Cut into 256x256 patches
patches, size_info= input_img_pad_and_cut(img, patch_size=256)

# 3. Step 2: Resize each patch from 256x256 → 512x512
resized_patches = [cv2.resize(patch, (512, 512), interpolation=cv2.INTER_LINEAR) for patch in patches]
print(size_info)

# 5. Step 4: Concat and crop back
final_image = output_img_concat_and_crop(resized_patches, size_info, patch_size=512)

# 6. Step 5: Visualize
plt.figure(figsize=(10, 10))
plt.imshow(final_image)
plt.axis('off')
plt.title('Reconstructed Final Image')
plt.show()







