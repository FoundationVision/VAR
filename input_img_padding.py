import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


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
    
    return patches


def img_resize(folder_path, patch_size=256):
    """
    Read all images from a folder, pad and cut each into patches, and collect all patches into one list.
    
    Args:
        folder_path (str): Path to the folder containing images.
        patch_size (int): Size of each patch (default 256).
    
    Returns:
        List of all patches from all images.
    """
    all_patches = []

    # Get img files from the folder
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]

    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)

        patches = input_img_pad_and_cut(img, patch_size=patch_size)
        all_patches.extend(patches)

    return all_patches
