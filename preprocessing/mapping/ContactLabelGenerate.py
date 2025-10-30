import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import SimpleITK as sitk
import scipy.ndimage as ndi
import os
from scipy.ndimage import gaussian_filter
from scipy.ndimage import binary_erosion, label
import logging

def setup_logging(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )

def load_nifti(file_path):
    image = sitk.ReadImage(file_path)
    return image


def save_image(original_image, output_array, output_path):
    output_image = sitk.GetImageFromArray(output_array.astype(np.float32))
    output_image.CopyInformation(original_image)
    sitk.WriteImage(output_image, output_path)

def calculate_contact_voxels(array, kernel_size=3, device='cuda'):
    
    background_value = np.min(array)

    tensor = torch.from_numpy(array).to(device=device, dtype=torch.int32)
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, D, H, W]
    
    pad_size = kernel_size // 2
    padded = F.pad(tensor, 
                  pad=(pad_size, pad_size, pad_size, pad_size, pad_size, pad_size), 
                  mode='constant', value=0)
    
    D, H, W = tensor.shape[2], tensor.shape[3], tensor.shape[4]
    
    # Initialize mask
    mask = torch.zeros_like(tensor, dtype=torch.bool, device=device)
    
    # Generate all possible shifts and accumulate mask
    for dx in range(-pad_size, pad_size + 1):
        for dy in range(-pad_size, pad_size + 1):
            for dz in range(-pad_size, pad_size + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                # Extract shifted tensor
                shifted = padded[:, :, 
                                 pad_size + dx : pad_size + dx + D,
                                 pad_size + dy : pad_size + dy + H,
                                 pad_size + dz : pad_size + dz + W]
                # Update mask
                # mask |= (tensor != background_value) & (shifted != background_value) & (shifted != tensor)
                mask |= (tensor > 0) & (shifted > 0) & (shifted != tensor)
    
    # Convert mask to integer
    marked = mask.int()
    
    # Move to CPU and convert to numpy
    marked_np = marked.squeeze().cpu().numpy()
    
    return marked_np

def mark_contact_voxels(array, kernel_size=3, device='cuda',mode = 'training'):

    contact_voxels = calculate_contact_voxels(array, kernel_size=kernel_size, device=device)
    if mode == 'training':
        labeled = -np.ones_like(array, dtype=np.uint8)
    else:
        labeled = np.zeros_like(array, dtype=np.uint8)
    
    labeled[(array > 0) & (contact_voxels == 1)] = 2
    
    labeled[(array > 0) & (contact_voxels == 0)] = 1
    
    return labeled

def compute_CFSM(input_array_4d, kernel_size):

    input_array = input_array_4d[0]
    # input_array = input_array_4d

    if np.unique(input_array).max()>0:

        contact_voxels = mark_contact_voxels(input_array,kernel_size)

        # print('distance_map_matrix.shape:',distance_map_matrix.shape)
        contact_voxels = contact_voxels.astype(np.int8)
        contact_voxels[contact_voxels==0] = -1
        
    
    contact_voxels_4d = np.expand_dims(contact_voxels,axis=0)

    return contact_voxels_4d
