import torch
import torch.nn.functional as F
import numpy as np
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

def calculate_contact_voxels(array, kernel_size=5, device='cuda'):


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
                mask |= (tensor > 0) & (shifted == background_value)
    
    # Convert mask to integer
    marked = mask.int()
    
    # Move to CPU and convert to numpy
    marked_np = marked.squeeze().cpu().numpy()
    
    return marked_np

'''
input: oringinal_image: 3D
output: 
'''
def generate_inner_outer_array(original_image, output_base_dir):
    image_array = sitk.GetArrayFromImage(original_image)
    unique_labels = np.unique(image_array)
    unique_labels = unique_labels[unique_labels != 0] 

    combined_label_outer_inner_eroded = np.full(image_array.shape, 0, dtype=np.float32) 
    
    for label in unique_labels:
        logging.info(f'Currently processing label: {label}') 
        binary_image = (image_array == label).astype(np.uint8)
        
        output_dir = os.path.join(output_base_dir, f'label_{label}')
        os.makedirs(output_dir, exist_ok=True)
        label_outer_inner_eroded = np.zeros_like(binary_image)

        # border calculation
        label_eroded_array = calculate_contact_voxels(binary_image,5) # border ==1
        label_outer_inner_eroded[binary_image==1] = 2
        label_outer_inner_eroded[label_eroded_array==1] = 1

        combined_label_outer_inner_eroded[binary_image == 1] = label_outer_inner_eroded[binary_image == 1]
        
    folder_path, vsdm_folder_name = os.path.split(output_base_dir)
    save_image(original_image, combined_label_outer_inner_eroded, os.path.join(folder_path, vsdm_folder_name + '_eroded.nii.gz'))
    return combined_label_outer_inner_eroded

def _bbox_from_mask(mask):
    pts = np.argwhere(mask)
    if pts.size == 0:
        return [[0, 0]] * 3
    mn, mx = pts.min(0), pts.max(0) + 1
    return [[int(a), int(b)] for a, b in zip(mn, mx)]

def _bbox_to_slice(bbox):
    return tuple(slice(lo, hi) for lo, hi in bbox)


from scipy.ndimage import distance_transform_edt, gaussian_filter
from typing import Literal

def compute_BIRM(
        input_array_4d: np.ndarray,
        spacing,
        kernel_size: int = 3,
        method: Literal['shift', 'laplacian'] = 'shift',
        *,
        d_threshold: float = 0.11,
        d2_threshold: float = 6,
        sigma: float = 2
    ) -> np.ndarray:

    seg = input_array_4d[0]             
    labels = np.unique(seg)
    labels = labels[labels > 0]

    out = np.full(seg.shape, -1, dtype=np.int8)   

    if method == 'shift':
        for lb in labels:
            obj = (seg == lb).astype(np.uint8)
            border = calculate_contact_voxels(obj, kernel_size)
            out[obj == 1] = 2           
            out[border == 1] = 1        
    elif method == 'laplacian':
        for lb in labels:
            obj = seg == lb
            if obj.sum() == 0:
                continue
            bbox = _bbox_from_mask(obj)
            slc = _bbox_to_slice(bbox)
            dist = distance_transform_edt(obj[slc])
            dist = gaussian_filter(dist, sigma=sigma)

            lap = sum(np.gradient(np.gradient(dist, axis=i), axis=i) for i in range(3))
            inside = (-lap) > d_threshold
            inside |= dist > d2_threshold

            local = np.ones_like(obj[slc], dtype=np.int8)  
            local[inside] = 2                              
            out[slc][obj[slc]] = local[obj[slc]]

    return out[np.newaxis, ...].astype(np.int8)