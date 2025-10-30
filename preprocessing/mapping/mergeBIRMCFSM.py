import os
import SimpleITK as sitk
import numpy as np

def merge_birm_cfsm(seg_birm, seg_cfsm):
    merged = seg_birm.copy()
    merged[seg_cfsm == 2] = 3
    return merged

def batch_merge_birm_cfsm_sitk(birm_dir, cfsm_dir):

    result_dir = os.path.join(os.path.dirname(birm_dir.rstrip('/')), "result")
    os.makedirs(result_dir, exist_ok=True)

    birm_files = [f for f in os.listdir(birm_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
    for fname in birm_files:
        birm_path = os.path.join(birm_dir, fname)
        cfsm_path = os.path.join(cfsm_dir, fname)
        if not os.path.exists(cfsm_path):
            continue

        birm_img = sitk.ReadImage(birm_path)
        cfsm_img = sitk.ReadImage(cfsm_path)
        birm_data = sitk.GetArrayFromImage(birm_img)    # z,y,x
        cfsm_data = sitk.GetArrayFromImage(cfsm_img)    # z,y,x

        # 4D (1, z, y, x)
        birm_data_4d = birm_data[np.newaxis, ...]
        cfsm_data_4d = cfsm_data[np.newaxis, ...]

        merged_4d = merge_birm_cfsm(birm_data_4d, cfsm_data_4d)
        merged_3d = np.squeeze(merged_4d)

        merged_img = sitk.GetImageFromArray(merged_3d)
        merged_img.CopyInformation(birm_img)

        out_path = os.path.join(result_dir, fname)
        sitk.WriteImage(merged_img, out_path)
