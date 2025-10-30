import os
import sys
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import SimpleITK as sitk
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure, label
from sklearn.utils import resample
from joblib import Parallel, delayed
import time
from datetime import datetime
# Tee class for logging
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

def extract_prefix(filename):

    suffixes_to_remove = [
        ".nii.gz"
    ]
    for suffix in suffixes_to_remove:
        if filename.endswith(suffix):
            return filename[:-len(suffix)]
    return filename


def load_gt_label_and_spacing(input_file):
    image = sitk.ReadImage(str(input_file))
    spacing = image.GetSpacing()[::-1]
    array = sitk.GetArrayFromImage(image)
    return spacing, array

def calculate_dice(vol1, vol2):
    intersection = np.logical_and(vol1, vol2).sum()
    return 2.0 * intersection / (vol1.sum() + vol2.sum() + 1e-8)

def calculate_3d_iou(vol1, vol2):
    intersection = np.logical_and(vol1, vol2).sum()
    union = np.logical_or(vol1, vol2).sum()
    return 0.0 if union == 0 else intersection / union

def calculate_3d_hd95_from_points(a, b):
    if not a.size or not b.size:
        return np.inf
    a = torch.from_numpy(a).float().cuda()
    b = torch.from_numpy(b).float().cuda()
    dist = torch.cdist(a[None], b[None])[0]
    d1 = torch.kthvalue(torch.min(dist, dim=1).values, int(0.95 * dist.shape[0]))[0]
    d2 = torch.kthvalue(torch.min(dist, dim=0).values, int(0.95 * dist.shape[1]))[0]
    return float(torch.max(d1, d2).cpu())

def calculate_3d_assd_from_points(a, b):
    if not a.size or not b.size:
        return np.inf
    a = torch.from_numpy(a).float().cuda()
    b = torch.from_numpy(b).float().cuda()
    dist = torch.cdist(a[None], b[None])[0]
    assd = 0.5 * (torch.min(dist, dim=1).values.mean() + torch.min(dist, dim=0).values.mean())
    return float(assd.cpu())

def extract_surface_points(volume, label, spacing, sample_size=10000):
    mask = (volume == label)
    struct = generate_binary_structure(3, 1)
    eroded = binary_erosion(mask, structure=struct)
    surface_mask = binary_dilation(mask, structure=struct) & ~eroded
    surface_points = np.argwhere(surface_mask)
    if surface_points.shape[0] > sample_size:
        surface_points = resample(surface_points, n_samples=sample_size, random_state=42)
    return surface_points * spacing

def calculate_sphere_radius(volume, label):
    points = np.argwhere(volume == label)
    if points.size == 0:
        return np.inf
    center = np.mean(points, axis=0)
    return np.max(np.linalg.norm(points - center, axis=1))

def match_labels(gt_volume, pred_volume, label_range):
    matches = {}
    for label in label_range:
        gt_mask = (gt_volume == label)
        if not gt_mask.any():
            continue
        iou_scores = {pl: calculate_3d_iou(gt_mask, pred_volume == pl) for pl in label_range if (pred_volume == pl).any()}
        matches[label] = max(iou_scores.items(), key=lambda x: x[1]) if iou_scores else (-1, 0)
    return matches

def match_labels_whole(gt_volume, pred_volume):
    return {**match_labels(gt_volume, pred_volume, range(1, 11)),
            **match_labels(gt_volume, pred_volume, range(11, 21)),
            **match_labels(gt_volume, pred_volume, range(21, 31))}

def evaluate_fracture_segmentation(matches, gt, pred, spacing):
    results = {}
    for label, (pred_label, iou) in matches.items():
        if iou > 0:
            gt_bin = (gt == label).astype(np.uint8)
            pred_bin = (pred == pred_label).astype(np.uint8)
            gt_pts = extract_surface_points(gt, label, spacing)
            pred_pts = extract_surface_points(pred, pred_label, spacing)
            dice = calculate_dice(gt_bin, pred_bin)
            hd95 = calculate_3d_hd95_from_points(gt_pts, pred_pts)
            assd = calculate_3d_assd_from_points(gt_pts, pred_pts)
        else:
            dice, iou = 0, 0
            radius = calculate_sphere_radius(gt, label)
            hd95 = 2 * radius
            assd = radius
        results[label] = (iou, dice, hd95, assd)
    return results

def evaluate_anatomical(gt, pred, spacing):
    gt_bin = (gt > 0).astype(np.uint8)
    pred_bin = (pred > 0).astype(np.uint8)
    iou = calculate_3d_iou(gt_bin, pred_bin)
    dice = calculate_dice(gt_bin, pred_bin)
    gt_pts = extract_surface_points(gt_bin, 1, spacing)
    pred_pts = extract_surface_points(pred_bin, 1, spacing)
    if pred_pts.size:
        hd95 = calculate_3d_hd95_from_points(gt_pts, pred_pts)
        assd = calculate_3d_assd_from_points(gt_pts, pred_pts)
    else:
        radius = calculate_sphere_radius(gt_bin, 1)
        hd95 = 2 * radius
        assd = radius
    return iou, dice, hd95, assd

def evaluate_local_dice(gt, pred, local):
    mask = (local == 1)
    gt_mask = (gt > 0).astype(np.uint8) * mask
    pred_mask = (pred > 0).astype(np.uint8) * mask
    return calculate_dice(gt_mask, pred_mask)

def evaluate_3d_single_case(gt, pred, spacing, local_mask=None):
    matches = match_labels_whole(gt, pred)
    fr = evaluate_fracture_segmentation(matches, gt, pred, spacing)
    fracture_iou = np.mean([v[0] for v in fr.values()])
    fracture_dice = np.mean([v[1] for v in fr.values()])
    fracture_hd95 = np.mean([v[2] for v in fr.values()])
    fracture_assd = np.mean([v[3] for v in fr.values()])
    fracture_local_dice = (
    evaluate_local_dice(gt, pred, local_mask) 
    if local_mask is not None and (local_mask > 0).any() 
    else -1)
    aiou, adice, ahd95, aassd = evaluate_anatomical(gt, pred, spacing)
    return {
        "fracture_iou": fracture_iou,
        "fracture_dice": fracture_dice,
        "fracture_local_dice": fracture_local_dice,
        "fracture_hd95": fracture_hd95,
        "fracture_assd": fracture_assd,
        "anatomical_iou": aiou,
        "anatomical_dice": adice,
        "anatomical_hd95": ahd95,
        "anatomical_assd": aassd
    }

def remove_small_components_multilabel(mask, min_size=1000, debug=True):
    mask_filtered = np.zeros_like(mask)
    for lab in np.unique(mask):
        if lab == 0: continue
        mask_lab = (mask == lab)
        labeled, num = label(mask_lab)
        for i in range(1, num+1):
            region = (labeled == i)
            if region.sum() >= min_size:
                mask_filtered[region] = lab
    return mask_filtered

from joblib import Parallel, delayed
import time
from datetime import datetime

def run_multiple_result_sets(
    gt_roots,
    infer_root,
    workdir,
    min_size=1000,
    debug=True,
    cfs_root="GT_cfs",
    log_filename='evaluation_debug_multiset.log',
    n_jobs=4,
    ccfed="both"  # 'raw', 'ccfed', 'both'
):
    import sys

    workdir = Path(workdir)
    workdir.mkdir(exist_ok=True, parents=True)

    log_path = workdir / log_filename
    f_log = open(log_path, 'w')
    sys.stdout = Tee(sys.stdout, f_log)

    cfs_dict_all = {}
    cfs_dir = Path(cfs_root)
    if cfs_dir.exists():
        cfs_files = list(cfs_dir.glob("*.nii.gz"))
        cfs_dict_all = {extract_prefix(f.name): f for f in cfs_files}
        print(f"[INFO] Loaded {len(cfs_dict_all)} CFS mask files")

    infer_dirs = [f for f in Path(infer_root).iterdir() if f.is_dir()]

    for gt_name, gt_dir in gt_roots.items():
        gt_files = sorted(list(gt_dir.glob("*.nii.gz")))
        gt_dict = {extract_prefix(f.name): f for f in gt_files}

        for infer_dir in infer_dirs:
            pred_files = sorted(list(infer_dir.glob("*.nii.gz")))
            pred_dict = {extract_prefix(f.name): f for f in pred_files}

            print(f"[DEBUG] GT keys: {list(gt_dict.keys())[:5]}")
            print(f"[DEBUG] Pred keys: {list(pred_dict.keys())[:5]}")

            common_keys = sorted(set(gt_dict) & set(pred_dict))
            total = len(common_keys)
            print(f"[{datetime.now().isoformat()}] {gt_name} x {infer_dir.name}: {total} matched cases")

            def evaluate_key(key, remove_cc=False):
                print(f"[{datetime.now().isoformat()}] Evaluating case: {key} | Remove CC: {remove_cc}")
                spacing, gt_volume = load_gt_label_and_spacing(gt_dict[key])
                _, pred_volume = load_gt_label_and_spacing(pred_dict[key])
                local_mask = None
                if key in cfs_dict_all:
                    _, local_mask = load_gt_label_and_spacing(cfs_dict_all[key])

                if remove_cc:
                    gt_volume = remove_small_components_multilabel(gt_volume, min_size=min_size, debug=debug)
                    pred_volume = remove_small_components_multilabel(pred_volume, min_size=min_size, debug=debug)

                metrics = evaluate_3d_single_case(gt_volume, pred_volume, spacing, local_mask)
                metrics['case'] = key
                print(f"[{datetime.now().isoformat()}] Finished case: {key}")
                return metrics

            if ccfed in ("raw", "both"):
                print(f"[{datetime.now().isoformat()}] ===== Start RAW evaluation ({gt_name} x {infer_dir.name}) =====")
                print(f"[INFO] Start parallel raw evaluation with {n_jobs} workers...")
                start = time.time()
                results_raw = Parallel(n_jobs=n_jobs)(
                    delayed(evaluate_key)(key, remove_cc=False) for key in common_keys
                )
                print(f"[{datetime.now().isoformat()}] ===== RAW evaluation finished in {time.time() - start:.1f} sec =====")

                df_raw = pd.DataFrame(results_raw)
                out_csv_raw = workdir / f"eval_GT_{gt_name}_to_{infer_dir.name}.csv"
                df_raw.to_csv(out_csv_raw, index=False)

            if ccfed in ("ccfed", "both"):
                print(f"[{datetime.now().isoformat()}] ===== Start CCFED evaluation ({gt_name} x {infer_dir.name}) =====")
                print(f"[INFO] Start parallel CCFED evaluation with {n_jobs} workers...")
                start = time.time()
                results_ccfed = Parallel(n_jobs=n_jobs)(
                    delayed(evaluate_key)(key, remove_cc=True) for key in common_keys
                )
                print(f"[{datetime.now().isoformat()}] ===== CCFED evaluation finished in {time.time() - start:.1f} sec =====")

                df_ccfed = pd.DataFrame(results_ccfed)
                out_csv_ccfed = workdir / f"eval_GT_{gt_name}_to_{infer_dir.name}_ccfed.csv"
                df_ccfed.to_csv(out_csv_ccfed, index=False)

    print("[FINISHED]")



def run_single_case(gt_path, pred_path, cfs_path=None, min_size=1000, remove_cc=False, debug=False):
    spacing, gt_volume = load_gt_label_and_spacing(gt_path)
    _, pred_volume = load_gt_label_and_spacing(pred_path)
    local_mask = None
    if cfs_path and Path(cfs_path).exists():
        _, local_mask = load_gt_label_and_spacing(cfs_path)

    if remove_cc:
        gt_volume = remove_small_components_multilabel(gt_volume, min_size=min_size, debug=debug)
        pred_volume = remove_small_components_multilabel(pred_volume, min_size=min_size, debug=debug)

    metrics = evaluate_3d_single_case(gt_volume, pred_volume, spacing, local_mask)
    return metrics