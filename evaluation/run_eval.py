from evaluation_metrics import run_multiple_result_sets
import os
from pathlib import Path


if __name__ == '__main__':

    workdir = "workdir"
    gt_dir= os.path.join(workdir,"GroundTruth")
    infer_root = os.path.join(workdir,"InstanceResults")
    cfs_dir= os.path.join(gt_dir,"GT_cfs")
    output_csv=os.path.join(workdir,"evalResults")
    n_jobs=8  
    min_size = 1000

    run_multiple_result_sets(
        gt_roots={
            'pengwin3D': Path(os.path.join(gt_dir , "GT_pengwin3D")),
        },
        infer_root=infer_root,
        workdir=output_csv,
        min_size=1000,
        debug=True,
        cfs_root=cfs_dir,
        n_jobs=n_jobs,
        ccfed="raw"
    )