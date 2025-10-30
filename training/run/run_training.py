import os
import json
from pathlib import Path
import importlib
import torch

# ======= Minimal user-editable settings =======

DATASET_DIR = None

TRAINER_SPEC = "InstanceSegFrac.trainers:ContrastiveAttentionMergedTraining"

# InstanceSegFrac configuration and fold
CONFIGURATION = "3d_fullres"
FOLD = 0

# Output directory for InstanceSegFrac results (checkpoints, logs, etc.)
RESULTS_DIR = Path("./Results")

# ======= Helper functions (no need to edit) =======

def log(msg: str):
    print(msg, flush=True)

def read_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def autodetect_dataset_dir() -> Path:

    root = Path("./Preprocessed")
    if root.is_dir():
        for sub in sorted(root.iterdir()):
            if (sub / "dataset.json").is_file():
                return sub

def find_plans_file(dataset_dir: Path) -> Path:
    """
    Return the path to the plans file located in the dataset directory.
    """
    for name in ("plans.json"):
        f = dataset_dir / name
        if f.is_file():
            return f
    raise FileNotFoundError(f"No plans file found in {dataset_dir}")

def import_trainer(trainer_spec: str):
    """
    Import a trainer from a "module.path:ClassName" spec and return the class object.
    """
    module_name, cls_name = trainer_spec.split(":", 1)
    mod = importlib.import_module(module_name)
    if not hasattr(mod, cls_name):
        raise ImportError(f"Class {cls_name} not found in module {module_name}")
    return getattr(mod, cls_name)

def main():
    # 1) Resolve dataset directory
    dataset_dir = Path(DATASET_DIR).resolve() if DATASET_DIR else autodetect_dataset_dir()
    if not (dataset_dir / "dataset.json").is_file():
        raise FileNotFoundError(f"{dataset_dir}/dataset.json does not exist")
    plans_file = find_plans_file(dataset_dir)

    # 2) Device selection (single GPU if available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)

    torch.backends.cudnn.benchmark = True

    # 3) Load plans and dataset.json
    plans = read_json(plans_file)
    dataset_json = read_json(dataset_dir / "dataset.json")

    # 4) Import and instantiate your custom trainer
    TrainerCls = import_trainer(TRAINER_SPEC)
    trainer = TrainerCls(
        plans=plans,
        configuration=CONFIGURATION,
        fold=FOLD,
        dataset_json=dataset_json,
        unpack_dataset=True,
        device=device
    )

    # 5) Run training
    trainer.run_training()
    log("==== Training finished. ====")

if __name__ == "__main__":
    main()
