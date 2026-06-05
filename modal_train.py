"""
Modal training job — DataCo Late Delivery Risk Modeling
Run with: modal run modal_train.py
Artifacts are saved to a Modal Volume, then downloaded locally.
"""

from __future__ import annotations

from pathlib import Path
import modal

# Train on the canonical cleaned dataset (01_ingestion_cleaning.py output):
# 29 columns, leakage + PII already removed. This keeps the model's expected
# feature set identical to the documented pipeline and to what app/app.py feeds.
CLEAN_CSV = "/opt/study/DAP/Data/filtered/clean_data.csv"
SCRIPT = "/opt/study/DAP/src-code/05_modeling.py"

volume = modal.Volume.from_name("dap-outputs")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "matplotlib",
        "seaborn",
        "shap",
    )
    .add_local_file(CLEAN_CSV, remote_path="/root/data/clean_data.csv")
    .add_local_file(SCRIPT, remote_path="/root/src/05_modeling.py")
)

app = modal.App("dap-dataco-training", image=image)


@app.function(
    gpu="T4",
    timeout=1800,
    volumes={"/root/outputs": volume},
)
def train() -> None:
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "modeling", "/root/src/05_modeling.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mod.CLEAN_CSV = Path("/root/data/clean_data.csv")
    mod.RAW_CSV = Path("/nonexistent/DataCoSupplyChainDataset.csv")
    mod.OUTPUT_DIR = Path("/root/outputs")
    mod.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mod.main()

    # Commit writes to the volume so they persist
    volume.commit()
    print("Artifacts committed to volume.")


@app.local_entrypoint()
def main():
    print("Submitting training job to Modal (GPU: T4)...")
    train.remote()

    # Download artifacts from volume. Refresh the local handle first: it was
    # created at import time, before train.remote() committed, so without a
    # reload listdir() returns a stale snapshot and silently misses the
    # freshly-written models (e.g. the per-model .pkl files).
    volume.reload()

    out_root = Path("Data/filtered/model_outputs")
    out_root.mkdir(parents=True, exist_ok=True)

    count = 0
    for entry in volume.listdir("/", recursive=True):
        remote_path = entry.path
        dest = out_root / remote_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        contents = b"".join(volume.read_file(remote_path))
        dest.write_bytes(contents)
        print(f"  saved: {dest}")
        count += 1

    print(f"\nDone — {count} artifacts saved to {out_root}/")
