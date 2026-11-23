"""Run the six stages from the project directory or any working directory."""
import argparse
import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parent
STAGES = [
    "01_data_exploration", "02_quality_control", "03_preprocessing",
    "04_clustering", "05_cell_annotation", "06_cell_classification",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results_matched")
    parser.add_argument("--pbmc3k-only", action="store_true", help="Run stages 1–5")
    args = parser.parse_args()
    output = args.output.resolve()
    for protected in (ROOT / "results", ROOT / "results_v2"):
        if output == protected or protected in output.parents:
            raise ValueError("Preserve results and results_v2. Choose a new output directory.")
    if (output / "tables" / "classification_metrics.csv").exists():
        raise FileExistsError(f"Results already exist in {output}. Choose a fresh output directory.")
    for subdir in ("figures", "tables", "models"):
        (output / subdir).mkdir(parents=True, exist_ok=True)
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["NUMBA_NUM_THREADS"] = "2"
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".cache" / "numba"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
    local_deps = ROOT / ".local-deps"
    if local_deps.is_dir():
        sys.path.insert(0, str(local_deps))
    sys.path.insert(0, str(ROOT / "src"))

    adata = None
    timings = {}
    for stage in STAGES[:5] if args.pbmc3k_only else STAGES:
        print(f"Running {stage}", flush=True)
        start = perf_counter()
        adata = importlib.import_module(stage).run(adata, ROOT, output)
        timings[stage] = round(perf_counter() - start, 3)
    packages = ["scanpy", "anndata", "numpy", "scipy", "pandas",
                "matplotlib", "seaborn", "igraph", "scikit-learn",
                "numba", "umap-learn", "joblib"]
    if not args.pbmc3k_only:
        packages.append("torch")
    record = {
        "python": sys.version.split()[0], "python_path": sys.executable,
        "packages": {name: version(name) for name in packages},
        "stage_seconds": timings, "total_seconds": round(sum(timings.values()), 2),
    }
    (output / "run_metadata.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Finished in {record['total_seconds']} s. Results: {output}")


if __name__ == "__main__":
    main()
