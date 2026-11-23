"""Load the existing Scanpy PBMC3k file and inspect the counts."""
import json

import numpy as np
import scanpy as sc
from scipy.sparse import issparse


def run(_adata, root, output):
    local = root / "data" / "pbmc3k_raw.h5ad"
    if local.is_file():
        adata = sc.read_h5ad(local)
    else:
        sc.settings.datasetdir = root / "data"
        adata = sc.datasets.pbmc3k()
    adata.var_names_make_unique()
    adata.X = adata.X.tocsr() if issparse(adata.X) else adata.X
    nonzero = int(adata.X.count_nonzero() if issparse(adata.X) else np.count_nonzero(adata.X))
    record = {
        "shape": list(adata.shape),
        "matrix_type": type(adata.X).__name__,
        "nonzero_entries": nonzero,
        "zero_fraction": 1 - nonzero / (adata.n_obs * adata.n_vars),
        "first_cell_ids": adata.obs_names[:8].tolist(),
        "first_genes": adata.var_names[:8].tolist(),
        "obs_columns": adata.obs.columns.tolist(),
        "var_columns": adata.var.columns.tolist(),
        "obsm_keys": list(adata.obsm),
    }
    print(adata, flush=True)
    (output / "tables" / "data_exploration.json").write_text(json.dumps(record, indent=2) + "\n")
    return adata
