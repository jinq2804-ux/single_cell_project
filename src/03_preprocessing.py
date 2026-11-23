"""Keep counts and all-gene log expression; scale only HVGs for PCA."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from common import SEED, save_figure


def run(adata, _root, output):
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4) # 对每个细胞的表达计数进行缩放，使其总和达到同一个目标值
    sc.pp.log1p(adata)
    # HVG(高变基因筛选)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)
    adata.var.to_csv(output / "tables" / "gene_statistics.csv", index_label="gene")
    
    fig, ax = plt.subplots(figsize=(6.6, 4.2), layout="constrained")
    for hvg, frame in adata.var.groupby("highly_variable"):
        ax.scatter(frame.means, frame.dispersions_norm, s=6, alpha=.55,
                   color="#207785" if hvg else "#ACB8BE", label="HVG" if hvg else "Other")
    ax.set(xlabel="Mean expression (Scanpy Seurat mean)",
           ylabel="Mean-bin normalized dispersion", title="PBMC3k highly variable genes")
    ax.legend(frameon=False)
    save_figure(output, "hvg", fig)

    # 真正提取 2000 个高变基因
    hvg = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(hvg, max_value=10) # 对每个基因进行标准化

    # PCA(主成分分析)
    sc.pp.pca(hvg, n_comps=50, svd_solver="arpack", random_state=SEED)
    adata.obsm["X_pca"] = hvg.obsm["X_pca"]
    adata.uns["pca"] = hvg.uns["pca"]
    adata.uns["pca"]["params"]["mask_var"] = "highly_variable"
    loadings = np.zeros((adata.n_vars, 50), dtype=hvg.varm["PCs"].dtype)
    loadings[adata.var.highly_variable.to_numpy()] = hvg.varm["PCs"]
    adata.varm["PCs"] = loadings
    ratio = adata.uns["pca"]["variance_ratio"]
    pcs = np.arange(1, len(ratio) + 1)
    pd.DataFrame({"pc": pcs, "variance_ratio": ratio, "cumulative_ratio": ratio.cumsum()}).to_csv(
        output / "tables" / "pca_variance.csv", index=False)
    
    fig, ax = plt.subplots(figsize=(6.4, 3.7), layout="constrained")
    ax.plot(pcs, ratio * 100, marker=".", color="#207785")
    ax.axvline(30, color="#A33D32", ls="--", lw=1, label="30 PCs used for graph")
    ax.set(xlabel="Principal component", ylabel="Variance explained (%)",
           title="PCA of scaled highly variable genes")
    ax.legend(frameon=False)
    save_figure(output, "pca_variance", fig)
    
    return adata
