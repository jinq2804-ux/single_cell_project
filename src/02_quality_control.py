"""Calculate QC metrics and filter the empirical tails of PBMC3k."""
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import seaborn as sns

from common import save_figure

METRICS = {"n_genes_by_counts": "Detected genes",
           "total_counts": "Total UMI counts", "pct_counts_mt": "Mitochondrial counts (%)"}
MIN_GENES = 330
MAX_GENES = 1900
MAX_MT = 6.0
MIN_CELLS_PER_GENE = 3


def run(adata, _root, output):
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    # 数据标记和保存
    qc = adata.obs[list(METRICS)].copy()
    qc["low_genes"] = qc.n_genes_by_counts < MIN_GENES
    qc["high_genes"] = qc.n_genes_by_counts > MAX_GENES
    qc["high_mt"] = qc.pct_counts_mt > MAX_MT
    qc["kept"] = ~qc[["low_genes", "high_genes", "high_mt"]].any(axis=1)
    tables = output / "tables"
    qc.to_csv(tables / "qc_per_cell.csv", index_label="cell_id")
    qc[list(METRICS)].describe(percentiles=[.01, .025, .25, .5, .75, .95, .99, .995]).to_csv(
        tables / "qc_summary.csv")

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.8), layout="constrained")
    limits = [[MIN_GENES, MAX_GENES], [], [MAX_MT]]
    for ax, (metric, label), thresholds in zip(axes, METRICS.items(), limits):
        sns.violinplot(y=qc[metric], ax=ax, color="#76B6C4", cut=0, inner="quart")
        ax.set_ylabel(label)
        ax.set_xticks([])
        for cutoff in thresholds:
            ax.axhline(cutoff, color="#A33D32", ls="--", linewidth=1)
    fig.suptitle("PBMC3k quality metrics before filtering")
    save_figure(output, "qc_violin", fig)
    for metric, name, thresholds in [
        ("n_genes_by_counts", "qc_counts_vs_genes", limits[0]),
        ("pct_counts_mt", "qc_counts_vs_mt", limits[2]),
    ]:
        fig, ax = plt.subplots(figsize=(6.2, 4.2), layout="constrained")
        for kept, group in qc.groupby("kept"):
            ax.scatter(group.total_counts, group[metric], s=10, alpha=.55,
                       label="Kept" if kept else "Filtered",
                       color="#237A92" if kept else "#B34C37")
        for cutoff in thresholds:
            ax.axhline(cutoff, color="#733B33", ls="--", lw=1)
        ax.set(xscale="log", xlabel=METRICS["total_counts"], ylabel=METRICS[metric],
               title="PBMC3k quality control")
        ax.legend(frameon=False)
        save_figure(output, name, fig)

    # 数据过滤
    before = adata.shape
    # 过滤细胞
    filtered = adata[qc.kept].copy()
    cells_only = filtered.shape
    # 过滤基因
    sc.pp.filter_genes(filtered, min_cells=MIN_CELLS_PER_GENE)

    # 汇总过滤前后信息
    pd.DataFrame([before, cells_only, filtered.shape],
                 index=["raw", "after_cell_qc", "after_gene_qc"],
                 columns=["cells", "genes"]).to_csv(tables / "qc_filter_counts.csv", index_label="stage")
    return filtered
