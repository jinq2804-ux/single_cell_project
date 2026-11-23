"""Rank markers on all retained log-expression genes and annotate cautiously."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

from common import save_figure

# Key:候选细胞类型, Value:识别这种细胞类型的一组 Marker Gene。
PANELS = {
    "CD4-like T": ["IL7R", "LTB", "CCR7", "MAL", "CD3D", "TRAC"],
    "CD8-like T": ["CD8A", "CD8B", "TRAC", "CD3D", "CCL5"],
    "NK": ["NKG7", "GNLY", "KLRD1", "PRF1", "GZMB"],
    "B": ["MS4A1", "CD79A", "CD79B", "CD74", "HLA-DRA"],
    "CD14-like monocyte": ["LYZ", "CD14", "S100A8", "S100A9", "FCN1"],
    "FCGR3A-like monocyte": ["FCGR3A", "MS4A7", "LST1", "IFITM3", "CST3"],
    "Dendritic-like": ["FCER1A", "CD1C", "CLEC10A", "HLA-DPA1"],
    "Platelet-like": ["PPBP", "PF4", "NRGN"],
}
# 关键标记基因
ANCHORS = {
    "CD4-like T": ["IL7R", "CCR7", "MAL"], "CD8-like T": ["CD8A", "CD8B"],
    "NK": ["GNLY", "KLRD1"], "B": ["MS4A1", "CD79A", "CD79B"],
    "CD14-like monocyte": ["CD14", "FCN1", "S100A8"],
    "FCGR3A-like monocyte": ["FCGR3A", "MS4A7"],
    "Dendritic-like": ["FCER1A", "CD1C", "CLEC10A"], "Platelet-like": ["PPBP", "PF4"],
}


def run(adata, _root, output):
    # 根据已有的聚类标签，对每个聚类与其他细胞进行差异表达分析，并对基因进行排序
    sc.tl.rank_genes_groups(adata,
                            "leiden", # 按照 Leiden 聚类标签分组
                            method="wilcoxon",
                            use_raw=False, # 使用当前 adata.X
                            tie_correct=True,
                            pts=True)
    ranked = sc.get.rank_genes_groups_df(adata, group=None)
    ranked.to_csv(output / "tables" / "cluster_de.csv", index=False)
    # 从所有 Marker Gene 面板中收集基因，并且只保留当前数据集实际存在的基因
    genes = [gene
             for gene in dict.fromkeys(
                 g for panel in PANELS.values()
                 for g in panel)
             if gene in adata.var_names]
    # 把选中的 Marker Gene 表达值提取出来，转换为 DataFrame
    expr = pd.DataFrame(
        adata[:, genes].X.toarray(),
        index=adata.obs_names,
        columns=genes)
    cluster = adata.obs.leiden
    means = expr.groupby(cluster, observed=True).mean() # 基因在细胞群的整体表达水平
    fractions = expr.gt(0).groupby(cluster, observed=True).mean() # 有多少比例的细胞检测到了该基因
    # 判断细胞类型
    mapping = []
    evidence = []
    for group in means.index:
        mask = cluster == group
        delta = means.loc[group] - expr.loc[~mask].mean()
        qvals = ranked.loc[ranked.group == group].set_index("names").pvals_adj
        candidates = []
        for cell_type, panel in PANELS.items():
            present = [g for g in panel if g in genes]
            support = [g for g in present
                       if fractions.loc[group, g] >= .15
                       and delta[g] >= .15
                       and qvals[g] < .05]
            # 通过Marker Gene 和 Anchor Gene 筛选
            anchors = set(support).intersection(ANCHORS[cell_type])
            score = float(delta[support].sum() / len(panel))
            eligible = len(support) >= 2 and bool(anchors)
            if cell_type in {"CD4-like T", "CD8-like T"}:
                eligible = eligible and any(fractions.loc[group, g] >= .15
                                            for g in ("CD3D", "TRAC") if g in genes)
            evidence.append((group, cell_type, score, ";".join(support), eligible))
            if eligible:
                candidates.append((score, cell_type, support))
        candidates.sort(reverse=True)
        if not candidates or (len(candidates) > 1 and candidates[1][0] >= .75 * candidates[0][0]):
            annotation = "Uncertain"
        else:
            annotation = candidates[0][1]
        top = ranked.loc[ranked.group == group, "names"].head(5)
        best_markers = ";".join(candidates[0][2]) if candidates else ""
        mapping.append((group, annotation, int(mask.sum()), best_markers, ";".join(top)))
    columns = ["leiden", "cell_type", "cells", "supporting_panel_genes", "top_wilcoxon_genes"]
    annotations = pd.DataFrame(mapping, columns=columns)
    annotations.to_csv(output / "tables" / "cell_annotation.csv", index=False)
    type_markers = []
    for cell_type, rows in annotations.groupby("cell_type"):
        supported = dict.fromkeys(
            gene for entry in rows.supporting_panel_genes for gene in entry.split(";") if gene
        )
        type_markers.append((cell_type, ",".join(rows.leiden), rows.cells.sum(), ";".join(supported)))
    pd.DataFrame(type_markers, columns=["cell_type", "clusters", "cells", "supporting_panel_genes"]).to_csv(
        output / "tables" / "cell_type_markers.csv", index=False)
    pd.DataFrame(evidence, columns=["leiden", "candidate", "score", "support", "eligible"]).to_csv(
        output / "tables" / "annotation_evidence.csv", index=False)
    adata.obs["cell_type"] = pd.Categorical(cluster.map(annotations.set_index("leiden").cell_type.to_dict())) # 把聚类编号映射为对应的细胞类型
    sc.pl.umap(adata, color="cell_type", title="PBMC3k marker-based annotations", show=False)
    save_figure(output, "umap_cell_type")

    plot = sc.pl.dotplot(adata, genes, groupby="leiden", use_raw=False, return_fig=True,
                        figsize=(15, 4.2), title="PBMC3k marker expression by cluster")
    plot.savefig(output / "figures" / "marker_dotplot.png", dpi=250, bbox_inches="tight")
    plt.close("all")

    # A per-gene z-score highlights patterns across clusters on comparable scales.
    deviation = means.std(ddof=0).replace(0, 1)
    z = (means - means.mean()) / deviation
    fig, ax = plt.subplots(figsize=(13, 4.5), layout="constrained")
    sns.heatmap(z, cmap="vlag", center=0, ax=ax,
                cbar_kws={"label": "Gene-wise z-score of cluster mean"})
    ax.set(xlabel="Marker gene", ylabel="Leiden cluster", title="PBMC3k marker profiles")
    ax.tick_params(axis="x", labelrotation=60)
    save_figure(output, "marker_heatmap", fig)

    adata.uns["annotation_method"] = "Cluster-vs-rest Wilcoxon and multi-gene expression panels"
    adata.write_h5ad(output / "pbmc3k_processed.h5ad", compression="gzip")
    return adata
