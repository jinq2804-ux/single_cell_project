"""Build one PCA neighbor graph for both UMAP and Leiden."""
import matplotlib.pyplot as plt
import scanpy as sc

from common import SEED, save_figure


def run(adata, _root, output):
    # KNN邻居图, 自动生成adata.obsp["distances"],adata.obsp["connectivities"]
    sc.pp.neighbors(adata,
                    n_neighbors=15, # 考虑每个细胞附近约 15 个邻居
                    n_pcs=30, # 选择前 30 个主成分用于构建邻居图
                    use_rep="X_pca", # 使用PCA降维后的数据作为输入
                    random_state=SEED)
    # UMAP 可视化:将二维坐标保存到adata.obsm["X_umap"]
    sc.tl.umap(adata,
               min_dist=.3, # 控制细胞点的紧凑程度
               random_state=SEED)
    # Leiden 聚类
    sc.tl.leiden(adata,
                 resolution=.7, # 控制社区划分粒度
                 flavor="igraph",
                 directed=False, # 按照无向图进行社区发现
                 n_iterations=2,
                 random_state=SEED,
                 key_added="leiden") #指定聚类结果保存到:adata.obs["leiden"]
    # value_counts() 统计每个聚类包含多少细胞
    counts = adata.obs.leiden.value_counts(sort=False)
    counts.rename("cells").to_csv(output / "tables" / "cluster_sizes.csv", index_label="leiden")

    # 使用计算好的 UMAP 坐标绘图，根据 Leiden 聚类标签给细胞着色
    sc.pl.umap(adata, color="leiden", title="PBMC3k Leiden clusters", show=False)
    save_figure(output, "umap_leiden")
    fig, ax = plt.subplots(figsize=(6.5, 3.6), layout="constrained")
    ax.bar(counts.index, counts, color="#207785")
    ax.set(xlabel="Leiden cluster", ylabel="Cells", title="PBMC3k cluster sizes")
    save_figure(output, "cluster_sizes", fig)
    
    return adata
