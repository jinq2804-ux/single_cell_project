import scanpy as sc

# 1. 加载 PBMC3k 单细胞数据集
adata = sc.datasets.pbmc3k()

# 2. 查看数据基本信息
print("数据基本信息：")
print(adata)

# 3. 查看表达矩阵形状
print("表达矩阵形状：", adata.X.shape)

# 4. 查看细胞信息
print("细胞信息：")
print(adata.obs.head())

# 5. 查看基因信息
print("基因信息：")
print(adata.var.head())

# 查看表达矩阵的数据类型
print("表达矩阵类型：", type(adata.X))

# 查看第一个细胞的表达数据
cell = adata.X[0].toarray().flatten()

# 找到表达量不为 0 的基因
import numpy as np

nonzero_indices = np.where(cell > 0)[0]

print("第一个细胞中有表达计数的基因数量：",
      len(nonzero_indices))

# 打印前10个有表达计数的基因
print("前10个非零表达基因：")

for i in nonzero_indices[:10]:
    print(
        adata.var_names[i],
        cell[i]
    )