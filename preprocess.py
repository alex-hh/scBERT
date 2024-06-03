import argparse
import scanpy as sc, numpy as np, pandas as pd, anndata as ad
from scipy import sparse


def main(args):
    panglao = sc.read_h5ad('./data/panglao_10000.h5ad')
    data = sc.read_h5ad(args.input_file)
    counts = sparse.lil_matrix((data.X.shape[0],panglao.X.shape[1]),dtype=np.float32)
    ref = panglao.var_names.tolist()
    obj = data.var_names.tolist()

    total_genes_found = 0
    for i in range(len(ref)):
        if ref[i] in obj:
            loc = obj.index(ref[i])
            counts[:,i] = data.X[:,loc]
            total_genes_found += 1
        else:
            print("Gene not found in data: ", ref[i], " at index: ", i)
            print("Counts for missing gene", counts[: , i])

    counts = counts.tocsr()
    new = ad.AnnData(X=counts)
    new.var_names = ref
    new.obs_names = data.obs_names
    new.obs = data.obs
    new.uns = panglao.uns

    sc.pp.filter_cells(new, min_genes=200)
    sc.pp.normalize_total(new, target_sum=1e4)
    sc.pp.log1p(new, base=2)
    new.write(args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    args = parser.parse_args()
    main(args)
