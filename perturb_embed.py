"""Embed a preprocessed dataset using the pretrained model.

TODO: add option to indicate missing genes with <mask> or <pad> tokens?

N.B. binning happens just by converting the datatype from float to long...
"""
# -*- coding: utf-8 -*-
import argparse
import gzip
import tqdm
import pickle as pkl
from functools import reduce
import numpy as np
import pandas as pd
import pickle as pkl
import torch
from performer_pytorch import PerformerLM
import scanpy as sc
from utils import *


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    CLASS = args.bin_num + 2
    SEQ_LEN = args.gene_num + 1

    model = PerformerLM(
        num_tokens = CLASS,
        dim = 200,
        depth = 6,
        max_seq_len = SEQ_LEN,
        heads = 10,
        local_attn_heads = 0,
        g2v_position_emb = True
    )

    data = sc.read_h5ad('data/norman/ctrl_norman_preprocessed.h5ad')

    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(device)

    batch_size = data.shape[0]
    model.eval()

    with open('data/norman/all_perts.pkl', 'rb') as f:
        all_perts = pkl.load(f)

    with open('data/norman/norman_mask_df.pkl', 'rb') as f:
        norman_cl_mask = pkl.load(f)

    genes = data.var_names.tolist()

    with torch.no_grad():
        all_pert_embs = {}
        non_exp_gene = norman_cl_mask.columns[norman_cl_mask.sum() == 0]
        #remove pert gene from data
        for pert in all_perts:
            print("Processing perturbation: ", pert)
            if pert in non_exp_gene:
                pass
            else:
                cl_mask = norman_cl_mask[pert].values
                slice_adata = data[cl_mask]
                gene_index = genes.index(pert)
                batch_size = slice_adata.shape[0]
                embs = []
                for index in tqdm.tqdm(range(batch_size)):
                    # now we need to set the measurement for the perturbed gene to 0
                    full_seq = slice_adata.X[index].toarray()[0]
                    print(full_seq.shape, full_seq[gene_index], full_seq)
                    full_seq[gene_index] = 0
                    full_seq[full_seq > (CLASS - 2)] = CLASS - 2
                    full_seq = torch.from_numpy(full_seq).long()
                    full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)  # q. why add 0?
                    full_seq = full_seq.unsqueeze(0)
                    emb = model(full_seq, return_encodings=True)
                    # TODO: add to the anndata file or whatever
                    embs.append(emb.squeeze(0).cpu().numpy())
                
                if args.average_embeddings:
                    all_pert_embs[pert] = np.array(embs).mean(1)
                    print(all_pert_embs[pert].shape)
                else:
                    raise NotImplementedError("Not implemented yet")
    
    # save the embeddings to gzipped pkl files
    with gzip.open('data/norman/scbert_perturbation_embeddings.pkl.gz', 'wb') as f:
        pkl.dump(all_pert_embs, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
    parser.add_argument("--gene_num", type=int, default=16906, help='Number of genes.')
    parser.add_argument("--model_path", type=str, default='data/panglao_pretrain.pth')
    parser.add_argument("--average_embeddings", action="store_true")
    parser.add_argument("--force_cpu", action="store_true")

    args = parser.parse_args()
    main(args)
