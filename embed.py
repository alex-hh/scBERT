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

    data = sc.read_h5ad("data/norman/ctrl_norman_preprocessed.h5ad")
    data = data.X

    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(device)

    batch_size = data.shape[0]
    model.eval()
    embs = []

    if args.average_embeddings:
        raise NotImplementedError("Averaging embeddings not implemented yet.")

    with torch.no_grad():
        for index in tqdm.tqdm(range(batch_size)):
            full_seq = data[index].toarray()[0]
            full_seq[full_seq > (CLASS - 2)] = CLASS - 2
            full_seq = torch.from_numpy(full_seq).long()
            full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)  # q. why add 0?
            full_seq = full_seq.unsqueeze(0)
            emb = model(full_seq, return_encodings=True)
            # TODO: add to the anndata file or whatever
            embs.append(emb.cpu().numpy())

    embs = np.array(embs)
    print("Shape of embeddings: ", embs.shape)
    with gzip.open('data/norman/scbert_control_embeddings.pkl.gz', 'wb') as f:
        pkl.dump(embs, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
    parser.add_argument("--gene_num", type=int, default=16906, help='Number of genes.')
    parser.add_argument("--model_path", type=str, default='data/panglao_pretrain.pth')
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--average_embeddings", action="store_true")

    args = parser.parse_args()
    main(args)
