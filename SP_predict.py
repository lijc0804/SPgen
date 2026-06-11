import os
import scanpy as sc
from scipy import sparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import math
from Bio import SeqIO
import re
from utils import (
    build_spatial_graph, GNNCell2ExprDecoder,
    get_scGPT_emb, get_ESM_emb, get_pubmedbert_emb, concate_gene_emb
)

def predict_in_batches(model, cell_input, gene_emb, edge_index, batch_size=2048):
    preds = []
    N = cell_input.size(0)
    for i in range(0, N, batch_size):
        cell_batch = cell_input[i:i+batch_size]

        with torch.no_grad():
            out = model(cell_batch, gene_emb, edge_index)  
        preds.append(out) 
        #torch.cuda.empty_cache()
    return torch.cat(preds, dim=0)

def main(args):
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")

    adata_raw = sc.read_h5ad(args.figure_path+"adata_with_predict.h5ad")
    print(adata_raw)

    adata_with_predict = adata_raw.copy()
    adata_with_predict.var_names = adata_with_predict.var['gene_name']
    adata_with_predict.X = adata_with_predict.layers['predict']


    fasta_file = "data/uniprot/uniprot_sprot.fasta"

    uniprot2gene = {}
    gene2uniprot = {}

    protein_list, gene_list = [], []
    for record in SeqIO.parse(fasta_file, "fasta"):
        desc = record.description
        
        taxid = re.search(r'OX=(\d+)', desc).group(1)
        if args.species != taxid:
            continue

        parts = record.id.split("|")
        if len(parts) < 2:
            continue
        uid = parts[1]

        gene_match = re.search(r"GN=([A-Za-z0-9_\-]+)", desc)
        if gene_match:
            gene_id = gene_match.group(1).split("-")[0].upper()
        else:
            gene_id = parts[2].upper()
        
        if uid not in uniprot2gene.keys():
            uniprot2gene[uid] = [gene_id]
            protein_list.append(uid)
            gene_list.append(gene_id)
        else:
            uniprot2gene[uid].append(gene_id)        
        if gene_id not in gene2uniprot.keys():
            gene2uniprot[gene_id] = [uid]
        else:
            gene2uniprot[gene_id].append(uid)

    print(f"UniProt ID: {len(protein_list)}")
    print(f"Gene Symbol: {len(gene_list)}")
    print(f"gene->protein pairs: {sum(len(v) for v in gene2uniprot.values())}")
    print(protein_list[:10])
    print(gene_list[:10])
    

    gene_emb_list, protein_mask_list, g_emb_dims = [], [], []

    if 'pubmedbert' in args.use_gene_emb:
        pubmedbert_ids = [line.strip() for line in open("pubmedbert/ids.txt")]
        pubmedbert_embeddings = np.load("pubmedbert/uniprot_pubmedbert_embeddings.npy")  # (N, D)
        pubmedbert_gene_df = pd.DataFrame(pubmedbert_embeddings.T, columns=pubmedbert_ids)
        gene_emb_pubmedbert, protein_mask_pubmedbert = get_pubmedbert_emb(protein_list, pubmedbert_gene_df, device)
        gene_emb_list.append(gene_emb_pubmedbert)
        protein_mask_list.append(protein_mask_pubmedbert)
        g_emb_dims.append(pubmedbert_embeddings.shape[1])

    if 'scGPT' in args.use_gene_emb:
        scGPT_gene_df = pd.read_csv('scGPT_emb/gene_all.tsv', sep='\t', index_col=0)
        gene_emb_scGPT, protein_mask_scGPT = get_scGPT_emb(protein_list, gene_list, scGPT_gene_df, device)
        gene_emb_list.append(gene_emb_scGPT)
        protein_mask_list.append(protein_mask_scGPT)
        g_emb_dims.append(gene_emb_scGPT.shape[1])

    if 'ESM' in args.use_gene_emb:
        emb = np.load("ESM_embedding/all_emb.npy")
        gene_names = pd.read_csv("ESM_embedding/gene_names.txt", header=None)[0].values
        ESM_gene_df = pd.DataFrame(emb, index=gene_names).T
        gene_emb_ESM, protein_mask_ESM = get_ESM_emb(protein_list, ESM_gene_df, device)
        gene_emb_list.append(gene_emb_ESM)
        protein_mask_list.append(protein_mask_ESM)
        g_emb_dims.append(gene_emb_ESM.shape[1])


    if args.use_gene_union:    
        gene_emb, protein_mask = concate_gene_emb(list(adata_raw.var_names), gene_emb_list, protein_mask_list, method='union')
    else:
        gene_emb, protein_mask = concate_gene_emb(list(adata_raw.var_names), gene_emb_list, protein_mask_list, method='intersection')
        gene_emb = gene_emb[protein_mask]
        protein_mask_np = protein_mask.detach().cpu().numpy()
        protein_list = [p for p, m in zip(protein_list, protein_mask_np) if m]
        gene_list = [p for p, m in zip(gene_list, protein_mask_np) if m]
    
    train_gene_idx = adata_raw.uns['train_gene_idx']
    test_gene_idx = adata_raw.uns['test_gene_idx']
    adata_train = adata_raw[:, train_gene_idx]
    cell_input = torch.Tensor(adata_raw.layers['GT'][:, train_gene_idx]).to(device)
    
    coords = torch.tensor(adata_train.obsm["spatial"], dtype=torch.float32, device=device)
    i_idx_spot, j_idx_spot, _ = build_spatial_graph(coords, K=4, device=device)
    edge_index_spot = torch.stack([i_idx_spot, j_idx_spot], dim=0)
    
    # load model
    emb2spatial_model = GNNCell2ExprDecoder(in_dim=cell_input.shape[1], g_emb_dims=g_emb_dims, emb_dim=gene_emb.shape[1], num_layers=4).to(device)

    emb2spatial_model.load_state_dict(torch.load(args.figure_path+"model_weights.pth", map_location=device))
    emb2spatial_model.eval()  
    if args.batch_size>0:
        G = gene_emb.shape[0]
        preds = []
        with torch.no_grad():
            for i in range(0, G, args.batch_size):
                j = min(i + args.batch_size, G)
                gene_emb_batch = gene_emb[i:j].to(device)
                # forward
                pred_batch = emb2spatial_model(cell_input, gene_emb_batch, edge_index_spot)
                preds.append(pred_batch.detach().cpu())
        predicted = torch.cat(preds, dim=1)   # (n_spots, G)
                
    else:
        with torch.no_grad():
            predicted = emb2spatial_model(cell_input, gene_emb, edge_index_spot)

    print(predicted.shape)

    adata_predict = sc.AnnData(X=predicted.detach().cpu().numpy())
    adata_predict.obsm["spatial"] = adata_raw.obsm["spatial"]
    adata_predict.var_names = protein_list 
    adata_predict.var['gene_name'] = gene_list
    adata_predict.var['uniprot_id'] = protein_list

    if 'spatial' in adata_raw.uns.keys():
        adata_predict.uns['spatial'] = adata_raw.uns['spatial']
        
    import squidpy as sq
    sq.gr.spatial_neighbors(adata_predict)
    sq.gr.spatial_autocorr(adata_predict, mode="moran", n_perms=1000)
    adata_predict.var['moranI'] = adata_predict.uns["moranI"]['I'][adata_predict.var_names]

    if args.use_gene_union:
        save_path = args.figure_path+"adata_predict_union.h5ad"
    else:
        save_path = args.figure_path+"adata_predict_intersection.h5ad"

    adata_predict.write(save_path)
    print('Saved to ' + save_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument( '--dataset_name', type=str, default="Cerebellum-PLATO", help='mouse_brain_coronal Intestinal-villi Cerebellum-PLATO') 
    parser.add_argument( '--device_id', type=int, default=1, help='device_id')
    parser.add_argument( '--species', type=str, default='10090', help='Mus musculus: 10090')
    parser.add_argument( '--use_gene_emb', type=str, default='scGPT+ESM+pubmedbert', help='scGPT+ESM+pubmedbert')
    parser.add_argument( '--use_gene_union', type=int, default=1, help='use_gene_union')
    parser.add_argument( '--batch_size', type=int, default=4096, help='batch_size')
    parser.add_argument( '--save_name', type=str, default='_demo_', help='_')
    args = parser.parse_args() 
    
    args.figure_path = "figures_" + args.dataset_name + args.save_name + args.use_gene_emb + '/'

    os.makedirs(args.figure_path, exist_ok=True)

    print('********************************************************************************************************')
    print(args)
    
    main(args)
    print('Finished on ', args.dataset_name, args.use_gene_emb)