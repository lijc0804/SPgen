import os
import scanpy as sc
from scipy import sparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import random
import squidpy as sq
import time

from utils import (
    analyze_results, analyze_distribution, build_spatial_graph,
    GNNCell2ExprDecoder, get_recon_loss, get_recon_loss_weighted,
    get_scGPT_emb, get_ESM_emb, get_pubmedbert_emb, concate_gene_emb,
)



def load_data(args):
    if args.dataset_name == 'mouse_brain_coronal':
        data_path = "data/mouse_brain_coronal/intersected_spatial_proteomics.h5ad"
    elif args.dataset_name == 'Intestinal-villi':
        data_path = "data/Flow2Spatial/Intestinal-villi.h5ad"
    elif args.dataset_name == 'Cerebellum-PLATO':
        data_path = "data/Flow2Spatial/Cerebellum-PLATO.h5ad"

    adata_raw = sc.read_h5ad(data_path)
    print(adata_raw)
    
    if 'cluster' in adata_raw.obs.keys():
        ax = sc.pl.spatial(
            adata_raw,
            color="cluster",
            spot_size=args.base_spot_size,
            cmap="tab20",
            show=False  
        )
        plt.savefig(args.figure_path+"cluster_spatial.png", dpi=300, bbox_inches="tight")
        plt.close() 

    if sparse.issparse(adata_raw.X):
        adata_raw.X = adata_raw.X.toarray()

    if args.dataset_name == 'mouse_brain_coronal':
        adata_raw.var['gene_name'] = adata_raw.var['gene'].str.upper()
    elif args.dataset_name == 'Intestinal-villi':
        adata_raw.var['gene_name'] = adata_raw.var_names.str.upper()
        adata_raw.var_names = adata_raw.var['PG']
    elif args.dataset_name == 'Cerebellum-PLATO':
        adata_raw.var['gene_name'] = adata_raw.var_names.str.upper()
        adata_raw.var_names = adata_raw.var['PG']
        
    analyze_distribution(adata_raw.X, save_path=args.figure_path)    

    return adata_raw


def train_model(args, adata_raw, split_rate=0.9):
    np.random.seed(14)
    n_genes = adata_raw.shape[1]
    all_gene_indices = np.arange(n_genes)
    np.random.shuffle(all_gene_indices)
    split_idx = int(split_rate * n_genes)
    train_gene_idx = all_gene_indices[:split_idx]
    test_gene_idx = all_gene_indices[split_idx:]
    print('test_gene_idx:', test_gene_idx)

    adata_train = adata_raw[:, train_gene_idx]
    adata_test = adata_raw[:, test_gene_idx]
    print(adata_train)
    print(adata_test)

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")

    gene_emb_train_list, gene_emb_test_list, protein_mask_train_list, protein_mask_test_list, g_emb_dims = [], [], [], [], []

    if 'pubmedbert' in args.use_gene_emb:
        pubmedbert_ids = [line.strip() for line in open("pubmedbert/ids.txt")]
        pubmedbert_embeddings = np.load("pubmedbert/uniprot_pubmedbert_embeddings.npy")  # (N, D)
        pubmedbert_gene_df = pd.DataFrame(pubmedbert_embeddings.T, columns=pubmedbert_ids)
        gene_emb_pubmedbert_train, protein_mask_pubmedbert_train = get_pubmedbert_emb(list(adata_train.var_names), pubmedbert_gene_df, device)
        gene_emb_pubmedbert_test, protein_mask_pubmedbert_test = get_pubmedbert_emb(list(adata_test.var_names), pubmedbert_gene_df, device)
        gene_emb_train_list.append(gene_emb_pubmedbert_train)
        gene_emb_test_list.append(gene_emb_pubmedbert_test)
        protein_mask_train_list.append(protein_mask_pubmedbert_train)
        protein_mask_test_list.append(protein_mask_pubmedbert_test)
        g_emb_dims.append(pubmedbert_embeddings.shape[1])

    if 'scGPT' in args.use_gene_emb:
        scGPT_gene_df = pd.read_csv('scGPT_emb/gene_all.tsv', sep='\t', index_col=0)
        gene_emb_scGPT_train, protein_mask_scGPT_train = get_scGPT_emb(list(adata_train.var_names), list(adata_train.var['gene_name']), scGPT_gene_df, device)
        gene_emb_scGPT_test, protein_mask_scGPT_test = get_scGPT_emb(list(adata_test.var_names), list(adata_test.var['gene_name']), scGPT_gene_df, device)
        del scGPT_gene_df
        gene_emb_train_list.append(gene_emb_scGPT_train)
        gene_emb_test_list.append(gene_emb_scGPT_test)
        protein_mask_train_list.append(protein_mask_scGPT_train)
        protein_mask_test_list.append(protein_mask_scGPT_test)
        g_emb_dims.append(gene_emb_scGPT_train.shape[1])

    if 'ESM' in args.use_gene_emb:
        emb = np.load("ESM_embedding/all_emb.npy")
        gene_names = pd.read_csv("ESM_embedding/gene_names.txt", header=None)[0].values
        ESM_gene_df = pd.DataFrame(emb, index=gene_names).T
        gene_emb_ESM_train, protein_mask_ESM_train = get_ESM_emb(list(adata_train.var_names), ESM_gene_df, device)
        gene_emb_ESM_test, protein_mask_ESM_test = get_ESM_emb(list(adata_test.var_names), ESM_gene_df, device)
        del ESM_gene_df
        gene_emb_train_list.append(gene_emb_ESM_train)
        gene_emb_test_list.append(gene_emb_ESM_test)
        protein_mask_train_list.append(protein_mask_ESM_train)
        protein_mask_test_list.append(protein_mask_ESM_test)
        g_emb_dims.append(gene_emb_ESM_train.shape[1])
        
    gene_emb_train, protein_mask_train = concate_gene_emb(list(adata_train.var_names), gene_emb_train_list, protein_mask_train_list)
    gene_emb_test, protein_mask_test = concate_gene_emb(list(adata_test.var_names), gene_emb_test_list, protein_mask_test_list)

    adata_train_X = torch.tensor(adata_train.X, dtype=torch.float32, device=device)
    if args.dataset_name == 'Cerebellum-PLATO':
        adata_train_X_norm = adata_train_X
    else:
        adata_train_X_norm = torch.log1p(adata_train_X)
    adata_train_X_norm = (adata_train_X_norm - adata_train_X_norm.mean(dim=0, keepdim=True)) / (adata_train_X_norm.std(dim=0, keepdim=True) + 1e-4)
    adata_test_X = torch.tensor(adata_test.X, dtype=torch.float32, device=device)
    if args.dataset_name == 'Cerebellum-PLATO':
        adata_test_X_norm = adata_test_X
    else:
        adata_test_X_norm = torch.log1p(adata_test_X)
    adata_test_X_norm = (adata_test_X_norm - adata_test_X_norm.mean(dim=0, keepdim=True)) / (adata_test_X_norm.std(dim=0, keepdim=True) + 1e-4)
    adata_test.X = adata_test_X_norm.detach().cpu().numpy()

    coords = torch.tensor(adata_train.obsm["spatial"], dtype=torch.float32, device=device)
    i_idx_spot, j_idx_spot, _ = build_spatial_graph(coords, K=4, device=device)
    edge_index_spot = torch.stack([i_idx_spot, j_idx_spot], dim=0)

    cell_input = adata_train_X_norm


    sq.gr.spatial_neighbors(adata_train)
    sq.gr.spatial_autocorr(adata_train, mode="moran", n_perms=1000)
    I_train = torch.tensor(adata_train.uns["moranI"]['I'][adata_train.var_names]).to(device)
    sq.gr.spatial_neighbors(adata_test)
    sq.gr.spatial_autocorr(adata_test, mode="moran", n_perms=1000)
    I_test = torch.tensor(adata_test.uns["moranI"]['I'][adata_test.var_names]).to(device)

    # === train ===
    emb2spatial_model = GNNCell2ExprDecoder(in_dim=cell_input.shape[1], g_emb_dims=g_emb_dims, emb_dim=gene_emb_train.shape[1], num_layers=4).to(device)
    optimizer = torch.optim.Adam(emb2spatial_model.parameters(), lr=1e-3)

    all_losses = []
    start = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    
    for epoch in range(args.train_epoch):    
        emb2spatial_model.train()
        optimizer.zero_grad()
        pred_expr = emb2spatial_model(cell_input, gene_emb_train, edge_index_spot)
        if 'weighted' in args.loss_type:
            train_loss_recon_mae = get_recon_loss_weighted(adata_train_X_norm, pred_expr, protein_mask_train, loss_type='mae', weight=abs(I_train))
            train_loss_recon_mse = get_recon_loss_weighted(adata_train_X_norm, pred_expr, protein_mask_train, loss_type='mse', weight=abs(I_train))        
        else:
            train_loss_recon_mae = get_recon_loss(adata_train_X_norm, pred_expr, protein_mask_train, loss_type='mae')
            train_loss_recon_mse = get_recon_loss(adata_train_X_norm, pred_expr, protein_mask_train, loss_type='mse')    
        
        if ('mae' in args.loss_type) and ('mse' in args.loss_type):
            train_loss = train_loss_recon_mae + train_loss_recon_mse  
        elif 'mae' in args.loss_type:
            train_loss = train_loss_recon_mae 
        elif 'mse' in args.loss_type:
            train_loss = train_loss_recon_mse  
        train_loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            emb2spatial_model.eval()
            with torch.no_grad():
                pred_expr_test = emb2spatial_model(cell_input, gene_emb_test, edge_index_spot)
                if 'weighted' in args.loss_type:
                    test_loss_recon_mae = get_recon_loss_weighted(adata_test_X_norm, pred_expr_test, protein_mask_test, loss_type='mae', weight=abs(I_test))
                    test_loss_recon_mse = get_recon_loss_weighted(adata_test_X_norm, pred_expr_test, protein_mask_test, loss_type='mse', weight=abs(I_test))
                else:
                    test_loss_recon_mae = get_recon_loss(adata_test_X_norm, pred_expr_test, protein_mask_test, loss_type='mae')
                    test_loss_recon_mse = get_recon_loss(adata_test_X_norm, pred_expr_test, protein_mask_test, loss_type='mse')
                
                if ('mae' in args.loss_type) and ('mse' in args.loss_type):
                    test_loss = test_loss_recon_mae + test_loss_recon_mse 
                elif 'mae' in args.loss_type:
                    test_loss = test_loss_recon_mae 
                elif 'mse' in args.loss_type:
                    test_loss = test_loss_recon_mse  
        
            all_losses.append((epoch, train_loss.item(), train_loss_recon_mae.item(),
                            test_loss.item(), test_loss_recon_mae.item()))
            print(f"Epoch {epoch}: train loss = {train_loss.item():.4f}, test_loss = {test_loss.item():.4f}")

            
    end = time.time()
    print(f"Training time: {end - start:.2f} seconds")
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) / 1024**2
    print(f"Peak GPU memory: {peak:.1f} MB")
    
    plt.figure(figsize=(7, 5))
    epochs_i, train_losses, train_losses_recon, test_losses, test_losses_recon = zip(*all_losses)
    plt.plot(epochs_i[1:], train_losses[1:], label='Train Loss', color='blue')
    plt.plot(epochs_i[1:], test_losses[1:], label='Test Loss', color='red', marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss over Epochs")
    plt.legend()
    plt.grid(color='lightgray', linestyle='--', linewidth=0.5)  # 添加网格线
    plt.tight_layout()
    plt.savefig(args.figure_path+"loss_curve.png", dpi=150)
    plt.close()
    
    torch.save(emb2spatial_model.state_dict(), args.figure_path+"model_weights.pth")

    
    emb2spatial_model.eval()
    with torch.no_grad():
        pred_expr = emb2spatial_model(cell_input, gene_emb_train, edge_index_spot)
        pred_expr_test = emb2spatial_model(cell_input, gene_emb_test, edge_index_spot)
                
    return adata_raw, adata_test, train_gene_idx, test_gene_idx, adata_train_X_norm, adata_test_X_norm, pred_expr, pred_expr_test, protein_mask_test, gene_emb_train, gene_emb_test


def save(args, adata_raw, train_gene_idx, test_gene_idx, adata_train_X_norm, adata_test_X_norm, pred_expr, pred_expr_test, gene_emb_train, gene_emb_test):
    adata_raw.layers['GT'] = np.zeros_like(adata_raw.X)
    adata_raw.layers['predict'] = np.zeros_like(adata_raw.X)
    adata_raw.uns['train_gene_idx'] = train_gene_idx
    adata_raw.uns['test_gene_idx'] = test_gene_idx
    adata_raw.varm['gene_emb'] = np.zeros([adata_raw.shape[1], gene_emb_train.shape[1]])
    adata_raw.layers['GT'][:, train_gene_idx] = adata_train_X_norm.detach().cpu().numpy()
    adata_raw.layers['GT'][:, test_gene_idx] = adata_test_X_norm.detach().cpu().numpy()
    adata_raw.layers['predict'][:, train_gene_idx] = pred_expr.detach().cpu().numpy()
    adata_raw.layers['predict'][:, test_gene_idx] = pred_expr_test.detach().cpu().numpy()
    adata_raw.varm['gene_emb'][train_gene_idx] = gene_emb_train.detach().cpu().numpy()
    adata_raw.varm['gene_emb'][test_gene_idx] = gene_emb_test.detach().cpu().numpy()

    adata_raw.write(args.figure_path+"adata_with_predict.h5ad")
    return 

def load_results(args):
    adata = sc.read_h5ad(args.figure_path+"adata_with_predict.h5ad")
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")

    train_gene_idx = adata.uns['train_gene_idx']
    test_gene_idx = adata.uns['test_gene_idx']
    adata_train = adata[:, train_gene_idx]
    adata_test = adata[:, test_gene_idx]
    sq.gr.spatial_neighbors(adata_test)
    sq.gr.spatial_autocorr(adata_test, mode="moran", n_perms=1000)
    adata_test_X_norm = torch.Tensor(adata.layers['GT'][:, test_gene_idx]).to(device)
    pred_expr_test = torch.Tensor(adata.layers['predict'][:, test_gene_idx]).to(device)
    protein_mask_test = torch.ones(len(adata_test.var_names), dtype=torch.bool).to(device)
    return adata, adata_test, adata_test_X_norm, pred_expr_test, protein_mask_test

def main(args):
    adata_raw = load_data(args)
    adata_raw, adata_test, train_gene_idx, test_gene_idx, adata_train_X_norm, adata_test_X_norm, pred_expr, pred_expr_test, protein_mask_test, gene_emb_train, gene_emb_test = train_model(args, adata_raw)
    save(args, adata_raw, train_gene_idx, test_gene_idx, adata_train_X_norm, adata_test_X_norm, pred_expr, pred_expr_test, gene_emb_train, gene_emb_test) 
    #adata_raw, adata_test, adata_test_X_norm, pred_expr_test, protein_mask_test = load_results(args)
    analyze_results(args, adata_raw, adata_test, adata_test_X_norm, pred_expr_test, protein_mask_test)
    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument( '--dataset_name', type=str, default="Cerebellum-PLATO", help='mouse_brain_coronal Intestinal-villi Cerebellum-PLATO') 
    parser.add_argument( '--device_id', type=int, default=0, help='device_id')
    parser.add_argument( '--use_gene_emb', type=str, default='scGPT+ESM+pubmedbert', help='use_gene_emb')
    parser.add_argument( '--loss_type', type=str, default='weighted_mae+mse', help='weighted_mae+mse')
    parser.add_argument( '--train_epoch', type=int, default=8000, help='train_epoch')
    parser.add_argument( '--save_name', type=str, default='_demo_', help='_')
    args = parser.parse_args() 
    
    args.figure_path = "figures_" + args.dataset_name + args.save_name + args.use_gene_emb + '/'

    os.makedirs(args.figure_path, exist_ok=True)
    
    if args.dataset_name == 'Cerebellum-PLATO':
        args.base_spot_size = 40
    else:
        args.base_spot_size = 1

    print('********************************************************************************************************')
    print(args)
    
    main(args)
    print('Finished on ', args.dataset_name, args.use_gene_emb)