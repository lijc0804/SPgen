import numpy as np
import scanpy as sc
import scipy.sparse
import matplotlib.pyplot as plt
import networkx as nx
import torch
import scanpy as sc
import scipy.sparse
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, DeepGraphInfomax
from torch_geometric.utils import softmax, degree
import sklearn
import copy
import squidpy as sq
import torch.nn as nn
from torch_scatter import scatter, scatter_mean
import umap
from mygene import MyGeneInfo
import pandas as pd
import math
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import matplotlib.cm as cm
import os
import traceback
from scipy.stats import gaussian_kde
import re
import random
from sklearn.metrics.pairwise import cosine_similarity
import anndata as ad
import shutil
from Bio import SeqIO

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    


def prot2gene_map(fasta_file="data/uniprot/uniprot_sprot.fasta", species='10090'):
    uniprot2gene = {}
    gene2uniprot = {}

    protein_list = []
    gene_list = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        desc = record.description

        # ===================== 1. species =====================
        taxid = re.search(r'OX=(\d+)', desc).group(1)
        if species != taxid:
            continue

        # 2. UniProt accession
        # sp|P12345|GENE_MOUSE ...
        parts = record.id.split("|")
        if len(parts) < 2:
            continue
        uid = parts[1]

        # 3. gene symbol (GN=)
        gene_match = re.search(r"GN=([A-Za-z0-9_\-]+)", desc)

        if gene_match:
            gene_id = gene_match.group(1).split("-")[0].upper()
        else:
            # fallback: entry name
            gene_id = parts[2].upper()


        # 4.  isoform
        gene_id = gene_id.split("-")[0]
        gene_id = gene_id.upper()
        
        # 5.  gene -> protein
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

    print(f"Species filter: {species}")
    print(f"UniProt IDs: {len(protein_list)}")
    print(f"Gene symbols: {len(gene_list)}")
    print(f"gene->protein pairs: {sum(len(v) for v in gene2uniprot.values())}")

    df_protein_gene = pd.DataFrame({
        "Protein_ID": protein_list,
        "Gene_Name": gene_list
    })
    #df_protein_gene.to_csv(save_file, index=False)
    return df_protein_gene

    
def prot2gene(uniprot_ids, species='10090', to_upper=True, return_df=False):
    df_protein_gene = prot2gene_map(species=species)
        
    mapping = dict(zip(
        df_protein_gene['Protein_ID'],
        df_protein_gene['Gene_Name']
    ))

    mapped = {}

    for uid in uniprot_ids:
        gene = mapping.get(uid, None)

        if gene is not None and to_upper:
            gene = gene.upper()

        mapped[uid] = gene

    if return_df:
        return pd.DataFrame({
            'UniProt_ID': list(mapped.keys()),
            'Gene_Symbol': list(mapped.values())
        })

    return mapped



def analyze_distribution(X_dense, save_path):
    data = np.ravel(X_dense)
    data = data[~np.isnan(data)]

    log_data = np.log1p(data)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].hist(data, bins=100, color='skyblue', edgecolor='black')
    axs[0].set_title("Original Distribution")
    axs[0].set_xlabel("Expression Value")
    axs[0].set_ylabel("Frequency")
    axs[0].set_yscale("log")  # 可选
    axs[1].hist(log_data, bins=100, color='salmon', edgecolor='black')
    axs[1].set_title("Log1p Transformed Distribution")
    axs[1].set_xlabel("log(1 + Expression Value)")
    axs[1].set_ylabel("Frequency")
    axs[1].set_yscale("log")  # 可选
    plt.tight_layout()
    plt.savefig(save_path+'/analyze_distribution.png')
    plt.close
    return

  
def build_spatial_graph(coords: torch.Tensor, K=9, device='cpu'):

    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    coords_np = coords.cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=K + 1, algorithm='ball_tree').fit(coords_np)
    distances, indices = nbrs.kneighbors(coords_np)  # [N, K+1]

    i_all, j_all, d_all = [], [], []
    N = coords.shape[0]
    for i in range(N):
        min_dist = distances[i][1]  
        threshold = min_dist * 1.5
        for j_idx, dist in zip(indices[i][1:], distances[i][1:]):  
            if dist <= threshold:
                i_all.append(i)
                j_all.append(j_idx)
                d_all.append(dist)

    return (torch.tensor(i_all, device=device),
            torch.tensor(j_all, device=device),
            torch.tensor(d_all, dtype=torch.float32, device=device))


import matplotlib.pyplot as plt
import networkx as nx

def plot_spatial_graph(coords: torch.Tensor, i_idx: torch.Tensor, j_idx: torch.Tensor, sample_size=None):

    if sample_size is not None:
        mask = (i_idx < sample_size) & (j_idx < sample_size)
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]
        coords = coords[:sample_size].cpu().numpy()
    else:
        coords = coords.cpu().numpy()

    G = nx.Graph()
    for i in range(coords.shape[0]):
        G.add_node(i, pos=tuple(coords[i]))

    for i, j in zip(i_idx.tolist(), j_idx.tolist()):
        G.add_edge(i, j)

    pos = nx.get_node_attributes(G, 'pos')
    plt.figure()
    nx.draw(G, pos, node_size=5, edge_color='gray', alpha=0.4, with_labels=False)
    plt.title("Spatial Graph")
    plt.axis('equal')
    plt.show()
    
    

def get_recon_loss(x, x_pred, protein_mask_loss=None, loss_type='mse'):
    if protein_mask_loss is not None:
        masked_x = x[:, protein_mask_loss]
        masked_pred = x_pred[:, protein_mask_loss]
    else:
        masked_x = x 
        masked_pred = x_pred
        
    if loss_type == 'mse':
        recon_loss = F.mse_loss(masked_pred, masked_x)
    elif loss_type == 'smooth_l1':
        recon_loss = F.smooth_l1_loss(masked_pred, masked_x)
    elif loss_type == 'mae':
        recon_loss = F.l1_loss(masked_pred, masked_x)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    return recon_loss

  
def get_recon_loss_weighted(x, x_pred, protein_mask_loss=None, loss_type='mse', weight=None):
    if protein_mask_loss is not None:
        masked_w = weight[protein_mask_loss]
        masked_x = x[:, protein_mask_loss]
        masked_pred = x_pred[:, protein_mask_loss]
    else:
        masked_w = weight
        masked_x = x 
        masked_pred = x_pred
    masked_w = masked_w/masked_w.mean()
    
    if loss_type == 'mse':
        loss_per_elem = F.mse_loss(masked_pred, masked_x, reduction='none')
        recon_loss = (loss_per_elem * masked_w).mean() 
    elif loss_type == 'smooth_l1':
        loss_per_elem = F.smooth_l1_loss(masked_pred, masked_x, reduction='none')
        recon_loss = (loss_per_elem * masked_w).mean()
    elif loss_type == 'mae':
        #recon_loss = F.l1_loss(masked_pred, masked_x)
        loss_per_elem = F.l1_loss(masked_pred, masked_x, reduction='none')
        recon_loss = (loss_per_elem * masked_w).mean()  
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    return recon_loss



def find_SVG(adata):
    sq.gr.spatial_neighbors(adata)

    sq.gr.spatial_autocorr(
        adata,
        mode="moran",        
        genes=None,          
        n_perms=1000        
    )

    moran_df = adata.uns["moranI"]
    #print(moran_df.head())
    return moran_df



def get_scGPT_emb(protein_list, gene_list, scGPT_gene_df, device):
    gene_embeddings_aligned = np.zeros((len(protein_list), scGPT_gene_df.shape[0]), dtype=np.float32)
    for i, g in enumerate(gene_list):
        if isinstance(g, str) and ";" in g:
            g = g.split(';')[0]
        if g in scGPT_gene_df.keys():
            gene_embeddings_aligned[i] = scGPT_gene_df[g]
        else:
            gene_embeddings_aligned[i] = 0         
    scGPT_gene_emb = torch.from_numpy(gene_embeddings_aligned).float().to(device)
    protein_mask = (scGPT_gene_emb.abs().sum(dim=1) != 0)
    masked_out_ids = [pid for pid, keep in zip(protein_list, protein_mask) if not keep]
    print(f"scGPT mask protein ID: {len(masked_out_ids), masked_out_ids[:10]}")
    return scGPT_gene_emb, protein_mask


def get_ESM_emb(protein_list, ESM_gene_df, device): # adata.var_names is uniprot protein id
    ESM_gene_df_sub = ESM_gene_df.reindex(columns=protein_list, fill_value=0)
    ESM_gene_emb = ESM_gene_df_sub.to_numpy().T
    ESM_gene_emb = torch.from_numpy(ESM_gene_emb).float().to(device)
    protein_mask_ESM = (ESM_gene_emb.abs().sum(dim=1) != 0)  # True 表示保留的蛋白
    masked_out_ids = [pid for pid, keep in zip(list(protein_list), protein_mask_ESM) if not keep]
    print(f"ESM mask protein ID: {len(masked_out_ids), masked_out_ids[:10]}")
    return ESM_gene_emb, protein_mask_ESM

def get_pubmedbert_emb(protein_list, gene_df, device):
    gene_embeddings_aligned = np.zeros((len(protein_list), gene_df.shape[0]), dtype=np.float32)
    for i, g in enumerate(protein_list):
        if isinstance(g, str) and ";" in g:
            g = g.split(';')[0]
        if g in gene_df.keys():
            gene_embeddings_aligned[i] = gene_df[g]
        else:
            gene_embeddings_aligned[i] = 0         
    pubmedbert_gene_emb = torch.from_numpy(gene_embeddings_aligned).float().to(device)
    protein_mask = (pubmedbert_gene_emb.abs().sum(dim=1) != 0)
    masked_out_ids = [pid for pid, keep in zip(protein_list, protein_mask) if not keep]
    print(f"pubmedbert mask protein ID: {len(masked_out_ids), masked_out_ids[:10]}")
    return pubmedbert_gene_emb, protein_mask


def concate_gene_emb(protein_list, emb_list, mask_list, method='union'):
    if (len(emb_list) == 1) and (len(mask_list) == 1):
        return emb_list[0], mask_list[0]
    else:
        gene_emb = torch.cat(emb_list, dim=1) 
        if method=='union':
            protein_mask = torch.stack(mask_list, dim=0).any(dim=0)  # 改 here！
        elif method=='intersection':
            protein_mask = torch.stack(mask_list, dim=0).all(dim=0) # protein_mask = mask_list[0] & mask_list[1] & ...
        masked_out_ids = [pid for pid, keep in zip(protein_list, protein_mask) if not keep]
        print(f"Mask protein ID: {len(masked_out_ids), masked_out_ids[:10]}")
        return gene_emb, protein_mask
    
def GO_KEGG(gene_list, background=None, figure_path='figures/', save_name='', organism='mouse'):
    import gseapy as gp
    #print(len(gene_list), len(background))
    if organism=='mouse':
        gene_sets_KEGG='KEGG_2019_Mouse'
    elif organism=='human':
        gene_sets_KEGG='KEGG_2019_Human'
    # GO
    try:
        result_go = gp.enrichr(gene_list=gene_list, 
                                organism=organism, 
                                gene_sets='GO_Biological_Process_2021', 
                                background=background,
                                # description='test', 
                                outdir=figure_path+'Enrichr_'+save_name)
        GO_files = [
            os.path.join(figure_path+'Enrichr_'+save_name, f)
            for f in os.listdir(figure_path+'Enrichr_'+save_name)
            if f.startswith("GO_Biological_Process") and f.endswith(".enrichr.reports.txt")
        ]
        if GO_files:
            for GO_file in GO_files:
                plot_GO_bar(
                    txt_file=GO_file,
                    top_n=8,
                    figsize=(13, 5),
                    base_cmap="Blues",
                    save_path=GO_file.replace('reports.txt', 'png')
                )
    except Exception as e:
        print(f"Error in GO: {e}")
        traceback.print_exc()  
    '''
    # KEGG
    try:
        result_kegg = gp.enrichr(gene_list=gene_list, 
                            organism=organism, 
                            gene_sets=gene_sets_KEGG, 
                            background=background,
                            #   description='test', 
                            outdir=figure_path+'Enrichr_'+save_name)
    except Exception as e:
        print(f"Error in KEGG: {e}")
        traceback.print_exc()  
    '''
    #result_go.res2d
    #result_kegg.res2d
    return


def kde_mode(x, grid_size=1000):
    """
    x: 1D numpy array
    """
    x = np.asarray(x)
    x = x[np.isfinite(x)]  # 去掉 nan / inf

    kde = gaussian_kde(x)

    x_grid = np.linspace(x.min(), x.max(), grid_size)
    density = kde(x_grid)

    mode = x_grid[np.argmax(density)]
    return mode

def analyze_recon_loss(x, x_pred, figure_path, protein_mask_loss=None, loss_type='spearman', plot_hist=True, figure_name=''):
    if protein_mask_loss is not None:
        masked_x = x[:, protein_mask_loss]
        masked_pred = x_pred[:, protein_mask_loss]
    else:
        masked_x = x
        masked_pred = x_pred

    masked_x_np = masked_x.detach().cpu().numpy()
    masked_pred_np = masked_pred.detach().cpu().numpy()
    scores = []
    for i in range(masked_x_np.shape[1]):
        x_col = masked_x_np[:, i]
        pred_col = masked_pred_np[:, i]
        if np.std(x_col) == 0 or np.std(pred_col) == 0:
            scores.append(np.nan)
        else:
            if loss_type == 'spearman':
                r, _ = spearmanr(x_col, pred_col)
            elif loss_type == 'pearson':
                r = np.corrcoef(x_col, pred_col)[0, 1]
            else:
                raise ValueError(f"Unsupported loss_type: {loss_type}")
            scores.append(r)

    scores = np.array(scores)
    
    if plot_hist:
        sns.set(style="whitegrid", context="talk")  
        plt.figure(figsize=(4, 6))  
        valid_scores = scores[~np.isnan(scores)]

        sns.violinplot(
            y=valid_scores,
            color="#9ecae1",
            width=0.35,
            linewidth=1.5
        )

        sns.stripplot(
            y=valid_scores,
            color="black",
            size=4,
            jitter=0.18,
            alpha=0.6
        )

        mean_val = np.mean(valid_scores)
        median_val = np.median(valid_scores)
        mode_val = kde_mode(valid_scores)

        plt.axhline(mode_val, color="blue", linestyle="-.", linewidth=3, label=f"Mode = {mode_val:.2f}")
        plt.axhline(median_val, color="green", linestyle="-.", linewidth=3, label=f"Median = {median_val:.2f}")

        plt.title(
            f"Distribution of {loss_type.capitalize()} Correlation",
            fontsize=16,
            pad=12
        )
        plt.ylabel(
            f"{loss_type.capitalize()} Correlation",
            fontsize=14
        )
        plt.xlabel("")

        plt.tick_params(axis='y', labelsize=12)

        plt.legend(fontsize=16, frameon=False, loc="lower right")
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(
            figure_path + "score_distribution_" + loss_type + figure_name + ".png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()
        
        
        sns.set(style="whitegrid", context="talk")  
        plt.figure(figsize=(6, 4))  

        valid_scores = scores[~np.isnan(scores)]

        sns.violinplot(
            x=valid_scores,
            color="#9ecae1",
            width=0.35,
            linewidth=1.5
        )

        sns.stripplot(
            x=valid_scores,
            color="black",
            size=4,
            jitter=0.18,
            alpha=0.6
        )

        mean_val = np.mean(valid_scores)
        median_val = np.median(valid_scores)
        mode_val = kde_mode(valid_scores)

        plt.axvline(mode_val, color="blue", linestyle="-.", linewidth=3, label=f"Mode = {mode_val:.2f}")
        plt.axvline(median_val, color="green", linestyle="-.", linewidth=3, label=f"Median = {median_val:.2f}")

        plt.title(
            f"Distribution of {loss_type.capitalize()} Correlation",
            fontsize=16,
            pad=12
        )
        plt.xlabel(
            f"{loss_type.capitalize()} Correlation",
            fontsize=14
        )
        plt.ylabel("")

        plt.tick_params(axis='x', labelsize=12)

        plt.legend(fontsize=16, frameon=False, loc="upper left")
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(
            figure_path + "score_distribution_" + loss_type + figure_name + "_T.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()
    return scores



def analyze_results(args, adata, adata_test, adata_test_X_norm, pred_expr_test, protein_mask_test, show_genes=True):
    test_scores = analyze_recon_loss(adata_test_X_norm, pred_expr_test, args.figure_path, protein_mask_test, loss_type='pearson')
    #test_scores_2 = analyze_recon_loss(adata_test_X_norm, pred_expr_test, figure_path, protein_mask_test, loss_type='spearman')
    test_scores_df = pd.DataFrame({"test_scores": test_scores}, index=list(adata_test.var_names[protein_mask_test.detach().cpu().numpy()]))

    # === SVG  ===
    svg_df = adata_test.uns["moranI"].join(test_scores_df, how="inner")

    sns.jointplot(
        data=svg_df,
        x="I",
        y="test_scores",
        kind="scatter",
        height=6,
        color="steelblue",
        marginal_kws=dict(bins=30, fill=True)
    )
    plt.suptitle("Joint Distribution of Pearson Correlation and I", y=1.02)
    plt.tight_layout()
    plt.savefig(args.figure_path+"joint_distribution.png", dpi=150)
    plt.close()

    # bins figures
    i_min_raw = svg_df["I"].min()
    i_max_raw = svg_df["I"].max()
    i_min = np.floor(i_min_raw * 10) / 10
    i_max = np.ceil(i_max_raw * 10) / 10
    '''
    if i_min >= 0:
        i_min = np.floor(i_min / 0.1) * 0.1
        if i_min > 0:
            i_min = max(0, i_min - 0.1)  # 稍微留一点空隙
    else:
        i_min = np.floor(i_min * 10) / 10
    '''
    bin_edges = np.arange(i_min, i_max + 0.1, 0.1)
    svg_df["I_bin"] = pd.cut(svg_df["I"], bins=bin_edges)

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=svg_df,
        x="I_bin",
        y="test_scores",
        color="lightblue",
        width=0.6,
        linewidth=2.5
    )
    sns.stripplot(
        data=svg_df,
        x="I_bin",
        y="test_scores",
        color="black",
        size=6,
        jitter=True,
        alpha=0.6
    )
    plt.grid(False)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Moran's I (Spatial Autocorrelation)",  fontsize=20)
    plt.ylabel("Pearson Correlations",  fontsize=20)
    plt.title("Pearson Correlations by Moran's I bins",  fontsize=20)
    plt.tight_layout()
    plt.savefig(args.figure_path + "joint_distribution_boxplot.png", dpi=150)
    plt.close()

    svg_df_sort = svg_df.sort_values("test_scores", ascending=False) #svg_df.sort_values("I", ascending=False)
    masked_protein_test = adata_test.var_names[protein_mask_test.detach().cpu().numpy()]
    filtered_svg_genes = [g for g in svg_df_sort.index.tolist() if g in set(masked_protein_test)]
    print('number of filtered_svg_genes:', len(filtered_svg_genes))

    if show_genes:
        proteins_list_all = []
        for pp in adata_test.var_names:
            if ";" in pp:
                pp_list = pp.split(';')
                for ppp in pp_list:
                    proteins_list_all.append(ppp)
            else:
                proteins_list_all.append(pp)
        mapped =  prot2gene(proteins_list_all, species='10090', to_upper=True, return_df=False)
            
        adata_predict = sc.AnnData(X=pred_expr_test.detach().cpu().numpy())
        adata_predict.obsm["spatial"] = adata_test.obsm["spatial"]
        adata_predict.var_names = adata_test.var_names
        if 'spatial' in adata.uns.keys():
            adata_predict.uns['spatial'] = adata.uns['spatial']

        adata_test.obs["group"] = "SP data"
        adata_predict.obs["group"] = "SPgen"
        n_genes2show = np.min([512, len(filtered_svg_genes)])
        nrows_img = 8  
        ncols_img = 3  
        
        plt.rcParams.update({
            "font.size": 18,            
            "axes.titlesize": 25,       
            "axes.labelsize": 25,       # x,y label
            "xtick.labelsize": 18,      
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "figure.titlesize": 25,
            "figure.autolayout": True,
        })

        for i_img in range(math.ceil(n_genes2show / nrows_img)):
            genes2show_batch = filtered_svg_genes[i_img * nrows_img: (i_img + 1) * nrows_img]
            fig, axes = plt.subplots(nrows=nrows_img, ncols=ncols_img, figsize=(ncols_img * 4, nrows_img * 4))  # 每图宽4高4

            for j, gene in enumerate(genes2show_batch):
                if 1:
                    vmax_value = "p98"
                    vmin_value = "p2"
                else:
                    combined = np.concatenate([adata_test[:, gene].X, adata_predict[:, gene].X], axis=0)
                    vmax_value = np.max(combined)
                    vmin_value = np.min(combined)

                test_val = adata_test[:, gene].X.flatten()
                pred_val = adata_predict[:, gene].X.flatten()

                mask = np.isfinite(test_val) & np.isfinite(pred_val)
                r, pval = pearsonr(test_val[mask], pred_val[mask])
                
                sc.pl.spatial(
                    adata_test, color=gene, ax=axes[j, 0], show=False, use_raw=False, img_key=None,
                    spot_size=args.base_spot_size, cmap='plasma', frameon=False,
                    vmax=vmax_value, vmin=vmin_value, ##
                )      
                #I = adata_test.uns["moranI"]['I'][gene]
                        
                sc.pl.spatial(
                    adata_predict, color=gene, ax=axes[j, 1], show=False, use_raw=False, img_key=None,
                    spot_size=args.base_spot_size, cmap='plasma', frameon=False,
                    vmax=vmax_value, vmin=vmin_value,  ##
                )

                if ';' in gene:
                    gene_list = gene.split(';')
                    for pp in gene_list:
                        if (pp in mapped.keys()) and (mapped[pp] is not None):
                            p2g = mapped[pp]
                            break
                else:    
                    p2g = mapped[gene] 
                    
                gene_name_print = gene
                if ';' in gene_name_print:
                    gene_name_print = min(gene_name_print.split(';'), key=len)
                if p2g is not None:
                    gene_name_print = gene_name_print + '(' + p2g +')'
                    
                if j==0: 
                    axes[j, 0].set_title(f"SP data\n {gene_name_print}", fontsize=20)
                    axes[j, 1].set_title(f"SPgen\n {gene_name_print}", fontsize=20)
                else:
                    axes[j, 0].set_title(f"{gene_name_print}", fontsize=20)
                    axes[j, 1].set_title(f"{gene_name_print}", fontsize=20)
                                
                fig = plt.gcf()
                for ax in fig.axes:
                    if not ax in axes:
                        ax.set_yticks([])       
                        ax.set_yticklabels([])  
                        ax.set_ylabel('')   
                
                        
                ax = axes[j, 2]   
                if 0:
                    sns.kdeplot(
                        x=test_val,
                        y=pred_val,
                        fill=True,
                        cmap="viridis",      
                        levels=100,
                        thresh=0.0,
                        bw_adjust=1.0,
                        ax=ax
                    )
                    ax.set_facecolor(cm.get_cmap("viridis")(0.0))
                    ax.grid(False)
                    
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    ax.text(
                        xlim[1], ylim[0], 
                        f"R: {r:.2f}",
                        ha='right', va='bottom',
                        fontsize=25,
                        color='white',     
                        bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3')
                    )
                else:
                    #ax.scatter(test_val, pred_val, s=20, alpha=0.5, color='deeppink')
                    #ax.grid(False)
                    
                    ax.scatter(
                        test_val,
                        pred_val,
                        s=30,             
                        alpha=1.0/np.log10(len(test_val)),       
                        color="#398DD1",  
                        edgecolors='none'
                    )

                    ax.set_facecolor('white')
                    ax.grid(False)

                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    ax.text(
                        xlim[1], ylim[0], 
                        f"R: {r:.2f}",
                        ha='right', va='bottom',
                        fontsize=25,
                        color='black',
                        bbox=dict(facecolor='white', edgecolor='black', alpha=0.7, boxstyle='round,pad=0.3')
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                ax.set_xlabel("SP data")
                ax.set_ylabel("SPgen")
                #ax.set_title("Joint distribution of Test vs Prediction")

            plt.tight_layout()
            fig.savefig(args.figure_path + f"svg_batch_{i_img}.png", dpi=150)
            plt.close(fig)
            
    return adata



def plot_GO_bar(
    txt_file,
    top_n=10,
    figsize=(12, 5),
    base_cmap="Blues",
    save_path=None,
    to_shorten=True,
):
    df = pd.read_csv(txt_file, sep="\t")

    df["Adjusted P-value"] = df["Adjusted P-value"].astype(float)
    df = df.sort_values("Adjusted P-value").head(top_n)

    if to_shorten:
        df["Term_clean"] = (
            df["Term"]
            .str.replace(r"\s*\(GO:\d+\)", "", regex=True)  
            .str.replace(r",.*$", "", regex=True)           
            .str.replace(r"\s*signaling pathway$", "", regex=True)
        )
    df["score"] = -np.log10(df["Adjusted P-value"])

    norm = plt.Normalize(df["score"].min(), df["score"].max())
    #cmap = cm.get_cmap(base_cmap)
    #colors = cmap(0.3 + 0.7 * norm(df["score"]))

    sns.set(style="white", context="paper", font_scale=1.4)

    plt.figure(figsize=figsize)

    plt.barh(
        y=df["Term_clean"],
        width=df["score"],
        #color=colors,
        edgecolor="none"
    )

    plt.gca().invert_yaxis()

    plt.xlabel(r"$-\log_{10}$(Adjusted P-value)", fontsize=20)
    plt.ylabel("")
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300)#, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
        plt.close()
    else:
        plt.show()
    return 



def parse_uniprot_sprot_species(dat_path):
    records = []

    acc = None
    species = []
    taxid = None

    with open(dat_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip()

            if line.startswith("AC   "):
                if acc is None:
                    acc = line.replace("AC   ", "").split(";")[0].strip()

            elif line.startswith("OS   "):
                species.append(line.replace("OS   ", "").strip())

            # Taxonomy ID
            elif line.startswith("OX   "):
                m = re.search(r"NCBI_TaxID=(\d+)", line)
                if m:
                    taxid = int(m.group(1))

            elif line == "//":
                if acc is not None:
                    records.append({
                        "accession": acc,
                        "species": " ".join(species).rstrip("."),
                        "taxid": taxid
                    })

                # reset
                acc = None
                species = []
                taxid = None

    return pd.DataFrame(records)




class ExpertDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, emb_dim, num_layers, dropout, use_residual):
        super().__init__()
        self.use_residual = use_residual
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        self.convs.append(GATConv(in_dim, hidden_dim, heads=2, concat=False))
        self.norms.append(nn.LayerNorm(hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=2, concat=False))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.convs.append(GATConv(hidden_dim, emb_dim, heads=2, concat=False))
        self.norms.append(nn.LayerNorm(emb_dim))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            h = conv(x, edge_index)
            h = self.norms[i](h)
            if i < len(self.convs) - 1:
                h = F.gelu(h) ########## h = F.gelu(h)  h = F.relu(h)
                h = self.dropout(h)
                if self.use_residual and h.shape == x.shape:
                    x = x + h
                else:
                    x = h
            else:
                x = h
        return x


    
class GNNCell2ExprDecoder(nn.Module):
    def __init__(self, in_dim, g_emb_dims, hidden_dim=1024, emb_dim=512, num_layers=4,
                 dropout=0.1, use_residual=True, mask_ratio=0.5):
        """
        g_emb_dims: list[int]
        """
        super().__init__()
        assert isinstance(g_emb_dims, (list, tuple)) and len(g_emb_dims) > 0, \
            "g_emb_dims e.g. [128, 256, 64]"

        self.num_branches = len(g_emb_dims)
        self.mask_ratio = mask_ratio
        self.emb_dim = emb_dim
        self.g_emb_dims = g_emb_dims

        self.proj_layers = nn.ModuleList([
            nn.Linear(in_dim, emb_dim) for in_dim in g_emb_dims
        ])

        total_in_dim = sum(g_emb_dims)
        self.hidden_dim = hidden_dim

        self.fusion_net = nn.Sequential(
            nn.Linear(total_in_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.num_branches),
            nn.Softmax(dim=-1)
        ).to(next(self.parameters()).device)

        self.decoder = ExpertDecoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            emb_dim=emb_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_residual=use_residual
        )

    def forward(self, cell_emb, gene_emb, edge_index):
        """
        cell_emb: [N_cells, emb_dim]
        gene_emb: [N_genes, sum(g_emb_dims)]
        """
        N_cells, _ = cell_emb.shape
        N_genes, D = gene_emb.shape


        gene_chunks = torch.split(gene_emb, self.g_emb_dims, dim=-1)

        if self.training and self.mask_ratio > 0:
            masked_chunks = []
            for g in gene_chunks:
                mask = (torch.rand_like(g) > self.mask_ratio).float()
                masked_chunks.append(g * mask)
            gene_chunks = masked_chunks

        gene_proj_list = [
            proj(g) for proj, g in zip(self.proj_layers, gene_chunks)
        ]  # [N_genes, emb_dim]

        stacked_gene_emb = torch.stack(
            gene_proj_list, dim=-1
        )  # [N_genes, emb_dim, num_branches]

        fusion_weight = self.fusion_net(
            gene_emb
        )  # [N_genes, num_branches]

        fusion_weight = fusion_weight.unsqueeze(1)  # [N_genes, 1, num_branches]

        fused_gene_emb = torch.sum(
            stacked_gene_emb * fusion_weight,
            dim=-1
        )  # [N_genes, emb_dim]

        with torch.no_grad():
            fused_gene_emb = F.normalize(fused_gene_emb, dim=-1)

        h = self.decoder(cell_emb, edge_index)  # [N_cells, emb_dim]

        recon = torch.matmul(
            h,
            fused_gene_emb.T
        )  # [N_cells, N_genes]

        self.fusion_weight = fusion_weight.squeeze(1)

        return recon

    def get_gate(self):
        return self.fusion_weight
        
    

