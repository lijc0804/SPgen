import torch
import esm
from typing import List, Tuple
import re
from Bio import SeqIO
import os
import numpy as np
import pandas as pd

class FlexibleEmbedder:
    def __init__(self, fasta_path: str = "data/uniprot/uniprot_sprot.fasta", esm_model_name: str = "esm1b_t33_650M_UR50S", gpu_id:int=0):
        if not os.path.exists(fasta_path):
            raise FileNotFoundError(f"FASTA file {fasta_path} not found.")
        self.records = {record.id: record for record in SeqIO.parse(fasta_path, "fasta")}

        # 加载ESM模型
        self.model, self.alphabet = esm.pretrained.esm1b_t33_650M_UR50S() if esm_model_name == "esm1b_t33_650M_UR50S" else esm.pretrained.load_model_and_alphabet(esm_model_name)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.repr_layer = 33  

    def fetch_by_gene_names(self, gene_names: List[str]) -> Tuple[List[str], List[str], List[str]]:
        success, sequences, missing = [], [], []

        for gene in gene_names:
            gene_upper = gene.upper()
            found = False
            for rid, record in self.records.items():
                description = record.description.upper()
                if f"GN={gene_upper}" in description:
                    sequences.append(str(record.seq))
                    success.append(gene)
                    found = True
                    break
            if not found:
                missing.append(gene)

        return success, sequences, missing

    def fetch_by_uniprot_ids(self, uniprot_ids: List[str]) -> Tuple[List[str], List[str], List[str]]:
        id_to_record = {}
        for rid, record in self.records.items():
            if "|" in rid:
                parts = rid.split("|")
                if len(parts) >= 2:
                    uniprot_id = parts[1]
                    id_to_record[uniprot_id] = record

        success, sequences, missing = [], [], []


        for uid in uniprot_ids:
            if ';' in uid:
                uid_list = uid.split(";")
                flag = 1
                for uid_ in uid_list:
                    if uid_ in id_to_record:
                        record = id_to_record[uid_]
                        sequences.append(str(record.seq))
                        success.append(uid_)
                        flag = 0
                        break
                if flag:
                    missing.append(uid)
            else:
                if uid in id_to_record:
                    record = id_to_record[uid]
                    sequences.append(str(record.seq))
                    success.append(uid)
                else:
                    missing.append(uid)

        return success, sequences, missing

    def embed_sequences(self, sequences: List[str], batch_size: int = 16) -> torch.Tensor:
        all_embeddings = []
        self.model.eval()

        trimmed_sequences = [seq[:1022] for seq in sequences]

        if not trimmed_sequences:
            raise RuntimeError("No avaliable protein sequence")

        with torch.no_grad():
            for i in range(0, len(trimmed_sequences), batch_size):
                batch_seqs = trimmed_sequences[i:i + batch_size]
                batch_labels = [("", seq) for seq in batch_seqs]
                _, _, batch_tokens = self.batch_converter(batch_labels)
                batch_tokens = batch_tokens.to(self.device)

                # 获取嵌入表示
                outputs = self.model(batch_tokens, repr_layers=[self.repr_layer])
                reps = outputs["representations"][self.repr_layer]
                emb = reps[:, 0, :]  
                all_embeddings.append(emb.cpu())

        return torch.cat(all_embeddings, dim=0)


    def embed(self, names: List[str], mode: str = "gene_name", batch_size: int = 16) -> Tuple[torch.Tensor, List[str], List[str], List[str]]:
        if mode not in ["gene_name", "uniprot_id"]:
            raise ValueError("mode 必须是 'gene_name' 或 'uniprot_id'！")

        if mode == "gene_name":
            success_ids, sequences, missing_ids = self.fetch_by_gene_names(names)
        else:
            success_ids, sequences, missing_ids = self.fetch_by_uniprot_ids(names)

        id_to_seq = dict(zip(success_ids, sequences))

        valid_ids, valid_seqs = [], []
        long_ids = []

        for sid, seq in id_to_seq.items():
            valid_ids.append(sid)
            valid_seqs.append(seq)
            if len(seq) > 1022:
                long_ids.append(sid)

        embedding_dict = {}  # sid -> embedding

        if valid_seqs:
            valid_embeddings = self.embed_sequences(valid_seqs, batch_size=batch_size)
            for sid, emb in zip(valid_ids, valid_embeddings):
                embedding_dict[sid] = emb
            emb_dim = valid_embeddings.shape[1]
        else:
            emb_dim = self.model.embed_tokens.embedding_dim

        aligned_embeddings = []
        for name in names:
            if name in embedding_dict:
                aligned_embeddings.append(embedding_dict[name])
            else:
                aligned_embeddings.append(torch.zeros(emb_dim))

        embeddings = torch.stack(aligned_embeddings, dim=0)

        return embeddings, valid_ids, missing_ids, long_ids

    def save_embeddings(self, embeddings: torch.Tensor, success_ids: List[str], save_path: str = "embeddings.npy"):
        embeddings_array = embeddings.numpy()
        np.save(save_path, embeddings_array)



import scanpy as sc
import pandas as pd

def make_protein_gene_file(fasta_file="data/uniprot/uniprot_sprot.fasta", save_file="ESM_embedding/protein_gene.csv", species=None):
    os.makedirs("ESM_embedding", exist_ok=True)
    uniprot2gene = {}
    gene2uniprot = {}

    protein_list = []
    gene_list = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        desc = record.description

        # ===================== 1. species =====================
        if species is not None:
            if species not in desc:
                continue


        # 2. UniProt accession
        # sp|P12345|GENE_MOUSE ...
        parts = record.id.split("|")
        if len(parts) < 2:
            continue
        uid = parts[1]

        # 3. get gene symbol (GN=)
        gene_match = re.search(r"GN=([A-Za-z0-9_\-]+)", desc)

        if gene_match:
            gene_id = gene_match.group(1).split("-")[0].upper()
        else:
            gene_id = parts[2].upper()


        # 4. delete isoform
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
    df_protein_gene.to_csv(save_file, index=False)
    return df_protein_gene

df_protein_gene = make_protein_gene_file(fasta_file="data/uniprot/uniprot_sprot.fasta", save_file="ESM_embedding/protein_gene.csv")
#df = pd.read_csv("ESM_embedding/protein_gene.csv")
uniprot_ids = df_protein_gene["Protein_ID"].tolist()

embedder = FlexibleEmbedder()
embeddings, success_ids, missing_ids, long_ids = embedder.embed(uniprot_ids, mode="uniprot_id")

print("Embeddings shape:", embeddings.shape)

embeddings_array = embeddings.numpy()
df = pd.DataFrame(embeddings_array.T, columns=uniprot_ids)  
#df.to_csv("ESM_embedding/all.tsv", sep='\t')
np.save("ESM_embedding/all_emb.npy", embeddings_array)   # shape: [n_gene, d]
pd.Series(uniprot_ids).to_csv("ESM_embedding/gene_names.txt", index=False, header=False)
