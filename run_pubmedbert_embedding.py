#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Offline UniProt SwissProt text -> PubMedBERT embeddings

Usage:

  # From a text file
  python uniprot_text_to_pubmedbert_embeddings.py \
      --ids_file ids.txt \
      --uniprot_dat ./data/uniprot/uniprot_sprot.dat \
      --out_prefix out/uniprot_pubmedbert

  # Or pass IDs directly
  python uniprot_text_to_pubmedbert_embeddings.py \
      --ids P69905 P68871 P01009 \
      --uniprot_dat ./data/uniprot/uniprot_sprot.dat \
      --out_prefix out/uniprot_pubmedbert
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Offline UniProt -> descriptive text -> PubMedBERT embeddings pipeline

Requirements:
    pip install torch transformers pandas pyarrow tqdm biopython
"""

import os, re, argparse, pickle, sys
from typing import List
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from Bio import SwissProt

EVIDENCE_BRACE_RE = re.compile(r"\{[^}]+\}")

# --------------------------- Text processing ---------------------------

def clean_uniprot_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = EVIDENCE_BRACE_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def concat_uniprot_fields(rec: SwissProt.Record) -> str:
    parts = []
    if rec.description:
        parts.append(clean_uniprot_text(rec.description))
    for c in rec.comments:
        if c:
            parts.append(clean_uniprot_text(c))
    return " \n".join(parts)

# --------------------------- PubMedBERT Embedder ---------------------------
class PubMedBERTEmbedder:
    def __init__(self, model_name="./biomedNLP", device=None,
                 max_length=512, batch_size=8,
                 pooling='cls', use_stride=False, stride=128):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=True)
        self.model.eval().to(self.device)
        self.max_length = max_length
        self.stride = stride if use_stride else 0
        self.batch_size = batch_size
        self.pooling = pooling.lower()
        self.hidden_size = self.model.config.hidden_size

    def _pool(self, outputs, attention_mask):

        if self.pooling == 'cls':
            # [batch, seq_len, hidden] 
            return outputs[:, 0, :]
        elif self.pooling == 'mean':
            # 平均池化
            mask = attention_mask.unsqueeze(-1).type_as(outputs)
            summed = (outputs * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            return summed / counts
        else:
            raise ValueError("pooling must be 'cls' or 'mean'")

    def embed_texts(self, texts: List[str]) -> torch.Tensor:

        embs = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding batches"):
            batch_texts = texts[i:i+self.batch_size]
            batch_embs = []
            for txt in batch_texts:
                if not txt:
                    txt = "[UNK]"  

                enc = self.tokenizer(
                    txt,
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length" if self.stride == 0 else False,
                    return_overflowing_tokens=True if self.stride > 0 else False,
                    stride=self.stride,
                )

                input_ids = enc["input_ids"].to(self.device)
                att_mask = enc["attention_mask"].to(self.device)

                chunk_vecs = []
                with torch.no_grad():
                    outputs = self.model(input_ids, attention_mask=att_mask)
                    pooled = self._pool(outputs.last_hidden_state, att_mask)  # [chunks, hidden]
                    chunk_vecs.append(pooled)

                chunk_vecs = torch.cat(chunk_vecs, dim=0)
                doc_emb = chunk_vecs.mean(dim=0) 
                batch_embs.append(doc_emb.cpu())

            embs.extend(batch_embs)

        return torch.stack(embs, dim=0).numpy()
    
# --------------------------- UniProt offline fetch ---------------------------

def build_accession_cache(dat_path, target_accessions, cache_path=None, use_cache=True):
    """
    Parse only needed accessions. 
    If use_cache=True and cache_path is provided, load/save cache; otherwise always parse.
    """
    acc2text = {}
    if use_cache and cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # 解析 .dat 文件
    with open(dat_path) as handle:
        for rec in tqdm(SwissProt.parse(handle), desc="Parsing UniProt .dat"):
            for acc in rec.accessions:
                if acc in target_accessions:
                    acc2text[acc] = concat_uniprot_fields(rec)
            if len(acc2text) >= len(target_accessions):
                break  # Got all needed

    if use_cache and cache_path:
        with open(cache_path, "wb") as f:
            pickle.dump(acc2text, f)

    return acc2text

# --------------------------- Utility ---------------------------

def expand_embeddings_to_df(matrix: np.ndarray, prefix: str = "emb") -> pd.DataFrame:
    cols = [f"{prefix}_{i:04d}" for i in range(matrix.shape[1])]
    return pd.DataFrame(matrix, columns=cols)

def read_ids(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

# 保留有用的 CC 类型
CC_KEEP_PREFIXES = [
    "FUNCTION",
    "TISSUE SPECIFICITY",
    "PTM",
    "DISEASE",
    "MISCELLANEOUS",
    "SIMILARITY",
]

EVIDENCE_BRACE_RE = re.compile(r"\{[^}]+\}")

def extract_useful_text_from_row(text: str) -> str:
    if not isinstance(text, str):
        return ""

    parts = []

    # DE：RecName / AltName / Contains
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("RecName:") or line.startswith("AltName:") or line.startswith("Contains:"):
            parts.append(clean_uniprot_text(line))

    for line in text.split("\n"):
        line = line.strip()
        for prefix in CC_KEEP_PREFIXES:
            if line.startswith(f"{prefix}:"):
                content = line[len(prefix)+1:].strip()
                parts.append(clean_uniprot_text(content))
                break

    return " \n".join(parts)

def clean_uniprot_df_text(df: pd.DataFrame) -> list:
    cleaned_texts = df["text"].apply(extract_useful_text_from_row).tolist()
    return cleaned_texts

def extract_uniprot_ids(dat_path: str, out_path: str = "ids.txt"):
    accessions = []

    with open(dat_path, "r", encoding="utf-8") as handle:
        for record in tqdm(SwissProt.parse(handle), desc="Parsing UniProt .dat"):
            if record.accessions:
                accessions.append(record.accessions[0])

    print(f"get {len(accessions)}  UniProt ID, writing to {out_path}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(accessions))

    return accessions

# --------------------------- Main ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids_file", type=str, default="pubmedbert/ids.txt", help="ids.txt")
    parser.add_argument("--uniprot_dat", type=str, default="./data/uniprot/uniprot_sprot.dat")
    parser.add_argument("--out_prefix", type=str, default="pubmedbert/uniprot_pubmedbert")
    parser.add_argument("--model", type=str, default="./biomedNLP")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    # make ids txt
    if not os.path.exists(args.ids_file):
        extract_uniprot_ids(dat_path=args.uniprot_dat, out_path=args.ids_file)
    accessions = read_ids(args.ids_file)

    # ---------- Build or load cache ----------
    acc2text = build_accession_cache(args.uniprot_dat, set(accessions), use_cache=False, cache_path=args.out_prefix+".cache.pkl")

    # ---------- Prepare DataFrame ----------
    data = []
    for acc in accessions:
        text = acc2text[acc]
        data.append({"accession": acc, "text": text})
    df = pd.DataFrame(data)
    df.to_csv(args.out_prefix + "_text.tsv", sep="\t", index=False)

    # ---------- Embed ----------
    embedder = PubMedBERTEmbedder(
        model_name=args.model,
        device=args.device,
        max_length=args.max_length,
        use_stride=False, 
        stride=args.stride,
        pooling='cls',
        batch_size=args.batch_size,
    )
    cleaned_text_list = clean_uniprot_df_text(df)
    emb_matrix = embedder.embed_texts(cleaned_text_list)
    np.save(args.out_prefix + "_embeddings.npy", emb_matrix)

    # ---------- Save parquet with expanded embedding ----------
    emb_df_ex = expand_embeddings_to_df(emb_matrix, prefix="emb")
    out_df = pd.concat([df.reset_index(drop=True), emb_df_ex], axis=1)
    out_df.to_parquet(args.out_prefix + "_embeddings.parquet", index=False)

    print(f"Done! Outputs:\n  {args.out_prefix}_text.tsv\n  {args.out_prefix}_embeddings.npy\n  {args.out_prefix}_embeddings.parquet")



if __name__ == "__main__":
    main()
