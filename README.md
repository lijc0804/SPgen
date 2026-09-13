# SPgen: Proteome-wide Spatial Proteomics generation using multi-modality foundation models

SPgen is a multimodal foundation-model-based framework for **proteome-wide spatial protein generation** from spatial proteomics data.

SPgen integrates pretrained representations of gene/transcriptomic information, protein amino acid sequences, and protein functional descriptions with spatial tissue information. By learning transferable relationships between intrinsic protein properties and experimentally measured spatial protein distributions, SPgen enables the prediction of spatial distributions for proteins beyond the experimentally measured protein panel.

## Key Features

- Proteome-wide spatial protein generation beyond experimentally measured proteins
- Integration of three pretrained foundation models:
  - **scGPT** for gene-level biological representations
  - **ESM** for protein sequence representations
  - **PubMedBERT** for protein functional-description representations
- Graph neural network modeling of spatial tissue structure
- Protein-level representation learning across multiple biological modalities
- Prediction of previously unmeasured proteins
- Evaluation on multiple published spatial proteomics datasets
- Reproducible protein-level train/test splitting and downstream evaluation

---

## Repository Structure

```text
SPgen/
├── README.md
├── License.md
├── SPgen_env.yml
├── SP_demo.py
├── SP_predict.py
├── utils.py
├── run_ESM.py
├── run_pubmedbert_embedding.py
├── SP_predict_analysis_Cerebellum-PLATO_demo.ipynb
├── SP_predict_test_analysis_Intestinal-villi_demo.ipynb
└── SP_predict_test_analysis_mouse_brain_coronal_demo.ipynb
```

The main scripts are:

- `SP_demo.py`: trains and evaluates SPgen on measured spatial proteins using a protein-level train/test split.
- `SP_predict.py`: applies the trained SPgen model to generate spatial distributions for additional proteins.
- `run_ESM.py`: generates protein sequence embeddings using ESM.
- `run_pubmedbert_embedding.py`: generates protein functional-description embeddings using PubMedBERT.
- `utils.py`: contains the model architecture, spatial graph construction, loss functions, embedding alignment, and evaluation utilities.
- `*_demo.ipynb`: downstream analysis and visualization notebooks for the three datasets.

---

# Installation

## 1. Create the Conda Environment

Clone or download this repository and create the provided environment:

```bash
conda env create -f SPgen_env.yml
conda activate SPgen_env
```

The provided environment was configured with:

```text
Python 3.9
PyTorch 2.1.2 + CUDA 12.1
Scanpy 1.10.3
Squidpy 1.6.1
AnnData 0.10.9
NumPy 1.26.4
Pandas 2.3.3
scikit-learn 1.6.0
PyTorch Geometric 2.6.1
```

SPgen is primarily designed for GPU execution.

For generating ESM and PubMedBERT representations, we suggest to install the corresponding model packages in an individual environment.


---

# Data and Model Availability

SPgen uses publicly available protein annotations, pretrained foundation models, and spatial proteomics datasets.

## 1. Protein Annotations from UniProt

Protein annotations, including **amino acid sequences and functional descriptions**, are obtained from UniProt:

https://www.uniprot.org/help/downloads

SPgen uses UniProt/Swiss-Prot files in both FASTA and DAT formats.

The expected files are:

```text
data/
└── uniprot/
    ├── uniprot_sprot.fasta
    └── uniprot_sprot.dat
```

`uniprot_sprot.fasta` is used for retrieving protein amino acid sequences and generating ESM representations.

`uniprot_sprot.dat` is used for extracting protein functional annotations and generating PubMedBERT representations.

---

# Pretrained Foundation Models

Three pretrained foundation models are employed in SPgen.

## scGPT

scGPT is used to obtain pretrained gene-level biological representations.

Official repository:

https://github.com/bowang-lab/scGPT/

SPgen expects the precomputed scGPT gene embeddings at:

```text
scGPT_emb/gene_all.tsv
```

The embeddings should be generated using the pretrained scGPT model following the instructions in the official scGPT repository.

---

## ESM

ESM is used to encode protein amino acid sequences.

The ESM-1b model used in this study is:

```text
esm1b_t33_650M_UR50S
```

The pretrained model is available from:

https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt

Protein sequence embeddings can be generated using:

```bash
python run_ESM.py
```

The script reads:

```text
data/uniprot/uniprot_sprot.fasta
```

and generates files including:

```text
ESM_embedding/
├── all_emb.npy
├── gene_names.txt
└── protein_gene.csv
```

`SP_demo.py` and `SP_predict.py` use:

```text
ESM_embedding/all_emb.npy
ESM_embedding/gene_names.txt
```

as the ESM representation input.

---

## PubMedBERT

PubMedBERT is used to encode textual descriptions of protein functions obtained from UniProt.

The pretrained model used in this study is available from:

https://huggingface.co/NeuML/pubmedbert-base-embeddings

Download the PubMedBERT model from the link above and save it to `./biomedNLP/` before running the embedding script.

SPgen uses `run_pubmedbert_embedding.py` to extract relevant UniProt functional descriptions and calculate PubMedBERT embeddings.

For example:

```bash
python run_pubmedbert_embedding.py \
    --uniprot_dat ./data/uniprot/uniprot_sprot.dat \
    --model ./biomedNLP \
    --out_prefix pubmedbert/uniprot_pubmedbert
```

The resulting files include:

```text
pubmedbert/
├── ids.txt
├── uniprot_pubmedbert_text.tsv
├── uniprot_pubmedbert_embeddings.npy
└── uniprot_pubmedbert_embeddings.parquet
```

SPgen uses:

```text
pubmedbert/ids.txt
pubmedbert/uniprot_pubmedbert_embeddings.npy
```

for downstream modeling.

---

# Spatial Proteomics Datasets

Three previously published spatial proteomics datasets are used in this study.

SPgen does not redistribute these datasets. Please download them from their original repositories and cite the corresponding studies when using them.

## 1. Intestinal Villi Dataset

The intestinal villi spatial proteomics dataset was released by the **PLATO** study.

The original dataset is available from the Flow2Spatial repository:

https://github.com/bioinfo-biols/Flow2Spatial/blob/main/tests/adata.h5ad

For use with the current SPgen scripts, place or rename the downloaded AnnData file as:

```text
data/Flow2Spatial/Intestinal-villi.h5ad
```

---

## 2. Cerebellum Dataset

The cerebellum spatial proteomics dataset was also released by the **PLATO** study.

The dataset is available from:

https://github.com/bioinfo-biols/Flow2Spatial/tree/main/datasets

The SPgen implementation expects:

```text
data/Flow2Spatial/Cerebellum-PLATO.h5ad
```

The Flow2Spatial repository provides the PLATO cerebellum spatial proteomics data in AnnData format.

---

## 3. Mouse Brain Coronal Dataset

The mouse brain coronal spatial proteomics dataset was released by the **Spatial-DC** study.

The original data are available from Zenodo:

https://zenodo.org/records/14523511

After downloading and preprocessing the corresponding data, SPgen expects the input file at:

```text
data/mouse_brain_coronal/intersected_spatial_proteomics.h5ad
```

---

# Expected Directory Structure

Before running SPgen, the relevant files should be organized approximately as follows:

```text
SPgen/
│
├── data/
│   ├── uniprot/
│   │   ├── uniprot_sprot.fasta
│   │   └── uniprot_sprot.dat
│   │
│   ├── Flow2Spatial/
│   │   ├── Intestinal-villi.h5ad
│   │   └── Cerebellum-PLATO.h5ad
│   │
│   └── mouse_brain_coronal/
│       └── intersected_spatial_proteomics.h5ad
│
├── scGPT_emb/
│   └── gene_all.tsv
│
├── ESM_embedding/
│   ├── all_emb.npy
│   ├── gene_names.txt
│   └── protein_gene.csv
│
├── pubmedbert/
│   ├── ids.txt
│   └── uniprot_pubmedbert_embeddings.npy
│
├── biomedNLP/
│   └── [PubMedBERT model files]
│
├── SP_demo.py
├── SP_predict.py
├── utils.py
└── ...
```

---

# Running SPgen

## Step 1. Train and Evaluate SPgen

Run the demonstration script on one of the three supported datasets.

For example, for the PLATO cerebellum dataset:

```bash
python SP_demo.py \
    --dataset_name Cerebellum-PLATO 
```

Supported dataset names are:

```text
Cerebellum-PLATO
Intestinal-villi
mouse_brain_coronal
```

---


# Spatial Graph Construction

SPgen represents tissue spatial organization as a graph constructed from spatial coordinates stored in:

```python
adata.obsm["spatial"]
```

In the current implementation, a spatial graph is constructed using:

```text
K = 4
```

spatial neighbors.

The resulting spatial graph and multimodal protein embeddings are jointly used by the graph neural network to generate spatial protein distributions.

---


# Step 2. Proteome-Wide Prediction

After running `SP_demo.py`, the trained model parameters and processed AnnData object are saved in the corresponding result directory.

Proteome-wide spatial predictions can then be generated using:

```bash
python SP_predict.py \
    --dataset_name Cerebellum-PLATO \
    --species 10090 
```

For mouse datasets:

```text
NCBI taxonomy ID = 10090
```

The `--species` parameter should be changed accordingly when applying SPgen to another species.


---

# Output Files

For the default cerebellum demonstration, results are stored in a directory similar to:

```text
figures_Cerebellum-PLATO_demo_scGPT+ESM+pubmedbert/
```

Important output files include:

```text
model_weights.pth
adata_with_predict.h5ad
loss_curve.png
adata_predict_union.h5ad
```

or, when intersection mode is used:

```text
adata_predict_intersection.h5ad
```

### `model_weights.pth`

Trained SPgen model parameters.

### `adata_with_predict.h5ad`

Contains measured and reconstructed proteins from the protein-level holdout experiment.

Important fields include:

```text
adata.layers["GT"]
adata.layers["predict"]
adata.uns["train_gene_idx"]
adata.uns["test_gene_idx"]
adata.varm["gene_emb"]
```

### `adata_predict_union.h5ad`

Contains proteome-wide spatial protein predictions.

The predicted protein identities are stored as UniProt IDs, with corresponding gene annotations included in `adata.var`.

---

# Step 3. Downstream Analysis

Three example notebooks are provided for reproducing downstream analyses and visualizations.

### Mouse brain coronal

```bash
jupyter notebook SP_predict_test_analysis_mouse_brain_coronal_demo.ipynb
```

### Intestinal villi

```bash
jupyter notebook SP_predict_test_analysis_Intestinal-villi_demo.ipynb
```

### Cerebellum

```bash
jupyter notebook SP_predict_analysis_Cerebellum-PLATO_demo.ipynb
```


---

# Reproducibility Notes

For reproducible execution, please ensure that:

1. The same UniProt release is used when generating ESM and PubMedBERT representations.
2. Dataset filenames and directory structures follow those described above.
3. The same pretrained versions of scGPT, ESM, and PubMedBERT are used.
4. The provided `SPgen_env.yml` environment is used whenever possible.
5. Random seeds in `SP_demo.py` are kept unchanged when reproducing the demonstration experiments.
6. The same dataset-specific preprocessing used in the released code is retained.
7. Protein identifiers are kept consistent with UniProt accessions and corresponding gene symbols.

Because the three protein representations originate from different pretrained models and identifier systems, proteins without an available representation in a particular modality are automatically masked for that modality by the current implementation.

---

# Typical Workflow

A complete SPgen workflow consists of:

```text
1. Download spatial proteomics datasets
             ↓
2. Download UniProt protein annotations
             ↓
3. Prepare scGPT gene representations
             ↓
4. Generate ESM protein sequence representations
             ↓
5. Generate PubMedBERT functional representations
             ↓
6. Run SP_demo.py
   - protein-level train/test split
   - SPgen training
   - held-out protein evaluation
             ↓
7. Run SP_predict.py
   - proteome-wide spatial prediction
             ↓
8. Analyze predictions using the provided notebooks
```

---


# Citation

If you use SPgen in your research, please cite:

**Li, J., Yang, K., Che, Q., Zheng, D., Wei, W., Jin, C., and Yuan, Y.
SPgen: Proteome-wide Spatial Proteomics generation using multi-modality foundation models.
bioRxiv 2026.07.16.739037 (2026).
https://doi.org/10.64898/2026.07.16.739037**

The complete citation information will be added upon publication.

Please also cite the original studies and resources corresponding to the spatial proteomics datasets, UniProt annotations, scGPT, ESM, and PubMedBERT when appropriate.

---
