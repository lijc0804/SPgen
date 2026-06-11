# SPgen Demo

This repository provides a demonstration workflow for SPgen.

## Step 1: Create the Environment

Create the conda environment using the provided configuration file:

```bash
conda env create -f SPgen_env.yml
conda activate SPgen_env
```

## Step 2: Run the Demo Script

Execute the demo script:

```bash
python SP_demo.py
```

## Step 3: Generate Predictions

Run the prediction script:

```bash
python SP_predict.py
```

## Step 4: Analyze the Prediction Results

Open and run the Jupyter notebook:

```bash
jupyter notebook SP_predict_test_analysis_mouse_brain_coronal_demo.ipynb
```

or

```bash
jupyter lab SP_predict_test_analysis_mouse_brain_coronal_demo.ipynb
```

## Workflow Summary

1. Create the environment using `SPgen_env.yml`.
2. Run `SP_demo.py`.
3. Run `SP_predict.py`.
4. Run `SP_predict_test_analysis_mouse_brain_coronal_demo.ipynb` for downstream analysis and visualization.
