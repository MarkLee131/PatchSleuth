# Phase 2: Fine-tuning CodeReviewer for re-ranking commits

This Folder contains the fine-tuning code for the CodeReviewer model to re-rank the commits. For convenience, we also reuse the code for ablation study and comparison with baselines.

## 1. Data Preparation

Phase-2 uses the output of Phase-1 as the input data, i.e., the top-100 commits (retrieved by our Hybrid Retriever) for each CVE. 


## 2. Fine-tuning CodeReviewer

```bash
.
|-- Ablation
|   |-- ablation_data_prepare.py
|   |-- codellama
|   |-- diff_only
|   `-- msg_only
|-- README.md
|-- configs.py # Configuration file for fine-tuning
|-- evaluate.py # Evaluation script
|-- evaluate_deprecated.py # Deprecated 
|-- load_data.py # Data loading script
|-- load_data_colbert.py # Data loading script for ColBERT (Baseline)
|-- load_data_deprecated.py # Deprecated
|-- main.py # Main script for fine-tuning
|-- main_deprecated.py # Deprecated
|-- metrics.py # Metrics calculation script
|-- models.py # Model design script for Phase-2
|-- output_1007 # Fine-tuned model files*
|   `-- Checkpoints 
|   `-- final_model.pt

```
Since the fine-tuned model files are large, we have not included them in the repository. However, you can download the fine-tuned model files from [[Google Drive]( https://drive.google.com/file/d/1s7pgHduaXoumEx_stb32S75Ysj39U0bd/view?usp=sharing)](https://drive.google.com/file/d/1JIxaZlZQHGQ_JDu3YXFfi0YTa9JK8YY4/view?usp=sharing).

To reproduce our metrics, we also provided the corresponding test set (Top 100 result for 480 CVEs, which is the output from Phase-1) I used for evaluating in our paper at [top_100_fusion.zip](./top_100_fusion.zip). 
