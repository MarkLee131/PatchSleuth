"""
2024.08.13

This script is used to evaluate the model on the test set.

2024.12.05
tip: use the conda environment `patch_match` to run this script.
"""



import os
import logging
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM
import configs
from load_data import CVEDataset
# from load_data_colbert import CVEDataset #### Updated to load_data_colbert


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

 
class CVEClassifier(pl.LightningModule):
    def __init__(self, 
                 num_classes=1,
                 dropout=0.1,
                 lr=5e-5,
                 num_train_epochs=20,
                 warmup_steps=1000,
                 ):
        
        super().__init__()
        self.codeReviewer = AutoModelForSeq2SeqLM.from_pretrained(
            "microsoft/codereviewer").encoder
        
        self.save_hyperparameters()
        self.dropout = dropout
        self.criterion = nn.BCEWithLogitsLoss()

        # Dropout layer
        self.dropout_layer = nn.Dropout(self.dropout)
        # Fully connected layer for output
        self.fc = nn.Linear(2 * self.codeReviewer.config.hidden_size, num_classes)

    
    def forward(self, input_ids_desc, attention_mask_desc, input_ids_msg_diff, attention_mask_msg_diff):
        
        # Get [CLS] embeddings for desc and msg+diff
        desc_cls_embed = self.codeReviewer(input_ids=input_ids_desc, attention_mask=attention_mask_desc).last_hidden_state[:, 0, :]
        msg_diff_cls_embed = self.codeReviewer(input_ids=input_ids_msg_diff, attention_mask=attention_mask_msg_diff).last_hidden_state[:, 0, :]
        
        # Concatenate [CLS] embeddings
        concatenated = torch.cat((desc_cls_embed, msg_diff_cls_embed), dim=1)
        
        # Apply dropout
        dropped = self.dropout_layer(concatenated)
        
        # Pass through the fully connected layer
        output = self.fc(dropped)
        
        return output

    
    
    def common_step(self, batch):
        predict = self(
            batch['input_ids_desc'],
            batch['attention_mask_desc'],
            batch['input_ids_msg_diff'],
            batch['attention_mask_msg_diff']
        )
        # ValueError: Target size (torch.Size([512])) must be the same as input size (torch.Size([512, 1]))
        predict = predict.squeeze(1)
        loss = self.criterion(predict, batch['label'])
        return loss


    def training_step(self, batch, dataloader_idx=None):
        loss = self.common_step(batch)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        loss = self.common_step(batch)
        self.log("validation_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        loss = self.common_step(batch)

        return loss

    def test_dataloader(self):
        return test_dataloader


def load_checkpoint(load_path, model):
    # torch.save(model, model_save_path)
    model = torch.load(load_path)
    
    return model

def save_outputs_to_csv(cve, output, label, data_path):
    """Save model outputs to a CSV file."""
    output_df = pd.DataFrame({'cve': [cve], 'output': [output], 'label': [label]})
    output_df.to_csv(data_path, mode='a', header=False, index=False)

def evaluate_single_batch(model, batch):
    """Evaluate model on a single batch and return the output."""
    device = configs.device
    model.to(device)

    input_keys = ['input_ids_desc', 'attention_mask_desc', 'input_ids_msg_diff', 'attention_mask_msg_diff']
    inputs = [batch[key].to(device) for key in input_keys]

    return model(*inputs)

def compute_metrics(cve_data, k_values, rank_output_path):
    """Compute metrics (recall@k, MRR, Manual Efforts) for the model outputs."""
    recalls = {k: [] for k in k_values}
    mrrs = []
    manual_efforts = {k: [] for k in k_values}
    manual_efforts_sum = {k: 0 for k in k_values}
    manual_efforts_count = {k: 0 for k in k_values}
    rank_data = []
    
    for cve, data in cve_data.items():
        data.sort(key=lambda x: x[0], reverse=True)
        ranks = [(i + 1, label) for i, (_, label) in enumerate(data)]

        for rank, label in ranks:
            rank_data.append({'cve': cve, 'rank': rank, 'label': label})
        
        # Compute metrics using the ranks
        positive_ranks = [rank for rank, label in ranks if label == 1]
        
        for k in k_values:
            top_k_counts = sum(1 for rank in positive_ranks if rank <= k)
            recalls[k].append(top_k_counts / len(positive_ranks) if positive_ranks else 0)

            if positive_ranks:
                min_rank_within_k = min([rank for rank in positive_ranks if rank < k], default=k)
            else:
                min_rank_within_k = k
            
            manual_efforts[k].append(min_rank_within_k)
            
            manual_efforts_sum[k] += min_rank_within_k
            manual_efforts_count[k] += 1
            
            # effort_k = sum(min(rank, k) for rank in ranks) / len(ranks) if ranks else 0
            # manual_efforts[k].append(effort_k)


        reciprocal_ranks = [1 / (rank) for rank in positive_ranks]
        mrrs.append(sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0)

    avg_recalls = {k: sum(recalls[k]) / len(recalls[k]) if recalls[k] else 0 for k in k_values}
    avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0
    avg_manual_efforts = {k: sum(manual_efforts[k]) / len(manual_efforts[k]) if manual_efforts[k] else 0 for k in k_values}
    avg_manual_efforts[1] = 1.0
    
    if os.path.exists(rank_output_path):
        os.rename(rank_output_path, rank_output_path + '.bak')
    
    df = pd.DataFrame(rank_data)
    df.to_csv(rank_output_path, index=False)
    
    
    
    return avg_recalls, avg_mrr, avg_manual_efforts

def evaluate(model,
             testing_loader,
             k_values,
             reload_from_checkpoint=False,
             load_path_checkpoint=None,
             data_path='/mnt/local/Baselines_Bugs/PatchFinder/metrics/CR_0831/predict.csv',
             rank_info_path='/mnt/local/Baselines_Bugs/PatchFinder/metrics/CR_0831/rank_info.csv'):
    
    """Evaluate the model on the given test loader."""
    if reload_from_checkpoint:
        load_checkpoint(load_path_checkpoint, model)

    model.eval()
    cve_data = {}
    
    if os.path.exists(data_path):
        os.rename(data_path, data_path + '.bak')

    with torch.no_grad():
        for _, batch in tqdm(enumerate(testing_loader, 0), total=len(testing_loader)):
            output = evaluate_single_batch(model, batch)
            for cve_i, output_i, label_i in zip(batch['cve'], output, batch['label']):
                cve_data.setdefault(cve_i, []).append((output_i.item(), label_i.item()))
                save_outputs_to_csv(cve_i, output_i.item(), label_i.item(), data_path)
        
    avg_recalls, avg_mrr, avg_manual_efforts = compute_metrics(cve_data, k_values, rank_info_path)

    for k in k_values:
        logging.info(f'Average Top@{k} recall: {avg_recalls[k]:.4f}')
        logging.info(f'Manual Efforts@{k}: {avg_manual_efforts[k]:.4f}')
    logging.info(f'Average MRR: {avg_mrr:.4f}')
    
    return avg_recalls, avg_mrr, avg_manual_efforts


def save_metrics_to_csv(avg_recalls, avg_mrr, manual_efforts, save_path):
    """Save recall, MRR, and manual efforts to a CSV file."""
    data = {
        'k': list(avg_recalls.keys()),
        'recall': [avg_recalls[k] for k in avg_recalls],
        'manual_effort': [manual_efforts[k] for k in avg_recalls],
        'MRR': [avg_mrr for _ in avg_recalls]
    }
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)




if __name__ == "__main__":

    MODEL_PATH = "/mnt/local/Baselines_Bugs/PatchFinder/metrics/CR_0831/Checkpoints/final_model.pt"
    # MODEL_PATH = "/mnt/local/Baselines_Bugs/PatchFinder/Phase-2/output_1004/Checkpoints/final_model.pt"  
    # MODEL_PATH = "//mnt/local/Baselines_Bugs/PatchFinder/Phase-2/output_1004_20epoch/Checkpoints/final_model.pt"
    # MODEL_PATH = "/mnt/local/Baselines_Bugs/PatchFinder/Phase-2/output_1007_20epoch/Checkpoints/final_model.pt" 
    
    test_file_fusion = '/mnt/local/Baselines_Bugs/CR_score/evaluate/fusion_unnormalized/top_100_fusion.csv'
    
    tmp_df = pd.read_csv(test_file_fusion)
    print(tmp_df.shape)
    unique_cve = tmp_df['cve'].unique()
    print(len(unique_cve))
    
    test_data = CVEDataset(test_file_fusion)  
    
    # test_data = CVEDataset(configs.test_file)
    test_dataloader = DataLoader(test_data, batch_size=32, num_workers=10)
    
    model = torch.load(MODEL_PATH)
    k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 100]
    
    logging.info(f'Evaluating model at {MODEL_PATH}:')
    
    data_path_flag = MODEL_PATH.split('/')[-1].split('.')[0]
    logging.info(f'data_path_flag: {data_path_flag}')
    
    # os.makedirs(f'/mnt/local/Baselines_Bugs/PatchFinder/metrics/CR_1007_20epoch', exist_ok=True)
    
    recalls, avg_mrr, manual_efforts = evaluate(
        model, 
        test_dataloader, 
        k_values, 
        reload_from_checkpoint=True,
        load_path_checkpoint=MODEL_PATH,
        # data_path=f'/mnt/local/Baselines_Bugs/PatchFinder/metrics/CR_0831/predict_{data_path_flag}.csv'
        
        # data_path=f'/mnt/local/Baselines_Bugs/PatchFinder/metrics/CR_1004_20epoch/predict_{data_path_flag}.csv',
        # rank_info_path=f'/mnt/local/Baselines_Bugs/PatchFinder/metrics/CR_1004_20epoch/rank_info_{data_path_flag}.csv'
        
        # data_path=f'/mnt/local/Baselines_Bugs/PatchFinder/metrics/CR_1007_20epoch/predict_{data_path_flag}.csv',
        # rank_info_path=f'/mnt/local/Baselines_Bugs/PatchFinder/metrics/CR_1007_20epoch/rank_info_{data_path_flag}.csv'
        
        data_path=f'/mnt/local/Baselines_Bugs/PatchFinder/metrics/CR_0831/2024_predict_{data_path_flag}.csv',
        rank_info_path=f'/mnt/local/Baselines_Bugs/PatchFinder/metrics/CR_0831/2024_rank_info_{data_path_flag}.csv'
        
        
        )

    # Save metrics
    # metrics_save_path = f'/mnt/local/Baselines_Bugs/PatchFinder/metrics/CR_0831/metrics_{data_path_flag}.csv'
    # metrics_save_path = f'/mnt/local/Baselines_Bugs/PatchFinder/metrics/CR_1004_20epoch/metrics_{data_path_flag}.csv'
    
    # metrics_save_path = f'/mnt/local/Baselines_Bugs/PatchFinder/metrics/CR_1007_20epoch/metrics_{data_path_flag}.csv'
    
    # metrics_save_path = f'/mnt/local/Baselines_Bugs/PatchFinder/metrics/CR_0831/1210metrics_{data_path_flag}.csv'
    metrics_save_path = f'/mnt/local/Baselines_Bugs/PatchFinder/metrics/CR_0831/2024_metrics_{data_path_flag}.csv'
    
    save_metrics_to_csv(recalls, avg_mrr, manual_efforts, metrics_save_path)

