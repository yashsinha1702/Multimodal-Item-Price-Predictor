# --- Full Script for CLIP-Large Price Prediction (with Transformer Fusion Head) ---
# Description:
# This script uses a powerful dual-encoder architecture with CLIP-Large. The encoders
# are frozen. A Transformer Fusion Head is trained on top of the extracted features
# to learn deep interactions between modalities before predicting the price.
#
# Requirements:
# pip install --upgrade transformers torch torchvision Pillow pandas tqdm scikit-learn accelerate
#
import os
import json
import re
import csv
import glob
from multiprocessing import Pool
import multiprocessing as mp
from tqdm import tqdm
from itertools import chain

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import CLIPModel, CLIPProcessor


# --- 1. Configuration ---
CONFIG = {
    "csv_file_path": "/home/abrol/khushal/Misc/Amazon_Challenge/Yash_EXp/student_resource/dataset/train.csv",
    "image_directory": "/home/abrol/khushal/Misc/Amazon_Challenge/Yash_EXp/student_resource/images",
    "output_jsonl_path": "training_dataset_clip_large.jsonl",
    "model_name": "openai/clip-vit-large-patch14",
    "epochs": 10,
    "batch_size": 16,
    "learning_rate": 5e-5,
    "device": "cuda:3" if torch.cuda.is_available() else "cpu",
    "num_workers": 4,
    "checkpoint_dir": "clip_large_fusion_checkpoints_224", # New directory for the fusion model
    "gradient_accumulation_steps": 4,
    "resume_from_checkpoint": True,
}


# --- Custom Loss & Metric Functions (Unchanged) ---
def sMAPE(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    y_true_np = y_true.detach().cpu().to(torch.float32).numpy()
    y_pred_np = y_pred.detach().cpu().to(torch.float32).numpy()
    
    y_pred_np = np.maximum(y_pred_np, 0)
    numerator = np.abs(y_pred_np - y_true_np)
    denominator = (np.abs(y_true_np) + np.abs(y_pred_np)) / 2
    ratio = np.where(denominator == 0, 0, numerator / denominator)
    return np.mean(ratio) * 100


# --- 2. Data Preparation (Unchanged) ---
def create_vlm_prompt(catalog_content: str) -> str:
    title, ipq, unit, value, description = "N/A", 1, "N/A", "N/A", ""
    remaining_content = str(catalog_content)
    title_match = re.search(r"Item Name:\s*(.*)", remaining_content, re.IGNORECASE)
    if title_match:
        title = title_match.group(1).strip()
    else:
        lines = [line.strip() for line in remaining_content.split('\n') if line.strip()]
        if lines:
            title = lines[0]
            remaining_content = remaining_content.replace(title, "", 1)
    quantity_patterns = {
        'ipq': [r"Item Pack Quantity:\s*(\d+)", r"IPQ:\s*(\d+)", r"Pack of\s*(\d+)", r"(\d+)\s*Count"],
        'unit': [r"Unit:\s*(.*)"],
        'value': [r"Value:\s*([\d\.]+)", r"Size:\s*([\d\.]+.*)"]
    }
    for key, patterns in quantity_patterns.items():
        for pattern in patterns:
            match = re.search(pattern, remaining_content, re.IGNORECASE)
            if match:
                extracted_value = match.group(1).strip()
                if key == 'ipq':
                    try: ipq = int(extracted_value)
                    except ValueError: ipq = 1
                elif key == 'unit': unit = extracted_value
                elif key == 'value': value = extracted_value
                remaining_content = re.sub(pattern, "", remaining_content, count=1, flags=re.IGNORECASE)
                break
    description = "\n".join(line.strip() for line in remaining_content.split('\n') if line.strip())
    prompt = (
        f"Title: {title}. "
        f"Pack Quantity: {ipq}. "
        f"Unit: {unit}. "
        f"Value: {value}. "
        f"Description: {description}"
    )
    return prompt

def process_row(args):
    row, img_dir, row_idx = args
    try:
        sample_id, cat_content, price = row.get('sample_id'), row.get('catalog_content'), row.get('price')
        if not all([sample_id, cat_content, price]): return None
        img_path = os.path.join(img_dir, f"{sample_id}.jpg")
        if not os.path.exists(img_path): return None
        return {"id": row_idx, "image_path": img_path, "prompt": create_vlm_prompt(cat_content), "price": float(price)}
    except Exception:
        return None

def generate_dataset_jsonl(config):
    output_path = config["output_jsonl_path"]
    if os.path.exists(output_path):
        print(f"Dataset file already exists at '{output_path}'. Skipping generation.")
        return
    print("Generating dataset JSONL file...")
    csv_path, img_dir = config["csv_file_path"], config["image_directory"]
    processed_rows, skipped_rows = 0, 0
    try:
        with open(csv_path, mode='r', encoding='utf-8') as infile, \
             open(output_path, mode='w', encoding='utf-8') as outfile:
            reader = list(csv.DictReader(infile))
            args_list = [(row, img_dir, i) for i, row in enumerate(reader)]
            with Pool(processes=max(1, config["num_workers"])) as pool:
                for result in tqdm(pool.imap(process_row, args_list), total=len(args_list), desc="Processing rows"):
                    if result is not None:
                        outfile.write(json.dumps(result) + '\n')
                        processed_rows += 1
                    else:
                        skipped_rows += 1
        print(f"Successfully converted {processed_rows} rows. Skipped {skipped_rows} rows.")
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found."); raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}"); raise


# --- 3. Custom Model, Dataset, and Collator (MODIFIED) ---
class CLIPForPricePrediction(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.base_model = CLIPModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        
        # The projection_dim for clip-large is 768
        embedding_dim = self.base_model.projection_dim
        
        # --- NEW: TRANSFORMER FUSION HEAD ---
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=12,  # 768 is divisible by 12
            dim_feedforward=embedding_dim * 4,
            dropout=0.2,
            activation='gelu',
            batch_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(transformer_layer, num_layers=2)

        self.regression_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim // 2, 1)
        )
        
        self.fusion_transformer.to(dtype=torch.bfloat16)
        self.regression_head.to(dtype=torch.bfloat16)
        # ------------------------------------

    def forward(self, **kwargs):
        # 1. Get base features from frozen CLIP encoders
        image_features = self.base_model.get_image_features(pixel_values=kwargs['pixel_values'])
        text_features = self.base_model.get_text_features(
            input_ids=kwargs['input_ids'],
            attention_mask=kwargs['attention_mask']
        )
        
        # --- NEW: FUSION LOGIC ---
        # 2. Prepare features as a sequence for the transformer
        image_features_seq = image_features.unsqueeze(1)
        text_features_seq = text_features.unsqueeze(1)
        combined_sequence = torch.cat([image_features_seq, text_features_seq], dim=1)
        
        # 3. Pass through the fusion transformer for cross-modal attention
        fused_features = self.fusion_transformer(combined_sequence)
        
        # 4. Use the output of the first token as the final fused representation
        fused_output = fused_features[:, 0, :]
        
        # 5. Predict the price
        return self.regression_head(fused_output)

class PricePredictionDataset(Dataset):
    def __init__(self, jsonl_path):
        self.data = [json.loads(line) for line in open(jsonl_path, 'r', encoding='utf-8')]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            image = Image.open(item['image_path']).convert('RGB')
        except (OSError, IOError) as e:
            print(f"WARNING: Corrupted image file at {item['image_path']}. Loading next sample.")
            return self.__getitem__((idx + 1) % len(self))
        return item['prompt'], image, torch.tensor(item['price'], dtype=torch.bfloat16)

class DataCollator:
    def __init__(self, processor, device):
        self.processor = processor
        self.device = device

    def __call__(self, batch):
        prompts, images, prices = zip(*batch)
        inputs = self.processor(
            text=list(prompts),
            images=list(images),
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        return inputs, torch.tensor(prices, dtype=torch.bfloat16).to(self.device)

def find_latest_checkpoint(checkpoint_dir):
    list_of_files = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
    if not list_of_files: return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


# --- 4. Training Loop (MODIFIED) ---
def train_model(config):
    print(f"--- Starting Training with CLIP-Large ({config['model_name']}) and Transformer FUSION HEAD ---")
    generate_dataset_jsonl(config)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    
    model = CLIPForPricePrediction(config['model_name']).to(config['device'])
    processor = CLIPProcessor.from_pretrained(config['model_name'])
    
    print("Freezing base CLIP model parameters...")
    for param in model.base_model.parameters():
        param.requires_grad = False
    
    # --- MODIFIED: Define trainable parameters for the new fusion head ---
    trainable_params_list = chain(model.fusion_transformer.parameters(), model.regression_head.parameters())
    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (Fusion Head): {trainable_params_count:,}")

    train_dataset = PricePredictionDataset(config['output_jsonl_path'])
    collator = DataCollator(processor, config['device'])
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collator,
        num_workers=config['num_workers'], persistent_workers=(config['num_workers'] > 0)
    )
    
    # --- MODIFIED: Optimizer now targets the new trainable parameters ---
    optimizer = torch.optim.AdamW(trainable_params_list, lr=config['learning_rate'])
    criterion = nn.MSELoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * config['epochs'])
    
    start_epoch = 0
    if config['resume_from_checkpoint']:
        latest_checkpoint = find_latest_checkpoint(config['checkpoint_dir'])
        if latest_checkpoint:
            print(f"Resuming training from checkpoint: {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint, map_location=config['device'])
            # --- MODIFIED: Load state dicts for both parts of the fusion head ---
            model.fusion_transformer.load_state_dict(checkpoint['fusion_transformer_state_dict'])
            model.regression_head.load_state_dict(checkpoint['regression_head_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch + 1}")
        else:
            print("No checkpoint found. Starting training from scratch.")
            
    for epoch in range(start_epoch, config['epochs']):
        print(f"\n--- Epoch {epoch+1}/{config['epochs']} ---")
        model.train()
        total_train_loss = 0
        optimizer.zero_grad() # Moved for cleaner gradient accumulation logic
        
        train_progress = tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}", total=len(train_loader))
        
        for step, batch in train_progress:
            inputs, prices = batch
            predicted_prices = model(**inputs).squeeze(-1)
            
            loss = criterion(predicted_prices, prices)
            batch_smape = sMAPE(prices, predicted_prices)
            
            loss = loss / config['gradient_accumulation_steps']
            loss.backward()
            
            if (step + 1) % config['gradient_accumulation_steps'] == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_train_loss += loss.item() * config['gradient_accumulation_steps']
            train_progress.set_postfix({
                'Loss': loss.item() * config['gradient_accumulation_steps'],
                'sMAPE': f"{batch_smape:.2f}%"
            })

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Average Training Loss for Epoch {epoch+1}: {avg_train_loss:.4f}")
        
        epoch_save_path = os.path.join(config['checkpoint_dir'], f"clip_large_fusion_epoch_{epoch+1}.pth")
        print(f"Saving model head at end of epoch {epoch+1} to {epoch_save_path}")
        # --- MODIFIED: Save state dicts for both parts of the new fusion head ---
        torch.save({
            'epoch': epoch,
            'fusion_transformer_state_dict': model.fusion_transformer.state_dict(),
            'regression_head_state_dict': model.regression_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, epoch_save_path)

    print(f"\n--- Training Complete ---")

if __name__ == '__main__':
    if 'cuda' in CONFIG['device'] and CONFIG['num_workers'] > 0:
        mp.set_start_method('spawn', force=True)
    train_model(CONFIG)