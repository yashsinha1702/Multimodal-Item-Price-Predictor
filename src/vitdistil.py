# --- Full Script for ViT + DistilBERT Price Prediction ---
# Description:
# This script uses an efficient dual-encoder architecture with a Vision Transformer (ViT)
# for images and a fast DistilBERT model for text. The encoders are frozen, and a
# custom regression head is trained on the combined features to predict product prices.
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

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- CRITICAL CHANGE: Use DistilBertTokenizer ---
from transformers import AutoModel, ViTImageProcessor, DistilBertTokenizer


# --- 1. Configuration ---
CONFIG = {
    "csv_file_path": "/home/abrol/khushal/Misc/Amazon_Challenge/Yash_EXp/student_resource/dataset/train.csv",
    "image_directory": "/home/abrol/khushal/Misc/Amazon_Challenge/Yash_EXp/student_resource/images",
    # --- MODEL CHANGE ---
    "output_jsonl_path": "training_dataset_vit_distilbert.jsonl",
    "image_model_name": "google/vit-base-patch16-224-in21k",
    "text_model_name": "distilbert-base-uncased",
    "epochs": 10,
    "batch_size": 48, # Increased batch size due to smaller text model
    "learning_rate": 1e-4,
    "device": "cuda:3" if torch.cuda.is_available() else "cpu",
    "num_workers": 4,
    # --- MODEL CHANGE ---
    "checkpoint_dir": "vit_distilbert_checkpoints_224",
    "gradient_accumulation_steps": 2,
    "resume_from_checkpoint": True,
}


# --- Custom Loss & Metric Functions ---
def sMAPE(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    y_true_np = y_true.detach().cpu().to(torch.float32).numpy()
    y_pred_np = y_pred.detach().cpu().to(torch.float32).numpy()
    
    y_pred_np = np.maximum(y_pred_np, 0)
    numerator = np.abs(y_pred_np - y_true_np)
    denominator = (np.abs(y_true_np) + np.abs(y_pred_np)) / 2
    ratio = np.where(denominator == 0, 0, numerator / denominator)
    return np.mean(ratio) * 100


# --- 2. Data Preparation ---
def create_vlm_prompt(catalog_content: str) -> str:
    # This function is model-agnostic and does not need changes.
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


# --- 3. Custom Model, Dataset, and Collator ---
class ViTDistilBERTForPricePrediction(nn.Module):
    def __init__(self, image_model_name, text_model_name):
        super().__init__()
        self.image_encoder = AutoModel.from_pretrained(image_model_name, torch_dtype=torch.bfloat16)
        self.text_encoder = AutoModel.from_pretrained(text_model_name, torch_dtype=torch.bfloat16)

        image_embedding_dim = self.image_encoder.config.hidden_size # 768 for vit-base
        text_embedding_dim = self.text_encoder.config.hidden_size   # 768 for distilbert-base
        combined_dim = image_embedding_dim + text_embedding_dim

        self.price_predictor = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(combined_dim // 2, 1),
        )
        self.price_predictor.to(dtype=torch.bfloat16)

    def forward(self, pixel_values, input_ids, attention_mask):
        image_outputs = self.image_encoder(pixel_values=pixel_values)
        image_features = image_outputs.pooler_output

        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # DistilBERT outputs last_hidden_state, we take the [CLS] token's embedding (first token)
        text_features = text_outputs.last_hidden_state[:, 0]

        combined_features = torch.cat([image_features, text_features], dim=1)
        
        return self.price_predictor(combined_features)

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
    def __init__(self, image_processor, tokenizer, device):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, batch):
        prompts, images, prices = zip(*batch)
        
        image_inputs = self.image_processor(images, return_tensors="pt")
        
        text_inputs = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512 # DistilBERT's max length
        )

        inputs = {
            "pixel_values": image_inputs['pixel_values'].to(self.device),
            "input_ids": text_inputs['input_ids'].to(self.device),
            "attention_mask": text_inputs['attention_mask'].to(self.device)
        }
        
        return inputs, torch.tensor(prices, dtype=torch.bfloat16).to(self.device)

def find_latest_checkpoint(checkpoint_dir):
    list_of_files = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
    if not list_of_files: return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


# --- 4. Training Loop ---
def train_model(config):
    print("--- Starting Training with ViT + DistilBERT ---")
    generate_dataset_jsonl(config)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    
    # --- MODEL CHANGE ---
    model = ViTDistilBERTForPricePrediction(
        config['image_model_name'],
        config['text_model_name']
    ).to(config['device'])
    
    # --- TOKENIZER CHANGE ---
    image_processor = ViTImageProcessor.from_pretrained(config['image_model_name'])
    tokenizer = DistilBertTokenizer.from_pretrained(config['text_model_name'])
    
    print("Freezing base encoder parameters...")
    for param in model.image_encoder.parameters():
        param.requires_grad = False
    for param in model.text_encoder.parameters():
        param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    train_dataset = PricePredictionDataset(config['output_jsonl_path'])
    collator = DataCollator(image_processor, tokenizer, config['device'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collator,
        num_workers=config['num_workers'],
        persistent_workers=(config['num_workers'] > 0)
    )
    
    optimizer = torch.optim.AdamW(model.price_predictor.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * config['epochs'])
    
    start_epoch = 0
    if config['resume_from_checkpoint']:
        latest_checkpoint = find_latest_checkpoint(config['checkpoint_dir'])
        if latest_checkpoint:
            print(f"Resuming training from checkpoint: {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint, map_location=config['device'])
            model.price_predictor.load_state_dict(checkpoint['model_state_dict'])
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
        
        train_progress = tqdm(
            enumerate(train_loader),
            desc=f"Epoch {epoch+1}",
            total=len(train_loader)
        )
        
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
        
        # --- FILENAME CHANGE ---
        epoch_save_path = os.path.join(config['checkpoint_dir'], f"vit_distilbert_epoch_{epoch+1}.pth")
        print(f"Saving model head at end of epoch {epoch+1} to {epoch_save_path}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.price_predictor.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, epoch_save_path)

    print(f"\n--- Training Complete ---")

if __name__ == '__main__':
    if 'cuda' in CONFIG['device'] and CONFIG['num_workers'] > 0:
        mp.set_start_method('spawn', force=True)
    train_model(CONFIG)