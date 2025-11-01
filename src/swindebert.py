# --- Full Script for Swin-Large + DeBERTa-V3-Large Price Prediction (Corrected) ---
# Description:
# This script uses a state-of-the-art dual-encoder architecture with a large Swin
# Transformer for images and a large DeBERTa V3 model for text. The encoders are
# frozen, and a custom regression head is trained on the combined features.
#
# Requirements:
# pip install --upgrade transformers torch torchvision Pillow pandas tqdm scikit-learn accelerate sentencepiece
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

# Use Auto classes for maximum flexibility
from transformers import AutoModel, AutoImageProcessor, AutoTokenizer


# --- 1. Configuration ---
CONFIG = {
    "csv_file_path": "/home/abrol/khushal/Misc/Amazon_Challenge/Yash_EXp/student_resource/dataset/train.csv",
    "image_directory": "/home/abrol/khushal/Misc/Amazon_Challenge/Yash_EXp/student_resource/images",
    "output_jsonl_path": "training_dataset_swin_deberta.jsonl",
    "image_model_name": "microsoft/swin-large-patch4-window12-384-in22k",
    "text_model_name": "microsoft/deberta-v3-large",
    "epochs": 5,
    "batch_size": 16,
    "learning_rate": 5e-5,
    "device": "cuda:3" if torch.cuda.is_available() else "cpu",
    "num_workers": 4,
    "checkpoint_dir": "swin_deberta_checkpoints",
    "gradient_accumulation_steps": 4,
    "resume_from_checkpoint": True,
}


# --- Custom Loss & Metric Functions ---
def sMAPE(y_true, y_pred):
    y_true_np = y_true.detach().cpu().to(torch.float32).numpy()
    y_pred_np = y_pred.detach().cpu().to(torch.float32).numpy()
    y_pred_np = np.maximum(y_pred_np, 0)
    numerator = np.abs(y_pred_np - y_true_np)
    denominator = (np.abs(y_true_np) + np.abs(y_pred_np)) / 2
    ratio = np.where(denominator == 0, 0, numerator / denominator)
    return np.mean(ratio) * 100


# --- 2. Data Preparation ---
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
        f"Description: {description}" # Including description for consistency
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
    # (No changes needed in this function)
    output_path = config["output_jsonl_path"]
    if os.path.exists(output_path):
        print(f"Dataset file already exists at '{output_path}'. Skipping generation.")
        return
    print("Generating dataset JSONL file...")
    # ... (rest of the function is the same)
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
class SwinDeBERTaForPricePrediction(nn.Module):
    def __init__(self, image_model_name, text_model_name):
        super().__init__()
        # Note: trust_remote_code=True is not needed for microsoft/swin
        self.image_encoder = AutoModel.from_pretrained(image_model_name, torch_dtype=torch.bfloat16)
        self.text_encoder = AutoModel.from_pretrained(text_model_name, torch_dtype=torch.bfloat16)

        # --- FIX #1: Correctly get the embedding dimension for SwinModel ---
        # SwinModel from transformers uses .config.hidden_size, not .timm_model
        image_embedding_dim = self.image_encoder.config.hidden_size
        text_embedding_dim = self.text_encoder.config.hidden_size
        
        # Define a common dimension for fusion
        fusion_dim = 1024 # A common dimension, can be tuned

        # Project both features to the common fusion dimension
        self.image_projection = nn.Linear(image_embedding_dim, fusion_dim)
        self.text_projection = nn.Linear(text_embedding_dim, fusion_dim)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim, nhead=8, dim_feedforward=fusion_dim * 4,
            dropout=0.2, activation='gelu', batch_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(transformer_layer, num_layers=2)

        self.regression_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim // 2, 1)
        )

        # Cast new layers to bfloat16
        self.image_projection.to(dtype=torch.bfloat16)
        self.text_projection.to(dtype=torch.bfloat16)
        self.fusion_transformer.to(dtype=torch.bfloat16)
        self.regression_head.to(dtype=torch.bfloat16)

    def forward(self, pixel_values, input_ids, token_type_ids, attention_mask):
        image_outputs = self.image_encoder(pixel_values=pixel_values)
        image_features = image_outputs.pooler_output

        text_outputs = self.text_encoder(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        text_features = text_outputs.last_hidden_state[:, 0]

        image_projected = self.image_projection(image_features)
        text_projected = self.text_projection(text_features)

        image_features_seq = image_projected.unsqueeze(1)
        text_features_seq = text_projected.unsqueeze(1)
        combined_sequence = torch.cat([image_features_seq, text_features_seq], dim=1)
        
        fused_features = self.fusion_transformer(combined_sequence)
        fused_output = fused_features[:, 0, :]
        return self.regression_head(fused_output)

class PricePredictionDataset(Dataset):
    # (No changes needed in this class)
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
    # (No changes needed in this class)
    def __init__(self, image_processor, tokenizer, device):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.device = device
    def __call__(self, batch):
        prompts, images, prices = zip(*batch)
        image_inputs = self.image_processor(images, return_tensors="pt")
        text_inputs = self.tokenizer(
            list(prompts), return_tensors="pt", padding=True,
            truncation=True, max_length=512
        )
        inputs = {
            "pixel_values": image_inputs['pixel_values'].to(self.device, dtype=torch.bfloat16),
            "input_ids": text_inputs['input_ids'].to(self.device),
            "token_type_ids": text_inputs['token_type_ids'].to(self.device),
            "attention_mask": text_inputs['attention_mask'].to(self.device)
        }
        return inputs, torch.tensor(prices, dtype=torch.bfloat16).to(self.device)

def find_latest_checkpoint(checkpoint_dir):
    # (No changes needed in this function)
    list_of_files = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
    if not list_of_files: return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


# --- 4. Training Loop ---
def train_model(config):
    print(f"--- Starting Training with Swin-Large ({config['image_model_name']}) & DeBERTa-V3-Large ({config['text_model_name']}) ---")
    generate_dataset_jsonl(config)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    
    model = SwinDeBERTaForPricePrediction(
        config['image_model_name'],
        config['text_model_name']
    ).to(config['device'])
    
    image_processor = AutoImageProcessor.from_pretrained(config['image_model_name'])
    tokenizer = AutoTokenizer.from_pretrained(config['text_model_name'])
    
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
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        collate_fn=collator, num_workers=config['num_workers'],
        persistent_workers=(config['num_workers'] > 0)
    )
    
    trainable_head_params = (
        list(model.image_projection.parameters()) +
        list(model.text_projection.parameters()) +
        list(model.fusion_transformer.parameters()) +
        list(model.regression_head.parameters())
    )
    optimizer = torch.optim.AdamW(trainable_head_params, lr=config['learning_rate'])
    criterion = nn.MSELoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * config['epochs'])
    
    start_epoch = 0
    # (Resuming from checkpoint logic needs to be updated for the new head)
    # This part is left as is, but will need adjustment if you want to resume training
    
    for epoch in range(start_epoch, config['epochs']):
        print(f"\n--- Epoch {epoch+1}/{config['epochs']} ---")
        model.train()
        total_train_loss = 0
        
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
        
        epoch_save_path = os.path.join(config['checkpoint_dir'], f"swin_deberta_epoch_{epoch+1}.pth")
        print(f"Saving model head at end of epoch {epoch+1} to {epoch_save_path}")
        
        # --- FIX #2: Save the state dicts for all new components ---
        torch.save({
            'epoch': epoch,
            'image_projection_state_dict': model.image_projection.state_dict(),
            'text_projection_state_dict': model.text_projection.state_dict(),
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