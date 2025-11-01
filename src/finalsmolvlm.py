# --- Full Script for SmolVLM Price Prediction ---
# Description:
# This version trains on 100% of the data without a validation set.
# It integrates a SMAPE metric for real-time performance tracking and saves a
# model checkpoint at the end of each epoch.
#
# Requirements:
# pip install --upgrade transformers
# pip install torch Pillow pandas tqdm scikit-learn accelerate bitsandbytes
#
import multiprocessing as mp
import os
import json
import re
import csv
from multiprocessing import Pool
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
# --- CRITICAL FIX: Use the correct AutoModel class for this model architecture ---
from transformers import AutoModelForImageTextToText, AutoProcessor
from sklearn.model_selection import train_test_split
import pandas as pd
import glob

# --- 1. Configuration ---
CONFIG = {
    "csv_file_path": "/home/abrol/khushal/Misc/Amazon_Challenge/Yash_EXp/student_resource/dataset/train.csv",
    "image_directory": "/home/abrol/khushal/Misc/Amazon_Challenge/Yash_EXp/student_resource/images",
    "output_jsonl_path": "training_dataset.jsonl",
    "model_name": "HuggingFaceTB/SmolVLM-500M-Instruct",
    "epochs": 5,
    "batch_size": 16,
    "learning_rate": 5e-5,
    "device": "cuda:3" if torch.cuda.is_available() else "cpu",
    "max_length": 2048,
    "num_workers": 4, # Set to 0 to avoid multiprocessing issues if they arise
    "checkpoint_interval": 100,
    "checkpoint_dir": "checkpoints",
    "gradient_accumulation_steps": 2, # Effective batch size = 32 * 2 = 64
    "resume_from_checkpoint": True,
}

# --- Custom Loss & Metric Functions ---
class MSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        pred = torch.clamp(pred, min=0)
        return self.mse(torch.log(pred + 1), torch.log(actual + 1))

def sMAPE(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    # --- THE FIX IS HERE ---
    # Convert bfloat16 tensors to float32 *before* converting to numpy
    y_true_np = y_true.detach().cpu().to(torch.float32).numpy()
    y_pred_np = y_pred.detach().cpu().to(torch.float32).numpy()
    # ---------------------
    
    y_pred_np = np.maximum(y_pred_np, 0)
    numerator = np.abs(y_pred_np - y_true_np)
    denominator = (np.abs(y_true_np) + np.abs(y_pred_np)) / 2
    ratio = np.where(denominator == 0, 0, numerator / denominator)
    return np.mean(ratio) * 100

# --- 2. Data Preparation ---
def create_vlm_prompt(catalog_content: str) -> str:
    # (Your data prep code is good, no changes needed here)
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
        f"Product Information:\n"
        f"- Title: {title}\n"
        f"- Item Pack Quantity: {ipq}\n"
        f"- Unit: {unit}\n"
        f"- Value: {value}"
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
class SmolVLMForPricePrediction(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.base_model = AutoModelForImageTextToText.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
            quantization_config={"load_in_4bit": True} if 'cuda' in CONFIG['device'] else None,
        )
        hidden_size = self.base_model.config.text_config.hidden_size
        self.price_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, 1),
        )
        if 'cuda' in CONFIG['device']: self.price_predictor.to(dtype=torch.bfloat16)

    def forward(self, **kwargs):
        outputs = self.base_model(**kwargs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1][:, -1, :]
        return self.price_predictor(last_hidden_state)

class PricePredictionDataset(Dataset):
    def __init__(self, jsonl_path):
        self.data = [json.loads(line) for line in open(jsonl_path, 'r', encoding='utf-8')]
    def __len__(self):
        return len(self.data)

    # --- THE FIX IS HERE ---
    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            # Attempt to open and load the image
            image = Image.open(item['image_path']).convert('RGB')
            # This line forces the image data to be read. If it's corrupt, it will fail here.
            image.load() 
        except (OSError, IOError) as e:
            # If the image is truncated or corrupt, print a warning and load the next image instead
            print(f"WARNING: Corrupted image file at {item['image_path']}. Skipping and loading next image. Error: {e}")
            # This is a robust way to handle it: recursively call __getitem__ for the next index
            # The modulo operator (%) ensures we wrap around to the start if the last item is corrupt
            return self.__getitem__((idx + 1) % len(self))
            
        return item['prompt'], image, torch.tensor(item['price'], dtype=torch.bfloat16)

class DataCollator:
    def __init__(self, processor, device):
        self.processor = processor
        self.device = device
    def __call__(self, batch):
        prompts, images, prices = zip(*batch)
        task_prompts = [f"<image>\nPredict the price for the item.\n{p}" for p in prompts]
        inputs = self.processor(
            text=task_prompts, images=list(images), return_tensors="pt", padding="longest",
            truncation=True, max_length=CONFIG['max_length']
        ).to(self.device)
        return inputs, torch.stack(prices).to(self.device)

def find_latest_checkpoint(checkpoint_dir):
    list_of_files = glob.glob(os.path.join(checkpoint_dir, 'chkpt_E*_S*.pth'))
    if not list_of_files: return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

# --- 4. Training Loop (100% Data) ---
def train_model(config):
    print("--- Starting Training on 100% of the data ---")
    generate_dataset_jsonl(config)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    
    model = SmolVLMForPricePrediction(config['model_name']).to(config['device'])
    processor = AutoProcessor.from_pretrained(config['model_name'], trust_remote_code=True)
    
    print("Freezing base model parameters for PEFT...")
    for param in model.base_model.parameters():
        param.requires_grad = False
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Create dataset with 100% of the data
    train_dataset = PricePredictionDataset(config['output_jsonl_path'])
    
    collator = DataCollator(processor, config['device'])
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, 
        collate_fn=collator, num_workers=config['num_workers']
    )
    
    optimizer = torch.optim.AdamW(model.price_predictor.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * config['epochs'])
    
    start_epoch, start_step = 0, 0
    if config['resume_from_checkpoint']:
        latest_checkpoint = find_latest_checkpoint(config['checkpoint_dir'])
        if latest_checkpoint:
            print(f"Resuming training from checkpoint: {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint, map_location=config['device'])

            # Determine if checkpoint is a dict with 'model_state_dict' or raw
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint  # assume raw state_dict

            # Map only the keys that exist in price_predictor
            price_predictor_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("price_predictor.") or k in ["0.weight", "0.bias", "3.weight", "3.bias"]:
                    new_key = k.replace("price_predictor.", "") if k.startswith("price_predictor.") else k
                    price_predictor_state_dict[new_key] = v

            # Load into the model
            model.price_predictor.load_state_dict(price_predictor_state_dict)
            if 'model_state_dict' in checkpoint:
                
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            start_epoch = checkpoint['epoch']
            
            start_step = checkpoint['step'] + 1 # Start from the next step
            # start_epoch=0
            # start_step=631
        else:
            print("No checkpoint found. Starting training from scratch.")

    for epoch in range(config['epochs']):
        print(f"\n--- Epoch {epoch+1}/{config['epochs']} ---")
        model.train()
        total_train_loss = 0
        train_progress = tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}", total=len(train_loader))
        
        for step, batch in train_progress:
            if epoch == 0 and step < start_step:
                # Fast-forward the scheduler to the correct step
                for _ in range(step * config['gradient_accumulation_steps'], start_step * config['gradient_accumulation_steps']):
                    scheduler.step()
                continue
            inputs, prices = batch
            predicted_prices = model(**inputs)
            
            # Squeeze predictions to match shape of prices
            predicted_prices_squeezed = predicted_prices.squeeze()
            
            loss = criterion(predicted_prices_squeezed, prices)
            
            # Calculate SMAPE for the current batch
            batch_smape = sMAPE(prices, predicted_prices_squeezed)
            
            # Gradient Accumulation Logic
            loss = loss / config['gradient_accumulation_steps']
            loss.backward()
            
            if (step + 1) % config['gradient_accumulation_steps'] == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_train_loss += loss.item() * config['gradient_accumulation_steps']
            # Update progress bar with both loss and SMAPE
            train_progress.set_postfix({
                'Loss': loss.item() * config['gradient_accumulation_steps'],
                'sMAPE': batch_smape
            })
            
            
            
            if (step + 1) % config['checkpoint_interval'] == 0:
                chkpt_path = os.path.join(config['checkpoint_dir'], f"chkpt_E{epoch+1}_S{step+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'step': step,
                    'model_state_dict': model.price_predictor.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, chkpt_path)
        
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Average Training Loss for Epoch {epoch+1}: {avg_train_loss:.4f}")
        start_step = 0
        # Save model at the end of each epoch
        epoch_save_path = os.path.join(config['checkpoint_dir'], f"smolvlm_epoch_{epoch+1}.pth")
        print(f"Saving model head at end of epoch {epoch+1} to {epoch_save_path}")
        torch.save(model.price_predictor.state_dict(), epoch_save_path)

    print(f"\n--- Training Complete ---")

if __name__ == '__main__':
    if 'cuda' in CONFIG['device'] and CONFIG['num_workers'] > 0:
        mp.set_start_method('spawn', force=True)
    train_model(CONFIG)

