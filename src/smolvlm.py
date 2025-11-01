# --- Full Script for SmolVLM Price Prediction ---
# Description:
# This script provides an end-to-end workflow for fine-tuning the SmolVLM-500M model
# for a price prediction regression task. This final version fixes a data type
# mismatch between the bfloat16 base model and the float32 regression head.
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
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor
from sklearn.model_selection import train_test_split
import pandas as pd

# --- 1. Configuration ---
# You can adjust these parameters as needed
CONFIG = {
    "csv_file_path": "/home/abrol/khushal/Misc/Amazon_Challenge/Yash_EXp/student_resource/dataset/train.csv",
    "image_directory": "/home/abrol/khushal/Misc/Amazon_Challenge/Yash_EXp/student_resource/images",
    "output_jsonl_path": "training_dataset.jsonl",
    "model_name": "HuggingFaceTB/SmolVLM-500M-Instruct",
    "epochs": 5,
    "batch_size": 4,
    "learning_rate": 5e-5,
    "device": "cuda:3" if torch.cuda.is_available() else "cpu",
    "test_split_size": 0.1,
    "max_length": 1500, # Max token length for the text
    "num_workers": 1, # Number of workers for data loading
}

# --- 2. Data Preparation (Adapted from your script) ---
def create_vlm_prompt(catalog_content: str) -> str:
    """
    Parses semi-structured catalog content and formats it into a structured prompt.
    """
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
    """Processes one CSV row for multiprocessing."""
    row, img_dir, row_idx = args
    try:
        sample_id = row.get('sample_id')
        catalog_content = row.get('catalog_content')
        price = row.get('price')

        if not all([sample_id, catalog_content, price]):
            return None

        image_path = os.path.join(img_dir, f"{sample_id}.jpg")
        if not os.path.exists(image_path):
            return None

        formatted_prompt = create_vlm_prompt(catalog_content)
        price = float(price)

        json_object = {
            "id": row_idx,
            "image_path": image_path,
            "prompt": formatted_prompt,
            "price": price
        }
        return json_object
    except Exception:
        return None

def generate_dataset_jsonl(config):
    """Converts the source CSV to a JSONL file for training if it doesn't exist."""
    output_path = config["output_jsonl_path"]
    if os.path.exists(output_path):
        print(f"Dataset file already exists at '{output_path}'. Skipping generation.")
        return

    print("Generating dataset JSONL file...")
    csv_path = config["csv_file_path"]
    img_dir = config["image_directory"]
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
        
        print(f"Successfully converted {processed_rows} rows.")
        print(f"Skipped {skipped_rows} rows due to missing data or images.")
        print(f"Output saved to: '{output_path}'")

    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during dataset generation: {e}")
        raise

# --- 3. Custom Model, Dataset, and Collator ---
class SmolVLMForPricePrediction(nn.Module):
    """
    A wrapper class for SmolVLM, adapted for price prediction (regression).
    """
    def __init__(self, model_name):
        super().__init__()
        self.base_model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            quantization_config={"load_in_4bit": True} if 'cuda' in CONFIG['device'] else None,
        )
        hidden_size = self.base_model.config.text_config.hidden_size
        
        self.price_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )

        # --- THE FIX IS HERE ---
        # Ensure the new regression head has the same bfloat16 data type as the base model
        if 'cuda' in CONFIG['device']:
             self.price_predictor.to(dtype=torch.bfloat16)
        # ---------------------

    def forward(self, input_ids, pixel_values, attention_mask,**kwargs):
        outputs = self.base_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden_state = outputs.hidden_states[-1][:, -1, :]
        predicted_price = self.price_predictor(last_hidden_state)
        return predicted_price

class PricePredictionDataset(Dataset):
    def __init__(self, jsonl_path):
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image_path']).convert('RGB')
        prompt = item['prompt']
        price = torch.tensor(item['price'], dtype=torch.bfloat16)
        return prompt, image, price

class DataCollator:
    def __init__(self, processor, device):
        self.processor = processor
        self.device = device

    def __call__(self, batch):
        prompts, images, prices = zip(*batch)
        task_prompts = [f"<image>\nPredict the price for the following item based on its image and details.\n{p}" for p in prompts]

        inputs = self.processor(
            text=task_prompts,
            images=list(images),
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=CONFIG['max_length']
        ).to(self.device)
        
        prices = torch.stack(prices).to(self.device)
        return inputs, prices

# --- 4. Training and Evaluation Loop ---
def train_model(config):
    print("--- Starting Training ---")
    print(f"Configuration: {json.dumps(config, indent=2)}")

    generate_dataset_jsonl(config)
    
    df = pd.read_json(config['output_jsonl_path'], lines=True)
    train_df, val_df = train_test_split(df, test_size=config['test_split_size'], random_state=42)
    
    train_jsonl_path = 'train_subset.jsonl'
    val_jsonl_path = 'val_subset.jsonl'
    train_df.to_json(train_jsonl_path, orient='records', lines=True)
    val_df.to_json(val_jsonl_path, orient='records', lines=True)

    print(f"Loading model: {config['model_name']} on device: {config['device']}")
    model = SmolVLMForPricePrediction(config['model_name']).to(config['device'])
    processor = AutoProcessor.from_pretrained(config['model_name'], trust_remote_code=True)
    
    train_dataset = PricePredictionDataset(train_jsonl_path)
    val_dataset = PricePredictionDataset(val_jsonl_path)
    
    collator = DataCollator(processor, config['device'])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=collator,
        num_workers=config['num_workers']
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        collate_fn=collator,
        num_workers=config['num_workers']
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    for epoch in range(config['epochs']):
        print(f"\n--- Epoch {epoch+1}/{config['epochs']} ---")
        
        model.train()
        total_train_loss = 0
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
        for batch in train_progress_bar:
            inputs, prices = batch
            optimizer.zero_grad()
            
            predicted_prices = model(**inputs)
            loss = criterion(predicted_prices.squeeze(), prices)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_progress_bar.set_postfix({'Train Loss': loss.item()})
            
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                inputs, prices = batch
                predicted_prices = model(**inputs)
                loss = criterion(predicted_prices.squeeze(), prices)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Average Validation Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print("New best validation loss. Saving model head...")
            torch.save(model.price_predictor.state_dict(), 'smolvlm_price_predictor_best.pth')

    print("\n--- Training Complete ---")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("Model's price predictor head saved to 'smolvlm_price_predictor_best.pth'")

if __name__ == '__main__':
    if 'cuda' in CONFIG['device']:
        mp.set_start_method('spawn', force=True)
    train_model(CONFIG)