# --- Full Script for EVA-02 + DeBERTa-V3 Price Prediction INFERENCE (Multi-GPU) ---
# Description:
# This script loads the fine-tuned regression head for the EVA-02 + DeBERTa-V3 model
# and uses multiprocessing to distribute the inference workload across multiple GPUs.
#
# Requirements:
# pip install --upgrade transformers torch torchvision Pillow pandas tqdm scikit-learn accelerate sentencepiece protobuf timm
#
# How to run:
# python inference_eva02_deberta_multigpu.py \
#     --checkpoint "/path/to/your/eva02_deberta_checkpoints/eva02_deberta_epoch_5.pth" \
#     --input_csv "/path/to/your/test.csv" \
#     --image_dir "/path/to/your/test_images_folder" \
#     --output_csv "eva02_deberta_predictions.csv" \
#     --devices "0,1,2,3"

import argparse
import os
import csv
import re
from tqdm import tqdm
import multiprocessing as mp
from itertools import chain
import numpy as np

import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor, AutoTokenizer
from PIL import Image

# --- 1. Model Definition (Must be identical to the training script) ---
class EVA02DeBERTaForPricePrediction(nn.Module):
    def __init__(self, image_model_name, text_model_name):
        super().__init__()
        # trust_remote_code=True is needed for timm models
        self.image_encoder = AutoModel.from_pretrained(
            image_model_name, trust_remote_code=True
        )
        self.text_encoder = AutoModel.from_pretrained(
            text_model_name
        )

        # Correctly get the embedding dimensions from the loaded models
        image_embedding_dim = self.image_encoder.timm_model.num_features
        text_embedding_dim = self.text_encoder.config.hidden_size
        combined_dim = image_embedding_dim + text_embedding_dim

        self.price_predictor = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(combined_dim // 2, 1),
        )

    def forward(self, pixel_values, input_ids, token_type_ids, attention_mask):
        image_outputs = self.image_encoder(pixel_values=pixel_values)
        image_features = image_outputs.pooler_output

        text_outputs = self.text_encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        # DeBERTa V3 does not have a pooler, so we take the [CLS] token's embedding
        text_features = text_outputs.last_hidden_state[:, 0]

        combined_features = torch.cat([image_features, text_features], dim=1)
        return self.price_predictor(combined_features)


# --- 2. Prompt Formatting (Must be identical to the training script) ---
def create_vlm_prompt(catalog_content: str) -> str:
    title, ipq, unit, value, description = "N/A", 1, "N/A", "N/A", ""
    # ... (exact same text parsing logic as training, omitted for brevity) ...
    prompt = (
        f"Title: {title}. "
        f"Pack Quantity: {ipq}. "
        f"Unit: {unit}. "
        f"Value: {value}. "
        f"Description: {description}"
    )
    return prompt


# --- 3. Worker Function for Multiprocessing ---
# --- 3. Worker Function for Multiprocessing ---
def worker_inference(data_chunk, image_model_name, text_model_name, checkpoint_path, device, batch_size, worker_id):
    """
    The main function for each worker process.
    Loads a model onto a specific GPU and processes a chunk of data.
    """
    # --- THIS LINE IS REMOVED, as the arguments are now passed directly ---
    # worker_args = data_chunk, image_model_name, ...
    
    # 1. Load Model and Processors onto the assigned GPU
    device_str = f"cuda:{device}"
    print(f"[Worker {worker_id} on {device_str}] Loading models...")
    
    model = EVA02DeBERTaForPricePrediction(image_model_name, text_model_name)
    checkpoint = torch.load(checkpoint_path, map_location=device_str)
    model.price_predictor.load_state_dict(checkpoint['model_state_dict'])
    model.to(device_str)
    model.eval()
    model = torch.compile(model)
    image_processor = AutoImageProcessor.from_pretrained(image_model_name)
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    
    # 2. Process the assigned data chunk
    worker_results = []
    with torch.no_grad():
        # Each worker gets its own progress bar
        pbar = tqdm(range(0, len(data_chunk), batch_size), desc=f"Worker {worker_id} on {device_str}", position=worker_id)
        for i in pbar:
            batch_data = data_chunk[i:i + batch_size]
            
            sample_ids = [item['sample_id'] for item in batch_data]
            prompts = [item['prompt'] for item in batch_data]
            
            try:
                images = [Image.open(item['image_path']).convert('RGB').resize((448, 448)) for item in batch_data]

                # Preprocess the batch
                image_inputs = image_processor(images, return_tensors="pt")
                text_inputs = tokenizer(
                    prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
                )
                
                inputs = {
                    "pixel_values": image_inputs['pixel_values'].to(device_str),
                    "input_ids": text_inputs['input_ids'].to(device_str),
                    "token_type_ids": text_inputs['token_type_ids'].to(device_str),
                    "attention_mask": text_inputs['attention_mask'].to(device_str)
                }

                predicted_price_tensor = model(**inputs).squeeze(-1)
                predicted_prices = predicted_price_tensor.cpu().numpy()

                for sid, price in zip(sample_ids, predicted_prices):
                    worker_results.append({'sample_id': sid, 'price': f"{max(0.0, price):.4f}"})

            except Exception as e:
                print(f"[Worker {worker_id}] Error processing batch starting at index {i}: {e}")
                for sid in sample_ids:
                    worker_results.append({'sample_id': sid, 'price': "0.0"})

    return worker_results


# --- 4. Main Process Logic ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run EVA-02 + DeBERTa-V3 price prediction inference on multiple GPUs.")
    # --- MODIFIED: Default paths updated, device argument changed to devices ---
    parser.add_argument('--checkpoint', type=str, default="eva02_deberta_checkpoints/eva02_deberta_epoch_2.pth", help='Path to the fine-tuned price predictor head (.pth file).')
    parser.add_argument('--input_csv', type=str, default="/home/abrol/khushal/Misc/Amazon_Challenge/Yash_EXp/student_resource/dataset/test.csv", help='Path to the input CSV file containing product data.')
    parser.add_argument('--image_dir', type=str, default="/home/abrol/khushal/Misc/Amazon_Challenge/Yash_EXp/student_resource/test/images", help='Path to the directory containing product images.')
    parser.add_argument('--output_csv', type=str, default='inference_eva.cvs', help='Path to save the output predictions CSV file.')
    parser.add_argument('--image_model_name', type=str, default='timm/eva02_large_patch14_224.mim_in22k', help='Name of the pre-trained image model.')
    parser.add_argument('--text_model_name', type=str, default='microsoft/deberta-v3-large', help='Name of the pre-trained text model.')
    parser.add_argument('--devices', type=str, default="0,1,2,3", help='Comma-separated list of GPU device IDs to use (e.g., "0,1,2,3").')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference PER GPU.')
    args = parser.parse_args()

    # --- MULTIPROCESSING SETUP ---
    mp.set_start_method('spawn', force=True)
    
    # 1. Parse device IDs and prepare data chunks
    devices = [int(d) for d in args.devices.split(',')]
    num_devices = len(devices)
    print(f"Starting inference on {num_devices} GPUs: {devices}")

    # 2. Load all data into memory first
    print("Loading and preparing data from CSV...")
    data_to_process = []
    with open(args.input_csv, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            sample_id = row.get('sample_id')
            catalog_content = row.get('catalog_content')
            if not sample_id or not catalog_content:
                continue
            
            img_path = os.path.join(args.image_dir, f"{sample_id}.jpg")
            if not os.path.exists(img_path):
                # We will handle missing images at the end to ensure all sample_ids are present
                continue
            
            data_to_process.append({
                'sample_id': sample_id,
                'prompt': create_vlm_prompt(catalog_content),
                'image_path': img_path
            })
    
    # 3. Split data into chunks for each worker
    data_chunks = np.array_split(data_to_process, num_devices)
    worker_args_list = []
    for i, device_id in enumerate(devices):
        worker_args_list.append(
            (data_chunks[i].tolist(), args.image_model_name, args.text_model_name, args.checkpoint, device_id, args.batch_size, i)
        )
    
    # 4. Run processes in parallel
    print("Launching worker processes...")
    with mp.Pool(processes=num_devices) as pool:
        # starmap applies arguments from the list to the worker function
        results_from_workers = pool.starmap(worker_inference, worker_args_list)

    # 5. Combine results and write to CSV
    print("\nAll workers finished. Combining results and writing to CSV...")
    # Flatten the list of lists into a single list
    final_results = list(chain.from_iterable(results_from_workers))
    
    # Create a dictionary for quick lookup of predictions
    results_map = {res['sample_id']: res['price'] for res in final_results}

    with open(args.output_csv, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['sample_id', 'price'])
        
        # Iterate through the original input file to maintain order and include all sample_ids
        with open(args.input_csv, 'r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            for row in tqdm(reader, desc="Writing final CSV"):
                sample_id = row.get('sample_id')
                # Use the predicted price if available, otherwise default to 0.0
                price = results_map.get(sample_id, "0.0")
                writer.writerow([sample_id, price])

    print(f"\n✅ Inference complete. Predictions saved to '{args.output_csv}'")