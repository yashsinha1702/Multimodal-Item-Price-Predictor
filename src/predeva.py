# --- Full Script for EVA-02 + DeBERTa-V3 (Fusion Head) INFERENCE ---
# Description:
# Loads the fine-tuned fusion head for the EVA-02 + DeBERTa-V3 model,
# processes an input CSV and corresponding images, and outputs price predictions.
#
# Requirements:
# pip install --upgrade transformers torch torchvision Pillow pandas tqdm scikit-learn accelerate sentencepiece protobuf timm
#
# How to run:
# python inference_eva02_deberta_fusion.py \
#     --checkpoint "eva02_deberta_checkpoints/eva02_deberta_epoch_5.pth" \
#     --input_csv "/path/to/test.csv" \
#     --image_dir "/path/to/test_images" \
#     --output_csv "eva02_deberta_fusion_predictions.csv"

import argparse
import os
import csv
import re
from tqdm import tqdm

import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor, AutoTokenizer
from PIL import Image

# --- 1. Model Definition (Must be identical to the training script) ---
class EVA02DeBERTaForPricePrediction(nn.Module):
    def __init__(self, image_model_name, text_model_name):
        super().__init__()
        self.image_encoder = AutoModel.from_pretrained(image_model_name, trust_remote_code=True)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)

        image_embedding_dim = self.image_encoder.timm_model.num_features
        text_embedding_dim = self.text_encoder.config.hidden_size
        
        # Fusion Head Definition
        fusion_dim = 1024
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


# --- 3. Function to Load Fine-Tuned Model (MODIFIED) ---
def load_finetuned_model(image_model_name, text_model_name, checkpoint_path, device):
    print(f"Loading base models '{image_model_name}' and '{text_model_name}'...")
    model = EVA02DeBERTaForPricePrediction(image_model_name, text_model_name)

    print(f"Loading fine-tuned fusion head from '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # --- MODIFIED: Load state dictionaries for all parts of the new head ---
    model.image_projection.load_state_dict(checkpoint['image_projection_state_dict'])
    model.text_projection.load_state_dict(checkpoint['text_projection_state_dict'])
    model.fusion_transformer.load_state_dict(checkpoint['fusion_transformer_state_dict'])
    model.regression_head.load_state_dict(checkpoint['regression_head_state_dict'])
    # --------------------------------------------------------------------

    image_processor = AutoImageProcessor.from_pretrained(image_model_name)
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    model.to(device)
    model.eval()
    
    # Use torch.compile for a significant speed-up
    print("Compiling model for faster inference...")
    model = torch.compile(model)
    
    print("✅ Model and processors loaded successfully.")
    return model, image_processor, tokenizer


# --- 4. Main Inference Logic ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run EVA-02 + DeBERTa-V3 (Fusion Head) price prediction inference.")
    # --- MODIFIED: Updated defaults for the fusion model ---
    parser.add_argument('--checkpoint', type=str, default="eva02_deberta_checkpoints/eva02_deberta_epoch_1.pth", help='Path to the fine-tuned fusion head (.pth file).')
    parser.add_argument('--input_csv', type=str, default="/home/abrol/khushal/Misc/Amazon_Challenge/Yash_EXp/student_resource/dataset/test.csv", help='Path to the input CSV file containing product data.')
    parser.add_argument('--image_dir', type=str, default="/home/abrol/khushal/Misc/Amazon_Challenge/Yash_EXp/student_resource/test/images", help='Path to the directory containing product images.')
    parser.add_argument('--output_csv', type=str, default='submission.csv', help='Path to save the output predictions CSV file.')
    parser.add_argument('--image_model_name', type=str, default='timm/eva02_large_patch14_224.mim_in22k', help='Name of the pre-trained image model.')
    parser.add_argument('--text_model_name', type=str, default='microsoft/deberta-v3-large', help='Name of the pre-trained text model.')
    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device to run inference on (e.g., "cuda:0" or "cpu").')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference.')
    args = parser.parse_args()

    print(f"Using device: {args.device}")

    model, image_processor, tokenizer = load_finetuned_model(
        args.image_model_name,
        args.text_model_name,
        args.checkpoint,
        args.device
    )

    # Prepare data for batching
    data_to_process = []
    with open(args.input_csv, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            sample_id = row.get('sample_id')
            catalog_content = row.get('catalog_content')
            if not sample_id or not catalog_content:
                continue
            
            data_to_process.append({
                'sample_id': sample_id,
                'prompt': create_vlm_prompt(catalog_content),
                'image_path': os.path.join(args.image_dir, f"{sample_id}.jpg")
            })

    results = []
    # Create a set of all sample_ids for final check
    all_sample_ids = {item['sample_id'] for item in data_to_process}

    with torch.no_grad():
        for i in tqdm(range(0, len(data_to_process), args.batch_size), desc="Predicting prices"):
            batch_data = data_to_process[i:i + args.batch_size]
            
            sample_ids = [item['sample_id'] for item in batch_data]
            prompts = [item['prompt'] for item in batch_data]
            
            # Filter out items with missing images before processing
            valid_batch_items = []
            valid_images = []
            for item in batch_data:
                if os.path.exists(item['image_path']):
                    valid_batch_items.append(item)
                    valid_images.append(Image.open(item['image_path']).convert('RGB').resize((448, 448)))
                else:
                    print(f"Warning: Image for {item['sample_id']} not found. Assigning price 0.0.")
                    results.append({'sample_id': item['sample_id'], 'price': "0.0000"})

            if not valid_images: # Skip batch if no images were found
                continue

            valid_sample_ids = [item['sample_id'] for item in valid_batch_items]
            valid_prompts = [item['prompt'] for item in valid_batch_items]
            
            try:
                # Preprocess the valid batch
                image_inputs = image_processor(valid_images, return_tensors="pt")
                text_inputs = tokenizer(
                    valid_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
                )
                
                inputs = {
                    "pixel_values": image_inputs['pixel_values'].to(args.device),
                    "input_ids": text_inputs['input_ids'].to(args.device),
                    "token_type_ids": text_inputs['token_type_ids'].to(args.device),
                    "attention_mask": text_inputs['attention_mask'].to(args.device)
                }

                predicted_price_tensor = model(**inputs).squeeze(-1)
                predicted_prices = predicted_price_tensor.cpu().numpy()

                for sid, price in zip(valid_sample_ids, predicted_prices):
                    results.append({'sample_id': sid, 'price': f"{max(0, price):.4f}"})

            except Exception as e:
                print(f"Error processing batch starting at index {i}: {e}")
                for sid in valid_sample_ids:
                    results.append({'sample_id': sid, 'price': "0.0000"})

    # Final check to ensure all sample_ids have a result
    results_map = {res['sample_id']: res['price'] for res in results}
    
    with open(args.output_csv, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['sample_id', 'price'])
        
        # Read from the original file to guarantee order and inclusion of all samples
        with open(args.input_csv, 'r', encoding='utf-8') as infile:
             reader = csv.DictReader(infile)
             for row in tqdm(reader, desc="Writing final CSV"):
                 sample_id = row.get('sample_id')
                 price = results_map.get(sample_id, "0.0000") # Default to 0 if any sample was missed
                 writer.writerow([sample_id, price])

    print(f"\n✅ Inference complete. Predictions have been saved to '{args.output_csv}'")