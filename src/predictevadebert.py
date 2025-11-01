# --- Full Script for EVA-02 + DeBERTa-V3 Price Prediction INFERENCE ---
# Description:
# This script loads the fine-tuned regression head for the EVA-02 + DeBERTa-V3 model,
# processes an input CSV and corresponding images, and outputs price predictions.
#
# Requirements:
# pip install --upgrade transformers torch torchvision Pillow pandas tqdm scikit-learn accelerate sentencepiece protobuf timm
#
# How to run:
# python inference_eva02_deberta.py \
#     --checkpoint "/path/to/your/eva02_deberta_checkpoints/eva02_deberta_epoch_5.pth" \
#     --input_csv "/path/to/your/test.csv" \
#     --image_dir "/path/to/your/test_images_folder" \
#     --output_csv "eva02_deberta_predictions.csv" \
#     --device "cuda:0"

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
        # trust_remote_code=True is needed for timm models
        # Note: We load in float32 for inference to ensure compatibility with all GPUs.
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


# --- 3. Function to Load Fine-Tuned Model ---
def load_finetuned_model(image_model_name, text_model_name, checkpoint_path, device):
    """
    Initializes the dual-encoder model and loads the fine-tuned price predictor head.
    """
    print(f"Loading base models '{image_model_name}' and '{text_model_name}'...")
    model = EVA02DeBERTaForPricePrediction(image_model_name, text_model_name)

    print(f"Loading fine-tuned regression head from '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.price_predictor.load_state_dict(checkpoint['model_state_dict'])

    image_processor = AutoImageProcessor.from_pretrained(image_model_name)
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    model.to(device)
    model.eval()
    model.compile()
    print("Model and processors loaded successfully.")
    return model, image_processor, tokenizer


# --- 4. Main Inference Logic ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run EVA-02 + DeBERTa-V3 price prediction inference.")
    parser.add_argument('--checkpoint', type=str,default="/home/abrol/khushal/Misc/Amazon_Challenge/Yash_EXp/student_resource/src/eva02_deberta_checkpoints/eva02_deberta_epoch_2.pth",  help='Path to the fine-tuned price predictor head (.pth file).')
    parser.add_argument('--input_csv', type=str,default="/home/abrol/khushal/Misc/Amazon_Challenge/Yash_EXp/student_resource/dataset/test.csv",  help='Path to the input CSV file containing product data.')
    parser.add_argument('--image_dir', type=str, default="/home/abrol/khushal/Misc/Amazon_Challenge/Yash_EXp/student_resource/test/images", help='Path to the directory containing product images.')
    parser.add_argument('--output_csv', type=str, default='evadebert_finalpredictions.csv', help='Path to save the output predictions CSV file.')
    parser.add_argument('--image_model_name', type=str, default='timm/eva02_large_patch14_224.mim_in22k', help='Name of the pre-trained image model.')
    parser.add_argument('--text_model_name', type=str, default='microsoft/deberta-v3-large', help='Name of the pre-trained text model.')
    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device to run inference on (e.g., "cuda:0" or "cpu").')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference.')
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
            
            img_path = os.path.join(args.image_dir, f"{sample_id}.jpg")
            if not os.path.exists(img_path):
                print(f"Warning: Image not found for sample_id {sample_id}. Skipping.")
                continue
            
            data_to_process.append({
                'sample_id': sample_id,
                'prompt': create_vlm_prompt(catalog_content),
                'image_path': img_path
            })

    results = []
    with torch.no_grad():
        for i in tqdm(range(0, len(data_to_process), args.batch_size), desc="Predicting prices"):
            batch_data = data_to_process[i:i + args.batch_size]
            
            sample_ids = [item['sample_id'] for item in batch_data]
            prompts = [item['prompt'] for item in batch_data]
            images = [Image.open(item['image_path']).convert('RGB').resize((448, 448)) for item in batch_data]

            try:
                # Preprocess the batch
                image_inputs = image_processor(images, return_tensors="pt")
                text_inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                inputs = {
                    "pixel_values": image_inputs['pixel_values'].to(args.device),
                    "input_ids": text_inputs['input_ids'].to(args.device),
                    "token_type_ids": text_inputs['token_type_ids'].to(args.device),
                    "attention_mask": text_inputs['attention_mask'].to(args.device)
                }

                predicted_price_tensor = model(**inputs).squeeze(-1)
                predicted_prices = predicted_price_tensor.cpu().numpy()

                for sid, price in zip(sample_ids, predicted_prices):
                    results.append({'sample_id': sid, 'price': f"{max(0, price):.4f}"})

            except Exception as e:
                print(f"Error processing batch starting at index {i}: {e}")
                for sid in sample_ids:
                    results.append({'sample_id': sid, 'price': "Error"})

    with open(args.output_csv, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['sample_id', 'price'])
        for result in results:
            writer.writerow([result['sample_id'], result['price']])

    print(f"\n✅ Inference complete. Predictions have been saved to '{args.output_csv}'")