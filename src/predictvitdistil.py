# --- Full Script for ViT + DistilBERT Price Prediction (Inference) ---
# Description:
# Loads a fine-tuned ViT + DistilBERT model checkpoint and predicts prices
# for given (sample_id, catalog_content, image) pairs.
#
# Usage Example:
# python inference_vit_distilbert.py \
#     --checkpoint_dir "vit_distilbert_checkpoints_224" \
#     --input_csv "/path/to/test.csv" \
#     --image_dir "/path/to/test_images_resized" \
#     --output_csv "vit_distilbert_predictions.csv" \
#     --device "cuda:0"

import os
import re
import csv
import glob
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from transformers import AutoModel, ViTImageProcessor, DistilBertTokenizer

# -------------------------------
# 1️⃣  Model Definition (same as training)
# -------------------------------
class ViTDistilBERTForPricePrediction(nn.Module):
    def __init__(self, image_model_name, text_model_name):
        super().__init__()
        self.image_encoder = AutoModel.from_pretrained(image_model_name, torch_dtype=torch.bfloat16)
        self.text_encoder = AutoModel.from_pretrained(text_model_name, torch_dtype=torch.bfloat16)

        image_embedding_dim = self.image_encoder.config.hidden_size
        text_embedding_dim = self.text_encoder.config.hidden_size
        combined_dim = image_embedding_dim + text_embedding_dim

        self.price_predictor = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(combined_dim // 2, 1),
        ).to(dtype=torch.bfloat16)

    def forward(self, pixel_values, input_ids, attention_mask):
        image_outputs = self.image_encoder(pixel_values=pixel_values)
        image_features = image_outputs.pooler_output

        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0]  # CLS token

        combined_features = torch.cat([image_features, text_features], dim=1)
        return self.price_predictor(combined_features)

# -------------------------------
# 2️⃣  Helper Function — Prompt Creator
# -------------------------------
def create_vlm_prompt(catalog_content: str) -> str:
    """Parses catalog content into structured text input."""
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
                    try:
                        ipq = int(extracted_value)
                    except ValueError:
                        ipq = 1
                elif key == 'unit':
                    unit = extracted_value
                elif key == 'value':
                    value = extracted_value
                remaining_content = re.sub(pattern, "", remaining_content, count=1, flags=re.IGNORECASE)
                break

    description = "\n".join(line.strip() for line in remaining_content.split('\n') if line.strip())
    return f"Title: {title}. Pack Quantity: {ipq}. Unit: {unit}. Value: {value}. Description: {description}"

# -------------------------------
# 3️⃣  Find Latest Checkpoint Automatically
# -------------------------------
def find_latest_checkpoint(checkpoint_dir):
    list_of_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if not list_of_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    latest_checkpoint = max(list_of_files, key=os.path.getctime)
    print(f"✅ Using latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint

# -------------------------------
# 4️⃣  Load Model + Tokenizers
# -------------------------------
def load_finetuned_model(image_model_name, text_model_name, checkpoint_path, device):
    print(f"Loading ViT + DistilBERT model for inference...")
    model = ViTDistilBERTForPricePrediction(image_model_name, text_model_name)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.price_predictor.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    image_processor = ViTImageProcessor.from_pretrained(image_model_name)
    tokenizer = DistilBertTokenizer.from_pretrained(text_model_name)
    print("✅ Model and processors loaded successfully.")
    return model, image_processor, tokenizer

# -------------------------------
# 5️⃣  Inference Logic
# -------------------------------
def run_inference(checkpoint_dir, input_csv, image_dir, output_csv, device,
                  image_model_name="google/vit-base-patch16-224-in21k",
                  text_model_name="distilbert-base-uncased"):
    
    checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    model, image_processor, tokenizer = load_finetuned_model(image_model_name, text_model_name, checkpoint_path, device)

    with open(input_csv, 'r', encoding='utf-8') as infile, \
         open(output_csv, 'w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)
        writer.writerow(['sample_id', 'price'])

        for row in tqdm(reader, desc="Predicting"):
            sample_id = row.get('sample_id')
            catalog_content = row.get('catalog_content')

            if not sample_id or not catalog_content:
                writer.writerow([sample_id or "N/A", "Missing data"])
                continue

            img_path = os.path.join(image_dir, f"{sample_id}.jpg")
            if not os.path.exists(img_path):
                writer.writerow([sample_id, "Image not found"])
                continue

            try:
                image = Image.open(img_path).convert('RGB').resize((224, 224))
                prompt = create_vlm_prompt(catalog_content)

                image_inputs = image_processor(images=image, return_tensors="pt")
                text_inputs = tokenizer(
                    [prompt],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )

                image_inputs = {k: v.to(device, dtype=torch.bfloat16) for k, v in image_inputs.items()}
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

                with torch.no_grad():
                    pred = model(
                        pixel_values=image_inputs['pixel_values'],
                        input_ids=text_inputs['input_ids'],
                        attention_mask=text_inputs['attention_mask']
                    ).item()

                writer.writerow([sample_id, f"{pred:.4f}"])

            except Exception as e:
                print(f"⚠️ Error with {sample_id}: {e}")
                writer.writerow([sample_id, "Error"])

    print(f"\n✅ Predictions saved to: {output_csv}")

# -------------------------------
# 6️⃣  CLI Entrypoint
# -------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ViT + DistilBERT Price Prediction Inference")
    parser.add_argument('--checkpoint', type=str,default="/home/abrol/khushal/Misc/Amazon_Challenge/Yash_EXp/student_resource/src/vit_distilbert_checkpoints_224",  help='Path to the fine-tuned price predictor head (.pth file).')
    parser.add_argument('--input_csv', type=str,default="/home/abrol/khushal/Misc/Amazon_Challenge/Yash_EXp/student_resource/dataset/sample_test.csv",  help='Path to the input CSV file containing product data.')
    parser.add_argument('--image_dir', type=str, default="/home/abrol/khushal/Misc/Amazon_Challenge/Yash_EXp/student_resource/dataset/sample_test_images_resized", help='Path to the directory containing product images.')
    parser.add_argument('--output_csv', type=str, default='vit_distil_predictions.csv', help='Path to save the output predictions CSV file.')
    parser.add_argument("--device", type=str, default="cuda:3" if torch.cuda.is_available() else "cpu", help="Device for inference.")
    args = parser.parse_args()

    print(f"Using device: {args.device}")
    run_inference(
        args.checkpoint,
        args.input_csv,
        args.image_dir,
        args.output_csv,
        args.device
    )
