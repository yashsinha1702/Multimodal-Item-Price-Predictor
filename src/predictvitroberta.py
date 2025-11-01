# --- Full Script for ViT + RoBERTa Price Prediction INFERENCE ---
# Description:
# Loads a fine-tuned ViT + RoBERTa dual-encoder model checkpoint
# and predicts prices for an input CSV of sample_ids and catalog_content.
#
# Requirements:
# pip install --upgrade transformers torch torchvision Pillow pandas tqdm scikit-learn accelerate
#
# Example usage:
# python inference_vit_roberta.py \
#     --checkpoint "vit_roberta_checkpoints_224/vit_roberta_epoch_4.pth" \
#     --input_csv "/path/to/test.csv" \
#     --image_dir "/path/to/test_images_resized" \
#     --output_csv "vit_roberta_predictions.csv" \
#     --device "cuda:0"

import argparse
import os
import csv
import re
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from transformers import AutoModel, ViTImageProcessor, RobertaTokenizer


# --- 1. Model Definition (Must match training exactly) ---
class ViTRoBERTaForPricePrediction(nn.Module):
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
        text_features = text_outputs.pooler_output

        combined_features = torch.cat([image_features, text_features], dim=1)
        return self.price_predictor(combined_features)


# --- 2. Prompt Function (same as training) ---
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


# --- 3. Model + Checkpoint Loading ---
def load_finetuned_model(image_model_name, text_model_name, checkpoint_path, device):
    print(f"Loading model with ViT='{image_model_name}' and RoBERTa='{text_model_name}'...")
    model = ViTRoBERTaForPricePrediction(image_model_name, text_model_name)

    print(f"Loading checkpoint from '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint  # direct state dict

    model.price_predictor.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    image_processor = ViTImageProcessor.from_pretrained(image_model_name)
    tokenizer = RobertaTokenizer.from_pretrained(text_model_name)

    print("✅ Model and processors loaded successfully.")
    return model, image_processor, tokenizer


# --- 4. Main Inference Logic ---
def run_inference(args):
    device = args.device
    model, image_processor, tokenizer = load_finetuned_model(
        args.image_model_name, args.text_model_name, args.checkpoint, device
    )

    with open(args.input_csv, 'r', encoding='utf-8') as infile, \
         open(args.output_csv, 'w', encoding='utf-8', newline='') as outfile:

        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)
        writer.writerow(['sample_id', 'price'])  # header

        with torch.no_grad():
            for row in tqdm(reader, desc="Predicting prices"):
                sample_id = row.get('sample_id')
                catalog_content = row.get('catalog_content')

                if not sample_id or not catalog_content:
                    continue

                img_path = os.path.join(args.image_dir, f"{sample_id}.jpg")
                if not os.path.exists(img_path):
                    writer.writerow([sample_id, "Image Not Found"])
                    continue

                try:
                    prompt_text = create_vlm_prompt(catalog_content)
                    image = Image.open(img_path).convert('RGB').resize((224, 224))

                    # Preprocess inputs
                    image_inputs = image_processor(images=image, return_tensors="pt")
                    text_inputs = tokenizer(
                        [prompt_text], return_tensors="pt",
                        padding=True, truncation=True, max_length=512
                    )

                    # Move tensors to device
                    image_inputs = {k: v.to(device, dtype=torch.bfloat16) for k, v in image_inputs.items()}
                    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

                    # Predict
                    predicted_price = model(
                        pixel_values=image_inputs['pixel_values'],
                        input_ids=text_inputs['input_ids'],
                        attention_mask=text_inputs['attention_mask']
                    ).item()

                    writer.writerow([sample_id, f"{predicted_price:.4f}"])

                except Exception as e:
                    print(f"⚠️ Error processing {sample_id}: {e}")
                    writer.writerow([sample_id, "Error"])

    print(f"\n✅ Inference complete. Predictions saved to: {args.output_csv}")


# --- 5. CLI Entrypoint ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ViT + RoBERTa Price Prediction Inference")
    parser.add_argument('--checkpoint', type=str,default="/home/abrol/khushal/Misc/Amazon_Challenge/Yash_EXp/student_resource/src/vit_roberta_checkpoints_224/vit_roberta_epoch_5.pth",  help='Path to the fine-tuned price predictor head (.pth file).')
    parser.add_argument('--input_csv', type=str,default="/home/abrol/khushal/Misc/Amazon_Challenge/Yash_EXp/student_resource/dataset/sample_test.csv",  help='Path to the input CSV file containing product data.')
    parser.add_argument('--image_dir', type=str, default="/home/abrol/khushal/Misc/Amazon_Challenge/Yash_EXp/student_resource/dataset/sample_test_images_resized", help='Path to the directory containing product images.')
    parser.add_argument('--output_csv', type=str, default='vit_roberta_predictions.csv', help='Path to save the output predictions CSV file.')
    parser.add_argument('--image_model_name', type=str, default='google/vit-base-patch16-224-in21k', help='ViT model name.')
    parser.add_argument('--text_model_name', type=str, default='roberta-base', help='RoBERTa model name.')
    parser.add_argument('--device', type=str, default="cuda:3" if torch.cuda.is_available() else "cpu", help='Device for inference.')
    args = parser.parse_args()

    print(f"Using device: {args.device}")
    run_inference(args)
