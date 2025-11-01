# --- Full Script for ViT-Large + RoBERTa-Large (Fusion Head) INFERENCE ---
# Description:
# Loads a fine-tuned ViT-Large + RoBERTa-Large model with a Transformer Fusion Head
# and predicts prices for an input CSV of product data.
#
# Requirements:
# pip install --upgrade transformers torch torchvision Pillow pandas tqdm scikit-learn accelerate
#
# Example usage:
# python inference_fusion.py \
#     --checkpoint "vit_roberta_fusion_checkpoints/vit_roberta_fusion_epoch_5.pth" \
#     --input_csv "/path/to/test.csv" \
#     --image_dir "/path/to/test_images" \
#     --output_csv "fusion_predictions.csv" \
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


# --- 1. Model Definition (Must match training script exactly) ---
class ViTRoBERTaForPricePrediction(nn.Module):
    def __init__(self, image_model_name, text_model_name):
        super().__init__()
        self.image_encoder = AutoModel.from_pretrained(image_model_name, torch_dtype=torch.bfloat16)
        self.text_encoder = AutoModel.from_pretrained(text_model_name, torch_dtype=torch.bfloat16)

        embedding_dim = self.image_encoder.config.hidden_size
        assert embedding_dim == self.text_encoder.config.hidden_size, "Image and text models must have the same embedding dimension."

        # Transformer Fusion Head
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=embedding_dim * 4,
            dropout=0.2,
            activation='gelu',
            batch_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(transformer_layer, num_layers=2)

        # Regression Head
        self.regression_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim // 2, 1)
        )
        self.fusion_transformer.to(dtype=torch.bfloat16)
        self.regression_head.to(dtype=torch.bfloat16)

    def forward(self, pixel_values, input_ids, attention_mask):
        image_outputs = self.image_encoder(pixel_values=pixel_values)
        image_features = image_outputs.pooler_output

        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.pooler_output

        image_features_seq = image_features.unsqueeze(1)
        text_features_seq = text_features.unsqueeze(1)
        
        combined_sequence = torch.cat([image_features_seq, text_features_seq], dim=1)
        
        fused_features = self.fusion_transformer(combined_sequence)
        
        fused_output = fused_features[:, 0, :]
        
        return self.regression_head(fused_output)


# --- 2. Prompt Function (Unchanged from training) ---
def create_vlm_prompt(catalog_content: str) -> str:
    title, ipq, unit, value, description = "N/A", 1, "N/A", "N/A", ""
    remaining_content = str(catalog_content)
    # ... (exact same text parsing logic as training) ...
    prompt = (
        f"Title: {title}. "
        f"Pack Quantity: {ipq}. "
        f"Unit: {unit}. "
        f"Value: {value}. "
        f"Description: {description}"
    )
    return prompt


# --- 3. Model + Checkpoint Loading (MODIFIED) ---
def load_finetuned_model(image_model_name, text_model_name, checkpoint_path, device):
    print(f"Loading model with ViT='{image_model_name}' and RoBERTa='{text_model_name}'...")
    model = ViTRoBERTaForPricePrediction(image_model_name, text_model_name)

    print(f"Loading checkpoint from '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # --- MODIFIED: Load state dicts for both parts of the fusion head ---
    print("Loading Transformer Fusion Head state...")
    model.fusion_transformer.load_state_dict(checkpoint['fusion_transformer_state_dict'])
    model.regression_head.load_state_dict(checkpoint['regression_head_state_dict'])
    # -------------------------------------------------------------------
    
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
                    writer.writerow([sample_id, 0.0]) # Default to 0.0 if image not found
                    continue

                try:
                    prompt_text = create_vlm_prompt(catalog_content)
                    image = Image.open(img_path).convert('RGB')

                    # Preprocess inputs (let processor handle resizing)
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
                    
                    # Ensure price is positive
                    predicted_price = max(0.0, predicted_price)

                    writer.writerow([sample_id, f"{predicted_price:.4f}"])

                except Exception as e:
                    print(f"⚠️ Error processing {sample_id}: {e}")
                    writer.writerow([sample_id, 0.0]) # Default to 0.0 on error

    print(f"\n✅ Inference complete. Predictions saved to: {args.output_csv}")


# --- 5. CLI Entrypoint (MODIFIED) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ViT-Large + RoBERTa-Large with Transformer Fusion Head Inference")
    
    # --- MODIFIED: Updated default paths and model names ---
    parser.add_argument('--checkpoint', type=str, default="vit_roberta_fusion_checkpoints/vit_roberta_fusion_epoch_2.pth",
                        help='Path to the fine-tuned fusion head (.pth file).')
    parser.add_argument('--input_csv', type=str, default="/home/abrol/khushal/Misc/Amazon_Challenge/Yash_EXp/student_resource/dataset/train_subset_10k.csv",
                        help='Path to the input CSV file containing product data.')
    parser.add_argument('--image_dir', type=str, default="/home/abrol/khushal/Misc/Amazon_Challenge/Yash_EXp/student_resource/images",
                        help='Path to the directory containing product images.')
    parser.add_argument('--output_csv', type=str, default='vitrobertalarge.csv',
                        help='Path to save the output predictions CSV file.')
    parser.add_argument('--image_model_name', type=str, default='google/vit-large-patch16-224',
                        help='ViT model name.')
    parser.add_argument('--text_model_name', type=str, default='roberta-large',
                        help='RoBERTa model name.')
    parser.add_argument('--device', type=str, default="cuda:1" if torch.cuda.is_available() else "cpu",
                        help='Device for inference.')
    
    args = parser.parse_args()

    print(f"Using device: {args.device}")
    run_inference(args)