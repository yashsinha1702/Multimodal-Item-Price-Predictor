# --- Full Script for CLIP Price Prediction INFERENCE ---
# Description:
# This script loads the fine-tuned regression head for the CLIP-based price prediction model,
# processes an input CSV and corresponding images, and outputs price predictions.
#
# Requirements:
# pip install --upgrade transformers torch torchvision Pillow pandas tqdm scikit-learn accelerate
#
# How to run:
# python inference_clip.py \
#     --checkpoint "/path/to/your/clip_checkpoints/clip_price_predictor_epoch_5.pth" \
#     --input_csv "/path/to/your/test.csv" \
#     --image_dir "/path/to/your/test_images_folder" \
#     --output_csv "clip_predictions.csv" \
#     --device "cuda:0"

import argparse
import os
import csv
import re

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from tqdm import tqdm

# --- 1. Model Definition (Must match the training script) ---
class CLIPForPricePrediction(nn.Module):
    """
    A wrapper for the CLIP model with a regression head for price prediction.
    The base vision and text encoders are frozen.
    """
    def __init__(self, model_name):
        super().__init__()
        # Use bfloat16 for potential performance gains if supported
        self.base_model = CLIPModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        embedding_dim = self.base_model.projection_dim
        
        # Regression head to predict a single price value
        self.price_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, 1),
        )
        # Ensure the head also uses bfloat16
        self.price_predictor.to(dtype=torch.bfloat16)

    def forward(self, **kwargs):
        # Get separate embeddings for image and text
        image_features = self.base_model.get_image_features(pixel_values=kwargs['pixel_values'])
        text_features = self.base_model.get_text_features(
            input_ids=kwargs['input_ids'],
            attention_mask=kwargs['attention_mask']
        )
        
        # Concatenate features and predict price
        combined_features = torch.cat([image_features, text_features], dim=1)
        return self.price_predictor(combined_features)


# --- 2. Prompt Formatting (Must match the training script) ---
def create_clip_prompt(catalog_content: str) -> str:
    """
    Parses catalog content into a structured text prompt for the CLIP model.
    """
    title, ipq, unit, value = "N/A", 1, "N/A", "N/A"
    remaining_content = str(catalog_content)
    
    # Extract Title
    title_match = re.search(r"Item Name:\s*(.*)", remaining_content, re.IGNORECASE)
    if title_match:
        title = title_match.group(1).strip()
    else:
        lines = [line.strip() for line in remaining_content.split('\n') if line.strip()]
        if lines:
            title = lines[0]
            remaining_content = remaining_content.replace(title, "", 1)
            
    # Extract Quantity, Unit, and Value
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
                
    # Construct the final prompt
    prompt = (
        f"Title: {title}. "
        f"Pack Quantity: {ipq}. "
        f"Unit: {unit}. "
        f"Value: {value}. "
    )
    return prompt


# --- 3. Function to Load Fine-Tuned Model ---
def load_finetuned_model(model_name, checkpoint_path, device):
    """
    Initializes the CLIP model and loads the fine-tuned price predictor head.
    """
    print(f"Loading base model '{model_name}'...")
    model = CLIPForPricePrediction(model_name=model_name)

    print(f"Loading fine-tuned regression head from '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check if the checkpoint is a dictionary containing the state dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        # Assume the checkpoint file is the state dict itself
        state_dict = checkpoint
    
    model.price_predictor.load_state_dict(state_dict)
    
    # Load the corresponding processor
    processor = CLIPProcessor.from_pretrained(model_name)
    
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    print("Model and processor loaded successfully.")
    return model, processor


# --- 4. Main Inference Logic ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run CLIP price prediction inference.")
    # The `parser.add_argument` lines you provided are setting up command-line arguments for the
    # Python script. Here's what each argument does:
    parser.add_argument('--checkpoint', type=str,default="/home/abrol/khushal/Misc/Amazon_Challenge/Yash_EXp/student_resource/src/clip_large_checkpoints_224/clip_large_epoch_8.pth",  help='Path to the fine-tuned price predictor head (.pth file).')
    parser.add_argument('--input_csv', type=str,default="/home/abrol/khushal/Misc/Amazon_Challenge/Yash_EXp/student_resource/dataset/test.csv",  help='Path to the input CSV file containing product data.')
    parser.add_argument('--image_dir', type=str, default="/home/abrol/khushal/Misc/Amazon_Challenge/Yash_EXp/student_resource/test/images", help='Path to the directory containing product images.')
    parser.add_argument('--output_csv', type=str, default='cliplarge_predictionsfinal.csv', help='Path to save the output predictions CSV file.')
    parser.add_argument('--model_name', type=str, default='openai/clip-vit-large-patch14', help='Name of the pre-trained CLIP model.')
    parser.add_argument('--device', type=str, default="cuda:1" if torch.cuda.is_available() else "cpu", help='Device to run inference on (e.g., "cuda:0" or "cpu").')
    args = parser.parse_args()

    print(f"Using device: {args.device}")

    # Load the model and processor
    model, processor = load_finetuned_model(args.model_name, args.checkpoint, args.device)

    # Open files for reading and writing
    with open(args.input_csv, 'r', encoding='utf-8') as infile, \
         open(args.output_csv, 'w', encoding='utf-8', newline='') as outfile:

        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)
        writer.writerow(['sample_id', 'price']) # Write header

        # Disable gradient calculations for efficiency
        with torch.no_grad():
            for row in tqdm(reader, desc="Predicting prices"):
                sample_id = row.get('sample_id')
                catalog_content = row.get('catalog_content')

                if not sample_id or not catalog_content:
                    continue

                img_path = os.path.join(args.image_dir, f"{sample_id}.jpg")
                
                if not os.path.exists(img_path):
                    print(f"Warning: Image not found for sample_id {sample_id}. Skipping.")
                    writer.writerow([sample_id, "Image Not Found"])
                    continue

                try:
                    # 1. Create the text prompt from catalog content
                    prompt_text = create_clip_prompt(catalog_content)
                    
                    # 2. Load and resize the image (CRITICAL: Must match training size)
                    image = Image.open(img_path).convert('RGB').resize((448, 448))
                    
                    # 3. Process image and text using the CLIP processor
                    inputs = processor(
                        text=[prompt_text],
                        images=[image],
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(args.device, dtype=torch.bfloat16)

                    # 4. Get the price prediction from the model
                    predicted_price_tensor = model(**inputs)
                    predicted_price = predicted_price_tensor.item()
                    
                    # 5. Write the result to the output CSV
                    writer.writerow([sample_id, f"{predicted_price:.4f}"])

                except Exception as e:
                    print(f"Error processing sample_id {sample_id}: {e}")
                    writer.writerow([sample_id, "Error"])

    print(f"\n✅ Inference complete. Predictions have been saved to '{args.output_csv}'")