import argparse
import os
import csv
import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
from tqdm import tqdm
import re

# --- 1. Model wrapper for SmolVLM + regression head ---
class SmolVLMForPricePrediction(nn.Module):
    def __init__(self, model_name="HuggingFaceTB/SmolVLM-500M-Instruct", device="cpu"):
        super().__init__()
        # Quantization only if running on CUDA
        quantization_config = {"load_in_4bit": True} if 'cuda' in str(device) else None
        self.base_model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            quantization_config=quantization_config,
        )
        hidden_size = self.base_model.config.text_config.hidden_size
        self.price_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, 1),
        )
        if 'cuda' in str(device):
            self.price_predictor.to(dtype=torch.bfloat16)

    def forward(self, input_ids, pixel_values, attention_mask, **kwargs):
        outputs = self.base_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden_state = outputs.hidden_states[-1][:, -1, :]
        predicted_price = self.price_predictor(last_hidden_state)
        return predicted_price

# --- 2. Prompt formatting ---
def create_vlm_prompt(catalog_content: str) -> str:
    title, ipq, unit, value = "N/A", 1, "N/A", "N/A"
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
    prompt = (
        f"Product Information:\n"
        f"- Title: {title}\n"
        f"- Item Pack Quantity: {ipq}\n"
        f"- Unit: {unit}\n"
        f"- Value: {value}"
    )
    return prompt

# --- 3. Load fine-tuned model ---
def load_finetuned_model(model_name, checkpoint_path, device):
    model = SmolVLMForPricePrediction(model_name=model_name, device=device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

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

    # Load processor
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)
    model.eval()
    return model, processor


# --- 4. Main inference ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run SmolVLM price prediction.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to fine-tuned head (.pth)')
    parser.add_argument('--input_csv', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--output_csv', type=str, default='predictions.csv')
    parser.add_argument('--model_name', type=str, default='HuggingFaceTB/SmolVLM-500M-Instruct')
    parser.add_argument('--max_length', type=int, default=1500)
    args = parser.parse_args()

    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device="cuda:0"
    #device="cpu"
    print(f"Using device: {device}")

    model, processor = load_finetuned_model(args.model_name, args.checkpoint, device)

    with open(args.input_csv, 'r', encoding='utf-8') as infile, \
         open(args.output_csv, 'w', encoding='utf-8', newline='') as outfile:

        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)
        writer.writerow(['sample_id', 'price'])

        with torch.no_grad():
            for row in tqdm(reader, desc="Predicting prices"):
                sample_id = row.get('sample_id')
                catalog_content = row.get('catalog_content')
                if not sample_id or not catalog_content:
                    continue

                img_path = os.path.join(args.image_dir, f"{sample_id}.jpg")
                # if not os.path.exists(img_path):
                #     writer.writerow([sample_id, "N/A"])
                #     continue

                try:
                    prompt_text = create_vlm_prompt(catalog_content)
                    image = Image.open(img_path).convert('RGB')
                    task_prompt = f"<image>\nPredict the price for the following item based on its image and details.\n{prompt_text}"

                    inputs = processor(
                        text=[task_prompt],
                        images=[image],
                        return_tensors="pt",
                        padding="longest",
                        truncation=True,
                        max_length=args.max_length
                    ).to(device, dtype=torch.bfloat16 if 'cuda' in device else torch.float32)

                    predicted_price_tensor = model(**inputs)
                    predicted_price = predicted_price_tensor.item()
                    writer.writerow([sample_id, f"{predicted_price:.4f}"])

                except Exception as e:
                    print(f"Error processing sample_id {sample_id}: {e}")
                    writer.writerow([sample_id, "Error"])

    print(f"\nInference complete. Predictions saved to '{args.output_csv}'")
