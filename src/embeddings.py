# save_embeddings.py (run on the same hardware)
import torch, os
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import json, math
from tqdm import tqdm
import numpy as np

MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"
IMAGE_DIR = "/home/gpu2/warade/student_resource/images"
JSONL = "training_dataset.jsonl"
OUT_DIR = "embeddings256"
os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda:0"
model = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
model.eval()

batch_size = 32
records = [json.loads(l) for l in open(JSONL)]
for i in range(0, len(records), batch_size):
    batch = records[i:i+batch_size]
    images = [Image.open(rec['image_path']).convert('RGB').resize((448,448), Image.Resampling.BICUBIC) for rec in batch]
    prompts = [f"<image>\nPredict the price for the item.\n{rec['prompt']}" for rec in batch]
    with torch.no_grad():
        inputs = processor(text=prompts, images=images, return_tensors="pt", padding="longest", max_length=1100)
        inputs = {k: v.to(device) for k,v in inputs.items()}
        # use autocast for speed & memory
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            out = model(**inputs, output_hidden_states=True)
        emb = out.hidden_states[-1][:, -1, :].to(torch.float16).cpu()   # keep float32 for head training (or float16 if you want)
    prices = torch.tensor([rec['price'] for rec in batch], dtype=torch.float16)
    ids = [rec['id'] for rec in batch]
    torch.save({"ids": ids, "emb": emb, "prices": prices}, os.path.join(OUT_DIR, f"emb_{i//batch_size:05d}.pt"))
    print(f"Saved batch {i//batch_size}")
