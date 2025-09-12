
import torch
from sentence_transformers import SentenceTransformer

# Load Gemma 3
from transformers import pipeline

pipeline = pipeline(
    task="text-generation",
    model="google/gemma-3-4b-it",
    device_map="auto",
    dtype="auto"
)


device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "google/embeddinggemma-300M"
model = SentenceTransformer(model_id).to(device=device)

print(f"Device: {model.device}")
print(model)
print("Total number of parameters in the model:", sum([p.numel() for _, p in model.named_parameters()]))