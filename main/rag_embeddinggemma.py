import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from utils.sim_knowledge_base import corp_knowledge_base
from utils.semantic_search import *

# Load Gemma 3
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

print("Available tasks:")
for name, prefix in model.prompts.items():
  print(f" {name}: \"{prefix}\"")

question = "How do I reset my password?" # @param ["How many days of annual paid leave do I get?", "How do I reset my password?", "What travel expenses can be reimbursed for a business trip?", "Can I receive personal packages at the office?"] {type:"string", allow-input: true}

# Define a minimum confidence threshold for a match to be considered valid
similarity_threshold = 0.4 # @param {"type":"slider","min":0,"max":1,"step":0.1}

# --- Main Search Logic ---

# In your application, `best_document` would result from a search.
# We initialize it to None to ensure it always exists.
best_document = None

# 1. Find the most relevat category
print("Step 1: Finding the best category...")
categories = [item["category"] for item in corp_knowledge_base]
best_category_index, category_score = find_best_category(
    model, question, categories
)

# Check if the category score meets the threshold
if category_score < similarity_threshold:
    print(f" `-> 🤷 No relevant category found. The highest score was only {category_score:.2f}.")
else:
    best_category = corp_knowledge_base[best_category_index]
    print(f" `-> ✅ Category Found: '{best_category['category']}' (Score: {category_score:.2f})")

    # 2. Find the most relevant document ONLY if a good category was found
    print("\nStep 2: Finding the best document in that category...")
    best_document_index, document_score = find_best_doc(
        model, question, best_category["documents"]
    )

    # Check if the document score meets the threshold
    if document_score < similarity_threshold:
        print(f" `-> 🤷 No relevant document found. The highest score was only {document_score:.2f}.")
    else:
        best_document = best_category["documents"][best_document_index]
        # 3. Display the final successful result
        print(f" `-> ✅ Document Found: '{best_document['title']}' (Score: {document_score:.2f})")


# GENERATE ANSWER
qa_prompt_template = """Answer the following QUESTION based only on the CONTEXT provided. If the answer cannot be found in the CONTEXT, write "I don't know."

---
CONTEXT:
{context}
---
QUESTION:
{question}
"""

# First, check if a valid document was found before proceeding.
if best_document and "content" in best_document:
    # If the document exists and has a "content" key, generate the answer.
    context = best_document["content"]

    prompt = qa_prompt_template.format(context=context, question=question)

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        },
    ]

    print("Question🙋‍♂️: " + question)
    # This part assumes your pipeline and response parsing logic are correct
    answer = pipeline(messages, max_new_tokens=256, disable_compile=True)[0]["generated_text"][1]["content"]
    print("Using document: " + best_document["title"])
    print("Answer🤖: " + answer)

else:
    # If best_document is None or doesn't have content, give a direct response.
    print("Question🙋‍♂️: " + question)
    print("Answer🤖: I'm sorry, I could not find a relevant document to answer that question.")

