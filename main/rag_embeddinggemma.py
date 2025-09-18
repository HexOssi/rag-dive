import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from utils.sim_knowledge_base import corp_knowledge_base
from utils.semantic_search import *

def load_models():
    """Loads the generation and embedding models."""
    print("Loading models...")
    
    # Load Gemma 3 for generation
    gen_pipeline = pipeline(
        task="text-generation",
        model="google/gemma-3-4b-it",
        device_map="auto",
        dtype="auto"
    )

    # Load embeddinggemma for retrieval
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model_id = "google/embeddinggemma-300M"
    embedding_model = SentenceTransformer(embedding_model_id).to(device=device)
    print(f"Embedding model loaded on: {embedding_model.device}")
    return gen_pipeline, embedding_model


def retrieve_document(model, question, knowledge_base, threshold):
    """Performs a two-step retrieval to find the most relevant document."""
    # 1. Find the most relevant category
    print("Step 1: Finding the best category...")
    categories = [item["category"] for item in knowledge_base]
    best_category_index, category_score = find_best_category(
        model, question, categories
    )

    if category_score < threshold:
        print(f" `-> 🤷 No relevant category found. The highest score was only {category_score:.2f}.")
        return None

    best_category = knowledge_base[best_category_index]
    print(f" `-> ✅ Category Found: '{best_category['category']}' (Score: {category_score:.2f})")

    # 2. Find the most relevant document in that category
    print("\nStep 2: Finding the best document in that category...")
    best_document_index, document_score = find_best_doc(
        model, question, best_category["documents"]
    )

    if document_score < threshold:
        print(f" `-> 🤷 No relevant document found. The highest score was only {document_score:.2f}.")
        return None

    best_document = best_category["documents"][best_document_index]
    print(f" `-> ✅ Document Found: '{best_document['title']}' (Score: {document_score:.2f})")
    return best_document


def generate_answer(gen_pipeline, question, document):
    """Generates an answer using the retrieved document as context."""
    qa_prompt_template = """Answer the following QUESTION based only on the CONTEXT provided. If the answer cannot be found in the CONTEXT, write "I don't know."

---
CONTEXT:
{context}
---
QUESTION:
{question}
"""
    print("\nQuestion🙋‍♂️: " + question)

    if document and "content" in document:
        context = document["content"]
        prompt = qa_prompt_template.format(context=context, question=question)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        # Generate and parse the answer robustly
        response = gen_pipeline(messages, max_new_tokens=256, disable_compile=True)
        try:
            answer = response[0]["generated_text"][1]["content"]
            print("Using document: " + document["title"])
            print("Answer🤖: " + answer)
        except (IndexError, KeyError, TypeError) as e:
            print(f"Error parsing model response: {e}")
            print("Answer🤖: I'm sorry, I encountered an error while generating the answer.")
    else:
        print("Answer🤖: I'm sorry, I could not find a relevant document to answer that question.")

def main():
    """Main function to run the RAG pipeline."""
    generation_pipeline, embedding_model = load_models()

    question = "How do I reset my password?"
    similarity_threshold = 0.4

    # --- Main RAG Logic ---
    best_document = retrieve_document(embedding_model, question, corp_knowledge_base, similarity_threshold)
    generate_answer(generation_pipeline, question, best_document)

if __name__ == "__main__":
    main()
