# --- Helper Functions for Semantic Search ---

def _calculate_best_match(similarities):
    print(similarities)
    if similarities is None or similarities.nelement() == 0:
        return None, 0.0

    # Find the index and value of the highest score
    best_index = similarities.argmax().item()
    best_score = similarities[0, best_index].item()

    return best_index, best_score

def find_best_category(model, query, candidates):
    """
    Finds the most relevant category from a list of candidates.

    Args:
        model: The SentenceTransformer model.
        query: The user's query string.
        candidates: A list of category name strings.

    Returns:
        A tuple containing the index of the best category and its similarity score.
    """
    if not candidates:
        return None, 0.0

    # Encode the query and candidate categories for classification
    query_embedding = model.encode(query, prompt_name="Classification")
    candidate_embeddings = model.encode(candidates, prompt_name="Classification")

    print(candidates)
    return _calculate_best_match(model.similarity(query_embedding, candidate_embeddings))

def find_best_doc(model, query, candidates):
    """
    Finds the most relevant document from a list of candidates.

    Args:
        model: The SentenceTransformer model.
        query: The user's query string.
        candidates: A list of document dictionaries, each with 'title' and 'content'.

    Returns:
        A tuple containing the index of the best document and its similarity score.
    """
    if not candidates:
        return None, 0.0

    # Encode the query for retrieval
    query_embedding = model.encode(query, prompt_name="Retrieval-query")

    # Encode the document for similarity check
    doc_texts = [
        f"title: {doc.get('title', 'none')} | text: {doc.get('content', '')}"
        for doc in candidates
    ]
    candidate_embeddings = model.encode(doc_texts)

    print([doc['title'] for doc in candidates])

    # Calculate cosine similarity
    return _calculate_best_match(model.similarity(query_embedding, candidate_embeddings))

