def calculate_context_overlap(reference_contexts, retrieved_contexts):
    """Calculate overlap between reference and retrieved contexts"""
    if not reference_contexts or not retrieved_contexts:
        return 0.0

    # Convert to sets of words for comparison
    ref_words = set()
    for ctx in reference_contexts:
        ref_words.update(ctx.lower().split())

    retr_words = set()
    for ctx in retrieved_contexts:
        retr_words.update(ctx.lower().split())

    if not ref_words or not retr_words:
        return 0.0

    intersection = len(ref_words.intersection(retr_words))
    union = len(ref_words.union(retr_words))

    return intersection / union if union > 0 else 0.0


def calculate_precision_at_k(reference_contexts, retrieved_contexts, k=5):
    """Calculate precision at k for context retrieval"""
    if not reference_contexts or not retrieved_contexts:
        return 0.0

    # Simple word-based precision calculation
    ref_words = set()
    for ctx in reference_contexts:
        ref_words.update(ctx.lower().split())

    retr_words = []
    for ctx in retrieved_contexts:
        retr_words.extend(ctx.lower().split())
    retr_words = retr_words[:k]

    if not retr_words:
        return 0.0

    relevant_words = sum(1 for word in retr_words if word in ref_words)
    return relevant_words / len(retr_words)


def calculate_recall_at_k(reference_contexts, retrieved_contexts, k=5):
    """Calculate recall at k for context retrieval"""
    if not reference_contexts or not retrieved_contexts:
        return 0.0

    ref_words = set()
    for ctx in reference_contexts:
        ref_words.update(ctx.lower().split())

    retr_words = set()
    for ctx in retrieved_contexts:
        retr_words.update(ctx.lower().split())
    retr_words = set(list(retr_words)[:k])

    if not ref_words:
        return 0.0

    relevant_words = len(ref_words.intersection(retr_words))
    return relevant_words / len(ref_words)
