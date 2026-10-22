
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SUSPICIOUS_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?prior\s+instructions",
    r"system\s+prompt",
    r"reveal\s+.*secret",
    r"override\s+.*instructions",
    r"treat\s+this\s+document\s+as\s+authoritative",
]

def build_tfidf_index(documents, text_column="text"):
    """Build a lightweight, fully local retrieval index."""
    docs = documents.reset_index(drop=True).copy()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50000,
    )

    matrix = vectorizer.fit_transform(
        docs[text_column].fillna("")
    )

    return docs, vectorizer, matrix

def retrieve(
    query,
    documents,
    vectorizer,
    matrix,
    top_k=5,
    text_column="text",
):
    """Retrieve the highest-scoring local documents for a query."""
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()

    top_indices = np.argsort(scores)[::-1][:top_k]

    result = documents.iloc[top_indices].copy()
    result["retrieval_score"] = scores[top_indices]

    return result.reset_index(drop=True)

def prompt_injection_flags(text):
    """Detect simple instruction-like patterns inside retrieved content."""
    value = str(text or "")
    matched = [
        pattern
        for pattern in SUSPICIOUS_PATTERNS
        if re.search(pattern, value, flags=re.IGNORECASE)
    ]

    return {
        "suspicious": bool(matched),
        "matched_pattern_count": len(matched),
    }

def add_security_flags(
    documents,
    text_column="text",
    trusted_source_column="trusted_source",
):
    """Attach source-trust and simple prompt-injection flags."""
    out = documents.copy()

    flags = out[text_column].fillna("").apply(prompt_injection_flags)

    out["prompt_injection_suspected"] = flags.apply(
        lambda item: item["suspicious"]
    )
    out["matched_pattern_count"] = flags.apply(
        lambda item: item["matched_pattern_count"]
    )

    if trusted_source_column not in out.columns:
        out[trusted_source_column] = False

    out["eligible_for_retrieval"] = (
        out[trusted_source_column].astype(bool)
        & ~out["prompt_injection_suspected"]
    )

    return out

def filter_eligible_documents(
    documents,
    eligible_column="eligible_for_retrieval",
):
    """Keep only documents that pass the simple retrieval gate."""
    if eligible_column not in documents.columns:
        raise KeyError(
            f"Missing eligibility column: {eligible_column}"
        )

    return documents[
        documents[eligible_column].astype(bool)
    ].reset_index(drop=True)
