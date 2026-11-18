
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


# ---------------------------------------------------------------------------
# Auditable RAG: chunk metadata, separate retrieval evaluation, grounded
# answers, and corpus provenance. Everything is local and deterministic.
# ---------------------------------------------------------------------------
import hashlib

import pandas as pd


def clean_stix_text(text):
    """Strip STIX citation markers, markdown links, HTML tags, and code ticks."""
    value = str(text or "")
    value = re.sub(r"\(Citation:[^)]*\)", "", value)
    value = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"<[^>]+>", "", value)
    value = value.replace("`", "")
    return re.sub(r"\s+", " ", value).strip()


def chunk_documents(
    documents,
    text_column="text",
    id_column="external_id",
    source_column="source",
    date_column="modified",
    max_words=60,
):
    """
    Split documents into retrievable chunks. Every chunk carries a chunk ID,
    its parent document ID, source, and date so any answer stays auditable.
    """
    rows = []
    for doc in documents.itertuples(index=False):
        record = doc._asdict()
        words = clean_stix_text(record.get(text_column)).split()
        for start in range(0, max(len(words), 1), max_words):
            piece = " ".join(words[start:start + max_words]).strip()
            if not piece:
                continue
            doc_id = record.get(id_column)
            rows.append({
                "chunk_id": f"{doc_id}#{start // max_words}",
                "doc_id": doc_id,
                "source": record.get(source_column),
                "date": str(record.get(date_column) or "")[:10],
                "trusted_source": bool(record.get("trusted_source", False)),
                "text": piece,
            })
    return pd.DataFrame(rows)


def evaluate_retrieval(queries, chunks, vectorizer, matrix, top_k=5, accept_subtechniques=True):
    """
    Score the retrieval layer on its own, before any answer is written.

    `queries` needs columns `query` and `expected_doc_id`. A hit means the
    expected document (or, optionally, one of its sub-techniques) appears in
    the top_k chunks.
    """
    def matches(doc_id, expected):
        doc_id = str(doc_id)
        return doc_id == expected or (accept_subtechniques and doc_id.startswith(expected + "."))

    rows = []
    for item in queries.itertuples(index=False):
        results = retrieve(item.query, chunks, vectorizer, matrix, top_k=top_k)
        doc_ids = results["doc_id"].tolist()
        positions = [i for i, d in enumerate(doc_ids) if matches(d, item.expected_doc_id)]
        rank = positions[0] + 1 if positions else None
        rows.append({
            "query": item.query,
            "expected_doc_id": item.expected_doc_id,
            "hit_at_k": rank is not None,
            "rank": rank,
            "reciprocal_rank": 1.0 / rank if rank else 0.0,
            "top_doc_id": doc_ids[0] if doc_ids else None,
            "top_score": float(results["retrieval_score"].iloc[0]) if len(results) else 0.0,
        })
    return pd.DataFrame(rows)


def grounded_answer(
    query,
    chunks,
    vectorizer,
    matrix,
    top_k=3,
    min_score=0.15,
    max_sentences=2,
):
    """
    A deliberately simple, extractive answer generator.

    It only writes sentences copied from retrieved chunks and cites the
    chunk each sentence came from. If no chunk clears `min_score`, it answers
    "Insufficient data" instead of guessing. Swap in an LLM later; keep the
    contract: every claim cites a chunk, or the system abstains.
    """
    results = retrieve(query, chunks, vectorizer, matrix, top_k=top_k)
    usable = results[results["retrieval_score"] >= min_score]
    if usable.empty:
        return {
            "query": query,
            "answer": "Insufficient data: no retrieved source supports an answer.",
            "claims": [],
            "abstained": True,
        }

    claims = []
    for chunk in usable.itertuples(index=False):
        candidates = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk.text)
            if len(s.split()) >= 6
        ]
        if not candidates or any(c["sentence"] == candidates[0] for c in claims):
            continue
        claims.append({
            "sentence": candidates[0],
            "chunk_id": chunk.chunk_id,
            "source": chunk.source,
            "date": chunk.date,
            "score": round(float(chunk.retrieval_score), 3),
        })
        if len(claims) >= max_sentences:
            break

    answer = " ".join(f"{c['sentence']} [{c['chunk_id']}]" for c in claims)
    return {"query": query, "answer": answer, "claims": claims, "abstained": False}


def groundedness_check(answer, chunks):
    """
    Generation check, separate from retrieval: does every cited sentence
    actually appear in the chunk it cites?
    """
    if answer["abstained"]:
        return pd.DataFrame([{"claim": answer["answer"], "chunk_id": None, "supported": True}])
    by_id = chunks.set_index("chunk_id")["text"]
    rows = []
    for claim in answer["claims"]:
        chunk_id = claim["chunk_id"]
        rows.append({
            "claim": claim["sentence"],
            "chunk_id": chunk_id,
            "supported": chunk_id in by_id.index and claim["sentence"] in by_id[chunk_id],
        })
    return pd.DataFrame(rows)


def resolve_technique_ids(techniques, expected):
    """
    Map (technique name, tactic) pairs to live ATT&CK IDs. Names are reused
    across tactics and IDs move between releases, so never hardcode IDs.
    """
    rows = []
    for name, tactic in expected:
        match = techniques[
            (techniques["name"] == name)
            & techniques["tactics"].str.contains(tactic, na=False)
        ]
        rows.append({
            "name": name,
            "tactic": tactic,
            "external_id": match["external_id"].iloc[0] if len(match) else None,
        })
    return pd.DataFrame(rows)


def content_hash(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16]


def corpus_manifest(chunks):
    """Snapshot of a known-good corpus: one hash per chunk ID."""
    return dict(zip(chunks["chunk_id"], chunks["text"].map(content_hash)))


def diff_against_manifest(chunks, manifest):
    """Label each chunk as unchanged, modified, or new relative to a baseline."""
    out = chunks.copy()

    def status(row):
        if row.chunk_id not in manifest:
            return "new"
        return "unchanged" if manifest[row.chunk_id] == content_hash(row.text) else "modified"

    out["provenance_status"] = [status(r) for r in out.itertuples(index=False)]
    return out


def quarantine(chunks_with_status):
    """
    Nothing new, modified, or untrusted reaches the trusted index without
    review. Instruction-pattern hits are attached as a review annotation,
    not used as the gate: pattern filters fire on legitimate text too.
    Returns (trusted, quarantined).
    """
    flags = chunks_with_status["text"].apply(prompt_injection_flags)
    suspicious = flags.apply(lambda f: f["suspicious"])
    held = (
        (chunks_with_status["provenance_status"] != "unchanged")
        | ~chunks_with_status["trusted_source"].astype(bool)
    )
    annotated = chunks_with_status.assign(injection_pattern=suspicious)
    return (
        annotated[~held].reset_index(drop=True),
        annotated[held].reset_index(drop=True),
    )


def restore_known_good(snapshot, approved=None):
    """
    Recovery: rebuild the corpus from the known-good snapshot, plus only the
    items a reviewer explicitly approved.
    """
    parts = [snapshot]
    if approved is not None and len(approved):
        parts.append(approved[snapshot.columns])
    return pd.concat(parts, ignore_index=True)
