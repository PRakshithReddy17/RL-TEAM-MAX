# -*- coding: utf-8 -*-
"""
BM25 Evidence Retriever

A lightweight BM25 implementation for retrieving synthetic evidence.
"""

import numpy as np
import pandas as pd
import re
from typing import List, Dict, Any


def lexical_tokens(text: str) -> List[str]:
    """Convert text to lowercase lexical tokens."""
    return re.findall(r"[a-z0-9]+", str(text).lower())


class BM25Retriever:
    """BM25-based evidence retriever for synthetic records."""

    def __init__(self, evidence_docs: pd.DataFrame):
        """
        Initialize the BM25 retriever.

        Args:
            evidence_docs: DataFrame with 'evidence' column
        """
        from rank_bm25 import BM25Okapi

        # Deduplicate evidence documents while retaining a stable document ID
        self.evidence_docs = (
            evidence_docs[["trail_id", "evidence"]]
            .drop_duplicates(subset=["evidence"])
            .reset_index(drop=True)
        )
        self.evidence_docs["doc_id"] = [
            f"syn_doc_{i:04d}" for i in range(len(self.evidence_docs))
        ]

        corpus_tokens = [lexical_tokens(x) for x in self.evidence_docs["evidence"]]
        self.bm25 = BM25Okapi(corpus_tokens)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve top-k evidence passages for a query.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of dicts with doc_id, trail_id, evidence, retrieval_score
        """
        query_tokens = lexical_tokens(query)
        scores = self.bm25.get_scores(query_tokens)

        if len(scores) == 0:
            return []

        order = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in order:
            results.append({
                "doc_id": self.evidence_docs.iloc[int(idx)]["doc_id"],
                "trail_id": self.evidence_docs.iloc[int(idx)]["trail_id"],
                "evidence": self.evidence_docs.iloc[int(idx)]["evidence"],
                "retrieval_score": float(scores[int(idx)])
            })
        return results

    def evaluate_retrieval(self, claims_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate retrieval quality on a dataset of claims.

        Args:
            claims_df: DataFrame with 'claim' and 'trail_id' columns

        Returns:
            Dictionary with retrieval metrics
        """
        retrieval_rows = []
        for row in claims_df.itertuples():
            hits = self.retrieve(row.claim, top_k=3)
            hit_trails = [h["trail_id"] for h in hits]

            retrieval_rows.append({
                "trail_id": row.trail_id,
                "step_id": getattr(row, 'step_id', ''),
                "gold_label": row.gold_label,
                "top1_source_match": bool(hit_trails and hit_trails[0] == row.trail_id),
                "top3_source_match": row.trail_id in hit_trails,
                "top1_score": hits[0]["retrieval_score"] if hits else 0.0,
                "retrieved_evidence": hits[0]["evidence"] if hits else ""
            })

        retrieval_eval = pd.DataFrame(retrieval_rows)
        metrics = {
            "test_claims": int(len(retrieval_eval)),
            "top1_source_match_rate": float(retrieval_eval["top1_source_match"].mean()),
            "top3_source_match_rate": float(retrieval_eval["top3_source_match"].mean())
        }

        return metrics, retrieval_eval


__all__ = [
    "BM25Retriever",
    "lexical_tokens"
]