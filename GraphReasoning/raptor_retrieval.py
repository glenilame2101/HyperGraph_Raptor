"""RAPTOR retrieval strategies.

Two retrieval methods from the RAPTOR paper:

1. **Tree Traversal** — Start at the top of the tree, pick top-k children
   per layer, expand downward.  The ``depth`` parameter controls specificity.

2. **Collapsed Tree** — Flatten all nodes across all levels into a single
   pool, rank by cosine similarity, greedily fill a token budget.  This is
   the stronger method in the paper.

Both methods can be accelerated with FAISS for nearest-neighbor search.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .raptor_tree import EmbeddingClient, RaptorIndex, RaptorNode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cosine similarity helpers
# ---------------------------------------------------------------------------

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _cosine_sim_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a query vector and a matrix of vectors."""
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    matrix_norm = matrix / norms
    return matrix_norm @ query_norm


# ---------------------------------------------------------------------------
# FAISS index builder (optional acceleration)
# ---------------------------------------------------------------------------

class FaissIndex:
    """Thin wrapper around a FAISS flat inner-product index.

    Works on L2-normalized vectors so inner product = cosine similarity.
    """

    def __init__(self, node_ids: list[str], embeddings: np.ndarray):
        import faiss

        self.node_ids = node_ids
        dim = embeddings.shape[1]

        # L2-normalize for cosine via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        self._normed = (embeddings / norms).astype(np.float32)

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self._normed)
        logger.info("FAISS index built: %d vectors, dim=%d", len(node_ids), dim)

    def search(self, query: np.ndarray, k: int = 100) -> list[tuple[str, float]]:
        """Return top-k ``(node_id, score)`` pairs."""
        q = query.astype(np.float32).reshape(1, -1)
        q = q / (np.linalg.norm(q) + 1e-10)
        scores, indices = self.index.search(q, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self.node_ids[idx], float(score)))
        return results


def build_faiss_index(index: RaptorIndex) -> FaissIndex:
    """Build a FAISS index over all nodes in a RaptorIndex."""
    node_ids = []
    embeddings = []
    for node in index.all_nodes():
        if node.embedding is not None:
            node_ids.append(node.id)
            embeddings.append(node.embedding)
    if not embeddings:
        raise ValueError("No embeddings found in index")
    matrix = np.vstack(embeddings).astype(np.float32)
    return FaissIndex(node_ids, matrix)


# ---------------------------------------------------------------------------
# Strategy 1 — Collapsed-tree retrieval (recommended)
# ---------------------------------------------------------------------------

def collapsed_tree_retrieve(
    query_embedding: np.ndarray,
    index: RaptorIndex,
    max_tokens: int = 4096,
    top_k_candidates: int = 200,
    faiss_index: Optional[FaissIndex] = None,
) -> list[tuple[RaptorNode, float]]:
    """Flatten all nodes, rank by similarity, greedily fill token budget.

    Parameters
    ----------
    query_embedding : np.ndarray
        The embedded query vector.
    index : RaptorIndex
        The RAPTOR index to search.
    max_tokens : int
        Maximum total tokens in the retrieved context.
    top_k_candidates : int
        Pre-filter to this many candidates before greedy selection.
    faiss_index : FaissIndex, optional
        Pre-built FAISS index for acceleration.

    Returns
    -------
    list of (RaptorNode, score)
        Retrieved nodes ordered by similarity, within token budget.
    """
    if faiss_index is not None:
        ranked = faiss_index.search(query_embedding, k=top_k_candidates)
    else:
        # Brute-force cosine
        all_nodes = [n for n in index.all_nodes() if n.embedding is not None]
        if not all_nodes:
            return []
        matrix = np.vstack([n.embedding for n in all_nodes])
        scores = _cosine_sim_batch(query_embedding, matrix)
        ranked_indices = np.argsort(scores)[::-1][:top_k_candidates]
        ranked = [(all_nodes[i].id, float(scores[i])) for i in ranked_indices]

    # Greedy token-budget filling
    result: list[tuple[RaptorNode, float]] = []
    total_tokens = 0

    for node_id, score in ranked:
        if node_id not in index.nodes:
            continue
        node = index.nodes[node_id]
        if total_tokens + node.token_count > max_tokens:
            continue  # skip this node, try smaller ones
        result.append((node, score))
        total_tokens += node.token_count

    logger.info(
        "Collapsed-tree retrieval: %d nodes, %d tokens (budget=%d)",
        len(result), total_tokens, max_tokens,
    )
    return result


# ---------------------------------------------------------------------------
# Strategy 2 — Tree-traversal retrieval
# ---------------------------------------------------------------------------

def tree_traverse_retrieve(
    query_embedding: np.ndarray,
    index: RaptorIndex,
    top_k_per_level: int = 3,
    max_depth: Optional[int] = None,
) -> list[tuple[RaptorNode, float]]:
    """Start at the highest level, pick top-k, expand children, repeat.

    Parameters
    ----------
    query_embedding : np.ndarray
        The embedded query vector.
    index : RaptorIndex
        The RAPTOR index to search.
    top_k_per_level : int
        Number of nodes to select at each level.
    max_depth : int, optional
        How many levels to traverse down (default: all levels).

    Returns
    -------
    list of (RaptorNode, score)
        All selected nodes across traversed levels.
    """
    if max_depth is None:
        max_depth = index.max_level

    selected: list[tuple[RaptorNode, float]] = []

    # Start from the highest level
    start_level = index.max_level
    candidates = [
        n for n in index.nodes_at_level(start_level)
        if n.embedding is not None
    ]

    for level in range(start_level, max(start_level - max_depth - 1, -1), -1):
        if not candidates:
            break

        # Score candidates
        emb_matrix = np.vstack([c.embedding for c in candidates])
        scores = _cosine_sim_batch(query_embedding, emb_matrix)

        # Pick top-k
        k = min(top_k_per_level, len(candidates))
        top_indices = np.argsort(scores)[::-1][:k]

        top_nodes = []
        for idx in top_indices:
            node = candidates[idx]
            score = float(scores[idx])
            selected.append((node, score))
            top_nodes.append(node)

        # Expand: children of selected nodes become next candidates
        next_candidates = []
        for node in top_nodes:
            children = index.children_of(node.id)
            next_candidates.extend(
                c for c in children if c.embedding is not None
            )

        # Deduplicate
        seen = set()
        candidates = []
        for c in next_candidates:
            if c.id not in seen:
                seen.add(c.id)
                candidates.append(c)

    logger.info("Tree-traversal retrieval: %d nodes selected", len(selected))
    return selected


# ---------------------------------------------------------------------------
# High-level query function
# ---------------------------------------------------------------------------

def query_raptor(
    query: str,
    index: RaptorIndex,
    embed_client: EmbeddingClient,
    *,
    method: str = "collapsed",
    max_tokens: int = 4096,
    top_k: int = 5,
    faiss_index: Optional[FaissIndex] = None,
) -> list[tuple[RaptorNode, float]]:
    """Query the RAPTOR index.

    Parameters
    ----------
    query : str
        Natural language query.
    index : RaptorIndex
        The RAPTOR index.
    embed_client : EmbeddingClient
        Used to embed the query.
    method : str
        ``"collapsed"`` (default, recommended) or ``"tree"``.
    max_tokens : int
        Token budget for collapsed retrieval.
    top_k : int
        top-k per level for tree traversal.
    faiss_index : FaissIndex, optional
        Pre-built FAISS index for acceleration.

    Returns
    -------
    list of (RaptorNode, score)
    """
    query_emb = np.asarray(embed_client.encode(query), dtype=np.float32)

    if method == "collapsed":
        return collapsed_tree_retrieve(
            query_emb, index,
            max_tokens=max_tokens,
            faiss_index=faiss_index,
        )
    elif method == "tree":
        return tree_traverse_retrieve(
            query_emb, index,
            top_k_per_level=top_k,
        )
    else:
        raise ValueError(f"Unknown retrieval method: {method!r}. Use 'collapsed' or 'tree'.")
