"""RAPTOR-style hierarchical RAG index builder.

Builds a tree/DAG bottom-up by repeatedly:
  1. Chunking text into leaf nodes
  2. Embedding chunks
  3. Reducing dimensionality with UMAP, clustering with GMM (BIC model selection)
  4. Summarizing each cluster with an LLM to form parent nodes
  5. Re-embedding summaries and repeating until stopping criteria are met

Soft clustering (GMM) means a node can belong to multiple parents, producing a
DAG rather than a strict tree.  Export utilities in ``raptor_export.py`` handle
both strict-tree and DAG serialization for D3.js visualization.
"""
from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

from GraphReasoning.prompt_config import get_prompt

import numpy as np
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding client protocol — anything with .encode(text) -> np.ndarray
# ---------------------------------------------------------------------------

class EmbeddingClient(Protocol):
    def encode(self, text: str) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# RaptorNode — the core data unit for every tree level
# ---------------------------------------------------------------------------

@dataclass
class RaptorNode:
    id: str
    level: int                          # 0 = leaf, 1+ = summary
    type: str                           # "leaf" | "summary"
    text: str
    token_count: int
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    metadata: dict = field(default_factory=dict)

    def to_dict(self, include_embedding: bool = False) -> dict:
        d = {
            "id": self.id,
            "level": self.level,
            "type": self.type,
            "text": self.text,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }
        if include_embedding and self.embedding is not None:
            d["embedding"] = self.embedding.tolist()
        return d


# ---------------------------------------------------------------------------
# Edge between parent (summary) and child
# ---------------------------------------------------------------------------

@dataclass
class RaptorEdge:
    source: str           # parent id
    target: str           # child id
    weight: float         # GMM membership probability
    edge_type: str = "parent_child"

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "edge_type": self.edge_type,
        }


# ---------------------------------------------------------------------------
# The full RAPTOR index
# ---------------------------------------------------------------------------

@dataclass
class RaptorIndex:
    nodes: dict[str, RaptorNode] = field(default_factory=dict)
    edges: list[RaptorEdge] = field(default_factory=list)
    max_level: int = 0

    # -- Convenience accessors ------------------------------------------------

    def nodes_at_level(self, level: int) -> list[RaptorNode]:
        return [n for n in self.nodes.values() if n.level == level]

    def all_nodes(self) -> list[RaptorNode]:
        return list(self.nodes.values())

    def children_of(self, node_id: str) -> list[RaptorNode]:
        child_ids = {e.target for e in self.edges if e.source == node_id}
        return [self.nodes[cid] for cid in child_ids if cid in self.nodes]

    def parents_of(self, node_id: str) -> list[tuple[RaptorNode, float]]:
        """Return (parent_node, weight) pairs."""
        return [
            (self.nodes[e.source], e.weight)
            for e in self.edges
            if e.target == node_id and e.source in self.nodes
        ]

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def _make_token_counter(encoding_name: str = "cl100k_base") -> Callable[[str], int]:
    """Return a function that counts tokens using tiktoken."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding(encoding_name)
        return lambda text: len(enc.encode(text))
    except ImportError:
        # Rough fallback: ~4 chars per token for English
        logger.warning("tiktoken not installed — using approximate token counter")
        return lambda text: max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Step 1 — Sentence-aware token chunker
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')


def chunk_text(
    text: str,
    chunk_size: int = 100,
    chunk_overlap: int = 0,
    encoding_name: str = "cl100k_base",
) -> list[dict]:
    """Split *text* into chunks of roughly *chunk_size* tokens.

    Keeps sentences intact.  Returns a list of dicts with keys
    ``text``, ``token_count``, ``start_char``, ``end_char``.
    """
    count_tokens = _make_token_counter(encoding_name)
    sentences = _SENTENCE_RE.split(text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks: list[dict] = []
    current_sents: list[str] = []
    current_tokens = 0
    current_start = 0  # char offset into original text

    for sent in sentences:
        t = count_tokens(sent)
        if current_tokens + t > chunk_size and current_sents:
            chunk_text_joined = " ".join(current_sents)
            chunk_start = text.find(current_sents[0], current_start)
            chunk_end = chunk_start + len(chunk_text_joined)
            chunks.append({
                "text": chunk_text_joined,
                "token_count": current_tokens,
                "start_char": max(chunk_start, 0),
                "end_char": chunk_end,
            })
            # Overlap: keep trailing sentences that fit within overlap budget
            overlap_sents: list[str] = []
            overlap_tokens = 0
            for s in reversed(current_sents):
                st = count_tokens(s)
                if overlap_tokens + st > chunk_overlap:
                    break
                overlap_sents.insert(0, s)
                overlap_tokens += st
            current_sents = overlap_sents
            current_tokens = overlap_tokens
            current_start = chunk_end

        current_sents.append(sent)
        current_tokens += t

    if current_sents:
        chunk_text_joined = " ".join(current_sents)
        chunk_start = text.find(current_sents[0], current_start)
        chunks.append({
            "text": chunk_text_joined,
            "token_count": current_tokens,
            "start_char": max(chunk_start, 0),
            "end_char": chunk_start + len(chunk_text_joined),
        })

    return chunks


# ---------------------------------------------------------------------------
# Step 2 — Embed a batch of nodes
# ---------------------------------------------------------------------------

def embed_nodes(
    nodes: list[RaptorNode],
    embed_client: EmbeddingClient,
) -> None:
    """Embed each node's text in-place using *embed_client*."""
    for node in nodes:
        if node.embedding is None:
            node.embedding = np.asarray(embed_client.encode(node.text), dtype=np.float32)


# ---------------------------------------------------------------------------
# Step 3 — UMAP reduction + GMM clustering with BIC
# ---------------------------------------------------------------------------

def _reduce_umap(
    embeddings: np.ndarray,
    n_components: int = 10,
    n_neighbors: int = 15,
    min_dist: float = 0.0,
    metric: str = "cosine",
) -> np.ndarray:
    """Dimensionality reduction with UMAP.  Skipped if n_samples < 20."""
    if embeddings.shape[0] < max(20, n_neighbors + 1):
        logger.info("Too few samples for UMAP (%d) — skipping reduction", embeddings.shape[0])
        return embeddings

    import umap
    reducer = umap.UMAP(
        n_components=min(n_components, embeddings.shape[0] - 2),
        n_neighbors=min(n_neighbors, embeddings.shape[0] - 1),
        min_dist=min_dist,
        metric=metric,
        random_state=42,
    )
    return reducer.fit_transform(embeddings)


def cluster_nodes(
    embeddings: np.ndarray,
    max_k: int = 20,
    membership_threshold: float = 0.1,
) -> tuple[int, np.ndarray]:
    """GMM clustering with BIC model selection.

    Returns
    -------
    best_k : int
        Optimal number of clusters.
    membership : np.ndarray  (N, best_k)
        Soft cluster membership probabilities.
    """
    n = embeddings.shape[0]
    if n <= 2:
        # Everything in one cluster
        return 1, np.ones((n, 1), dtype=np.float64)

    upper_k = min(max_k, n)
    best_bic = np.inf
    best_k = 1
    best_gmm = None

    for k in range(1, upper_k + 1):
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                max_iter=200,
                random_state=42,
                reg_covar=1e-5,
            )
            gmm.fit(embeddings)
            bic = gmm.bic(embeddings)
            if bic < best_bic:
                best_bic = bic
                best_k = k
                best_gmm = gmm
        except Exception:
            logger.debug("GMM fit failed for k=%d, skipping", k)
            continue

    if best_gmm is None:
        return 1, np.ones((n, 1), dtype=np.float64)

    membership = best_gmm.predict_proba(embeddings)
    logger.info("GMM selected k=%d (BIC=%.1f)", best_k, best_bic)
    return best_k, membership


def assign_clusters(
    node_ids: list[str],
    membership: np.ndarray,
    threshold: float = 0.1,
) -> dict[int, list[tuple[str, float]]]:
    """Convert soft membership matrix to cluster → [(node_id, weight)] map.

    A node appears in every cluster where its probability exceeds *threshold*.
    """
    n_clusters = membership.shape[1]
    clusters: dict[int, list[tuple[str, float]]] = {k: [] for k in range(n_clusters)}

    for i, nid in enumerate(node_ids):
        for k in range(n_clusters):
            prob = float(membership[i, k])
            if prob >= threshold:
                clusters[k].append((nid, prob))

    # Drop empty clusters
    return {k: v for k, v in clusters.items() if v}


# ---------------------------------------------------------------------------
# Step 3b — Two-step clustering per RAPTOR paper (Section 3)
# ---------------------------------------------------------------------------

def _two_step_cluster(
    embeddings: np.ndarray,
    node_ids: list[str],
    *,
    max_k: int = 20,
    membership_threshold: float = 0.1,
    umap_dim: int = 10,
    n_neighbors_global: int = -1,
    n_neighbors_local: int = 10,
) -> dict[int, list[tuple[str, float]]]:
    """Two-step clustering as described in the RAPTOR paper.

    The paper states: *"Our algorithm varies n_neighbors to create a
    hierarchical clustering structure: it first identifies global clusters
    and then performs local clustering within these global clusters."*

    Step 1 — Global UMAP (large ``n_neighbors``) → GMM → coarse clusters.
    Step 2 — For each coarse cluster, local UMAP (small ``n_neighbors``)
             → GMM → fine-grained clusters.

    Returns ``cluster_label -> [(node_id, weight)]`` mapping.
    """
    n = len(node_ids)

    # Auto-select global n_neighbors (paper: preserves global structure)
    if n_neighbors_global < 0:
        n_neighbors_global = max(15, int(n ** 0.5))

    # Too few nodes for two-step — fall back to single pass
    if n < 6:
        reduced = _reduce_umap(embeddings, n_components=min(umap_dim, max(1, n - 2)))
        _, membership = cluster_nodes(reduced, max_k=min(max_k, n))
        return assign_clusters(node_ids, membership, threshold=membership_threshold)

    # ── Step 1: Global clustering ──────────────────────────────────────────
    logger.info("  Global UMAP (n_neighbors=%d)", min(n_neighbors_global, n - 1))
    global_reduced = _reduce_umap(
        embeddings,
        n_components=min(umap_dim, n - 2),
        n_neighbors=min(n_neighbors_global, n - 1),
    )
    global_k, global_membership = cluster_nodes(global_reduced, max_k=min(max_k, n))

    if global_k <= 1:
        # Unimodal at global level — return single-pass result
        return assign_clusters(node_ids, global_membership, threshold=membership_threshold)

    global_clusters = assign_clusters(node_ids, global_membership, threshold=membership_threshold)
    logger.info("  Global pass: %d clusters", len(global_clusters))

    # ── Step 2: Local clustering within each global cluster ────────────────
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    all_local: dict[int, list[tuple[str, float]]] = {}
    counter = 0

    for gc_label, gc_members in global_clusters.items():
        gc_ids = [nid for nid, _ in gc_members]

        if len(gc_ids) < 4:
            # Too small to sub-cluster — keep as-is
            all_local[counter] = gc_members
            counter += 1
            continue

        gc_embs = np.vstack([embeddings[id_to_idx[nid]] for nid in gc_ids])

        logger.info(
            "  Local clustering in global cluster %d (%d nodes, n_neighbors=%d)",
            gc_label, len(gc_ids), min(n_neighbors_local, len(gc_ids) - 1),
        )
        local_reduced = _reduce_umap(
            gc_embs,
            n_components=min(umap_dim, len(gc_ids) - 2),
            n_neighbors=min(n_neighbors_local, len(gc_ids) - 1),
        )
        local_k, local_membership = cluster_nodes(
            local_reduced, max_k=min(max_k, len(gc_ids)),
        )
        local_clusters = assign_clusters(
            gc_ids, local_membership, threshold=membership_threshold,
        )

        for lc_members in local_clusters.values():
            all_local[counter] = lc_members
            counter += 1

    logger.info("  Total fine-grained clusters: %d", len(all_local))
    return all_local


# ---------------------------------------------------------------------------
# Step 4 — Summarize clusters with an LLM
# ---------------------------------------------------------------------------

_SUMMARIZE_PROMPT = get_prompt("raptor", "summarize_user")


def summarize_cluster(
    texts: list[str],
    llm_call: Callable[[str], str],
    max_context_tokens: int = 4096,
    token_counter: Optional[Callable[[str], int]] = None,
) -> str:
    """Concatenate member texts and summarize with an LLM.

    If the concatenation exceeds *max_context_tokens*, it is truncated to fit
    (the recursive recluster approach handles this at a higher level).
    """
    if token_counter is None:
        token_counter = _make_token_counter()

    combined = "\n---\n".join(texts)
    tokens = token_counter(combined)
    if tokens > max_context_tokens:
        # Simple truncation fallback — caller should recluster instead
        ratio = max_context_tokens / tokens
        combined = combined[: int(len(combined) * ratio * 0.95)]

    prompt = _SUMMARIZE_PROMPT.format(text=combined)
    return llm_call(prompt)


# ---------------------------------------------------------------------------
# Step 5 — Recursive recluster for oversized clusters
# ---------------------------------------------------------------------------

def _recluster_if_needed(
    member_nodes: list[RaptorNode],
    embed_client: EmbeddingClient,
    llm_call: Callable[[str], str],
    max_context_tokens: int,
    token_counter: Callable[[str], int],
    membership_threshold: float,
    max_k: int,
) -> list[str]:
    """If combined tokens exceed budget, recursively sub-cluster and summarize.

    Returns a list of summary texts (one per sub-cluster that fits the budget).
    """
    total_tokens = sum(n.token_count for n in member_nodes)
    if total_tokens <= max_context_tokens:
        return ["\n---\n".join(n.text for n in member_nodes)]

    if len(member_nodes) <= 2:
        # Can't cluster further — just truncate
        return ["\n---\n".join(n.text for n in member_nodes)]

    # Sub-cluster
    embs = np.vstack([n.embedding for n in member_nodes])
    reduced = _reduce_umap(embs, n_components=min(10, len(member_nodes) - 2))
    _, sub_membership = cluster_nodes(reduced, max_k=min(max_k, len(member_nodes)))
    node_ids = [n.id for n in member_nodes]
    sub_clusters = assign_clusters(node_ids, sub_membership, threshold=membership_threshold)

    # If clustering couldn't split (k=1), truncate to avoid infinite recursion
    if len(sub_clusters) <= 1:
        logger.warning(
            "Recluster produced single cluster for %d nodes — truncating to fit budget",
            len(member_nodes),
        )
        return ["\n---\n".join(n.text for n in member_nodes)]

    node_map = {n.id: n for n in member_nodes}
    result_texts: list[str] = []
    for members in sub_clusters.values():
        sub_nodes = [node_map[nid] for nid, _ in members if nid in node_map]
        sub_total = sum(n.token_count for n in sub_nodes)
        if sub_total <= max_context_tokens:
            result_texts.append("\n---\n".join(n.text for n in sub_nodes))
        else:
            # Recurse deeper
            result_texts.extend(
                _recluster_if_needed(
                    sub_nodes, embed_client, llm_call,
                    max_context_tokens, token_counter,
                    membership_threshold, max_k,
                )
            )
    return result_texts


# ---------------------------------------------------------------------------
# Full RAPTOR tree builder
# ---------------------------------------------------------------------------

def build_raptor_index(
    text: str,
    embed_client: EmbeddingClient,
    llm_call: Callable[[str], str],
    *,
    doc_id: str = "",
    chunk_size: int = 100,
    chunk_overlap: int = 0,
    max_depth: int = 5,
    min_cluster_input: int = 3,
    max_k: int = 20,
    membership_threshold: float = 0.1,
    max_context_tokens: int = 4096,
    umap_dim: int = 10,
    n_neighbors_global: int = -1,
    n_neighbors_local: int = 10,
    encoding_name: str = "cl100k_base",
) -> RaptorIndex:
    """Build a full RAPTOR hierarchical index from raw text.

    Parameters
    ----------
    text : str
        The raw document text to index.
    embed_client : EmbeddingClient
        Anything with ``.encode(text) -> np.ndarray``.
    llm_call : callable
        ``llm_call(prompt: str) -> str`` that returns the LLM's response.
    doc_id : str
        Identifier for the source document.
    chunk_size / chunk_overlap : int
        Token-based chunking parameters.
    max_depth : int
        Maximum number of summarization layers.
    min_cluster_input : int
        Stop recursion if fewer than this many nodes at a level.
    max_k : int
        Upper bound on GMM cluster count.
    membership_threshold : float
        Minimum probability to assign a node to a cluster (soft clustering).
    max_context_tokens : int
        Max tokens allowed in a single summarization call.
    umap_dim : int
        Target dimensionality for UMAP reduction.
    n_neighbors_global : int
        UMAP n_neighbors for the global clustering pass (-1 = auto).
    n_neighbors_local : int
        UMAP n_neighbors for the local clustering pass within each
        global cluster.
    encoding_name : str
        tiktoken encoding for token counting.

    Returns
    -------
    RaptorIndex
        The complete hierarchical index with nodes, edges, and embeddings.
    """
    token_counter = _make_token_counter(encoding_name)
    index = RaptorIndex()

    # ----- Step 1: Chunk into leaves -----------------------------------------
    logger.info("Chunking document into leaves (chunk_size=%d)", chunk_size)
    raw_chunks = chunk_text(text, chunk_size, chunk_overlap, encoding_name)

    for i, chunk in enumerate(raw_chunks):
        node = RaptorNode(
            id=f"raptor_L0_{i:04d}",
            level=0,
            type="leaf",
            text=chunk["text"],
            token_count=chunk["token_count"],
            metadata={
                "doc_id": doc_id,
                "chunk_idx": i,
                "source_offset_start": chunk["start_char"],
                "source_offset_end": chunk["end_char"],
            },
        )
        index.nodes[node.id] = node

    leaves = index.nodes_at_level(0)
    logger.info("Created %d leaf nodes", len(leaves))

    if not leaves:
        return index

    # ----- Step 2: Embed leaves ----------------------------------------------
    logger.info("Embedding %d leaf nodes", len(leaves))
    embed_nodes(leaves, embed_client)

    # ----- Steps 3–5: Recursive clustering + summarization -------------------
    current_level_nodes = leaves

    for depth in range(1, max_depth + 1):
        n = len(current_level_nodes)
        if n < min_cluster_input:
            logger.info("Stopping at level %d: only %d nodes (< %d)", depth, n, min_cluster_input)
            break

        logger.info("Building level %d from %d nodes", depth, n)

        # Embed any un-embedded nodes (summaries from prior iteration)
        embed_nodes(current_level_nodes, embed_client)

        # Stack embeddings
        emb_matrix = np.vstack([node.embedding for node in current_level_nodes])
        node_ids = [node.id for node in current_level_nodes]

        # Two-step clustering per RAPTOR paper (global → local)
        clusters = _two_step_cluster(
            emb_matrix, node_ids,
            max_k=max_k,
            membership_threshold=membership_threshold,
            umap_dim=umap_dim,
            n_neighbors_global=n_neighbors_global,
            n_neighbors_local=n_neighbors_local,
        )

        if len(clusters) <= 1 and depth > 1:
            logger.info("Stopping at level %d: clustering produced single cluster", depth)
            break

        logger.info("Level %d: %d clusters", depth, len(clusters))

        # Build parent nodes for each cluster
        new_parents: list[RaptorNode] = []
        node_map = {n.id: n for n in current_level_nodes}

        for cluster_label, members in clusters.items():
            child_nodes = [node_map[nid] for nid, _ in members if nid in node_map]
            if not child_nodes:
                continue

            # Check token budget — recluster if needed
            combined_tokens = sum(cn.token_count for cn in child_nodes)
            if combined_tokens > max_context_tokens:
                text_chunks = _recluster_if_needed(
                    child_nodes, embed_client, llm_call,
                    max_context_tokens, token_counter,
                    membership_threshold, max_k,
                )
            else:
                text_chunks = ["\n---\n".join(cn.text for cn in child_nodes)]

            # Summarize each text chunk into a parent
            for chunk_i, chunk_text_str in enumerate(text_chunks):
                summary = summarize_cluster(
                    [chunk_text_str], llm_call,
                    max_context_tokens, token_counter,
                )
                parent_id = f"raptor_L{depth}_{cluster_label:04d}"
                if len(text_chunks) > 1:
                    parent_id += f"_sub{chunk_i}"

                parent_node = RaptorNode(
                    id=parent_id,
                    level=depth,
                    type="summary",
                    text=summary,
                    token_count=token_counter(summary),
                    metadata={
                        "child_ids": [nid for nid, _ in members],
                        "cluster_label": cluster_label,
                    },
                )
                index.nodes[parent_node.id] = parent_node
                new_parents.append(parent_node)

                # Create edges from parent to children
                for child_id, weight in members:
                    index.edges.append(
                        RaptorEdge(
                            source=parent_node.id,
                            target=child_id,
                            weight=weight,
                        )
                    )

        if not new_parents:
            logger.info("Stopping at level %d: no parents created", depth)
            break

        index.max_level = depth
        current_level_nodes = new_parents
        logger.info("Level %d complete: %d parent nodes", depth, len(new_parents))

    logger.info(
        "RAPTOR index complete: %d nodes, %d edges, %d levels",
        index.node_count, index.edge_count, index.max_level,
    )
    return index
