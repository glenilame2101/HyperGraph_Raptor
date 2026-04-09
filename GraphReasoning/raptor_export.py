"""Export a RaptorIndex to D3.js-friendly JSON formats.

Two export strategies:

1. **Strict Tree** — nested JSON for ``d3.hierarchy`` / ``d3.tree``.
   Each child is assigned to its single most-likely parent (max membership
   probability).  Alternatively, nodes can be duplicated under each parent.

2. **DAG / Graph** — flat ``{nodes, links}`` JSON for force-directed or
   layered DAG layouts.  Preserves soft-clustering edges with weights.

Embeddings are stored separately in ``.npz`` to keep viz JSON lightweight.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .raptor_tree import RaptorEdge, RaptorIndex, RaptorNode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding persistence
# ---------------------------------------------------------------------------

def save_embeddings_npz(index: RaptorIndex, path: str | Path) -> Path:
    """Save all node embeddings to a compressed ``.npz`` file.

    Keys are node IDs; values are float32 vectors.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for nid, node in index.nodes.items():
        if node.embedding is not None:
            arrays[nid] = node.embedding.astype(np.float32)
    np.savez_compressed(str(out), **arrays)
    logger.info("Saved %d embeddings to %s", len(arrays), out)
    return out


def load_embeddings_npz(index: RaptorIndex, path: str | Path) -> int:
    """Load embeddings from ``.npz`` back into an existing index.

    Returns the number of embeddings loaded.
    """
    data = np.load(str(path))
    loaded = 0
    for nid in data.files:
        if nid in index.nodes:
            index.nodes[nid].embedding = data[nid]
            loaded += 1
    return loaded


# ---------------------------------------------------------------------------
# Full node catalog (with text + metadata, no embeddings)
# ---------------------------------------------------------------------------

def export_nodes_json(index: RaptorIndex, path: str | Path) -> Path:
    """Export all nodes (text, metadata, token_count) to JSON.

    This is the canonical source of truth — viz exports are derived from it.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "node_count": index.node_count,
        "edge_count": index.edge_count,
        "max_level": index.max_level,
        "nodes": [n.to_dict(include_embedding=False) for n in index.all_nodes()],
        "edges": [e.to_dict() for e in index.edges],
    }
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Exported %d nodes + %d edges to %s", index.node_count, index.edge_count, out)
    return out


# ---------------------------------------------------------------------------
# Strategy 1 — Strict Tree JSON (for d3.hierarchy)
# ---------------------------------------------------------------------------

def _build_tree_edges(index: RaptorIndex) -> dict[str, str]:
    """Assign each child to its single strongest parent (max weight).

    Returns a mapping ``child_id -> parent_id``.
    """
    best_parent: dict[str, tuple[str, float]] = {}
    for edge in index.edges:
        child = edge.target
        if child not in best_parent or edge.weight > best_parent[child][1]:
            best_parent[child] = (edge.source, edge.weight)
    return {child: parent for child, (parent, _) in best_parent.items()}


def _build_nested(
    node_id: str,
    index: RaptorIndex,
    children_map: dict[str, list[str]],
) -> dict:
    """Recursively build a nested dict for d3.hierarchy."""
    node = index.nodes[node_id]
    result = {
        "id": node.id,
        "name": node.text[:120] + ("..." if len(node.text) > 120 else ""),
        "level": node.level,
        "type": node.type,
        "token_count": node.token_count,
    }
    kids = children_map.get(node_id, [])
    if kids:
        result["children"] = [
            _build_nested(cid, index, children_map) for cid in kids
        ]
    return result


def export_tree_json(index: RaptorIndex, path: str | Path) -> Path:
    """Export a strict tree (nested JSON) for ``d3.hierarchy``.

    Each child is assigned to exactly one parent (highest GMM probability).
    Orphan leaves that have no parent are grouped under a synthetic root.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Build parent -> [children] map using max-weight assignment
    child_to_parent = _build_tree_edges(index)
    parent_to_children: dict[str, list[str]] = {}
    for child, parent in child_to_parent.items():
        parent_to_children.setdefault(parent, []).append(child)

    # Find root nodes (nodes that are not children of anyone in the tree)
    all_children = set(child_to_parent.keys())
    all_parents = set(child_to_parent.values())
    roots = all_parents - all_children

    # Also find orphan leaves (no parent at all)
    all_node_ids = set(index.nodes.keys())
    orphans = all_node_ids - all_children - all_parents

    if not roots and not orphans:
        # Edge case: single node
        roots = set(index.nodes.keys())

    # If multiple roots, wrap in a synthetic root
    root_list = sorted(roots | orphans)

    if len(root_list) == 1:
        tree = _build_nested(root_list[0], index, parent_to_children)
    else:
        tree = {
            "id": "raptor_root",
            "name": "RAPTOR Index Root",
            "level": index.max_level + 1,
            "type": "root",
            "token_count": 0,
            "children": [
                _build_nested(rid, index, parent_to_children) for rid in root_list
            ],
        }

    out.write_text(json.dumps(tree, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Exported strict tree to %s", out)
    return out


# ---------------------------------------------------------------------------
# Strategy 2 — DAG / Graph JSON (for force-directed / layered DAG)
# ---------------------------------------------------------------------------

def export_dag_json(
    index: RaptorIndex,
    path: str | Path,
    min_weight: float = 0.0,
) -> Path:
    """Export the full DAG with soft-clustering edges.

    Output: ``{ nodes: [...], links: [...] }`` suitable for D3 force layout.

    Parameters
    ----------
    min_weight : float
        Exclude edges below this weight threshold.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    nodes_data = []
    for node in index.all_nodes():
        nodes_data.append({
            "id": node.id,
            "level": node.level,
            "type": node.type,
            "name": node.text[:120] + ("..." if len(node.text) > 120 else ""),
            "token_count": node.token_count,
        })

    links_data = []
    for edge in index.edges:
        if edge.weight >= min_weight:
            links_data.append({
                "source": edge.source,
                "target": edge.target,
                "weight": round(edge.weight, 4),
            })

    data = {"nodes": nodes_data, "links": links_data}
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "Exported DAG (%d nodes, %d links) to %s",
        len(nodes_data), len(links_data), out,
    )
    return out


# ---------------------------------------------------------------------------
# Retrieval overlay JSON (for highlighting query results in D3)
# ---------------------------------------------------------------------------

def export_retrieval_overlay(
    query: str,
    retrieved_ids: list[str],
    scores: list[float],
    path: str | Path,
) -> Path:
    """Export a retrieval result overlay for D3 visualization.

    D3 can use this to highlight retrieved nodes and dim the rest.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "query": query,
        "retrieved_node_ids": retrieved_ids,
        "scores": [round(s, 4) for s in scores],
    }
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# RAPTOR → Hypergraph conversion (for 1-1 visual comparison)
# ---------------------------------------------------------------------------

def raptor_to_hypergraph(index: RaptorIndex) -> "HypergraphBuilder":
    """Convert a RaptorIndex into a Hypergraph for visual comparison.

    The mapping exploits the core parallel between RAPTOR and hypergraphs:
    **each soft cluster IS a hyperedge**.  Because GMM soft-clustering allows
    a node to belong to multiple clusters, the same RAPTOR node can appear
    in multiple hyperedges — exactly the defining property of a hypergraph.

    Mapping
    -------
    - Every RAPTOR node (leaf or summary) → ``HyperNode``
    - Every cluster (identified by its parent/summary node) → ``HyperEdge``
        * ``source`` = child node labels (the clustered chunks/summaries)
        * ``target`` = [parent/summary node label]
        * ``label``  = short snippet of the summary text
    """
    from .hypergraph_store import HypergraphBuilder

    builder = HypergraphBuilder(source_document="raptor_tree")

    def _node_label(node: RaptorNode) -> str:
        """Short, unique label for visualization."""
        snippet = node.text[:55].strip().replace("\n", " ")
        return snippet + ("..." if len(node.text) > 55 else "")

    # Group edges by parent → each parent = one cluster = one hyperedge
    parent_children: dict[str, list[tuple[str, float]]] = {}
    for edge in index.edges:
        parent_children.setdefault(edge.source, []).append(
            (edge.target, edge.weight)
        )

    for parent_id, children in parent_children.items():
        parent_node = index.nodes.get(parent_id)
        if parent_node is None:
            continue

        child_labels: list[str] = []
        for child_id, _weight in children:
            child_node = index.nodes.get(child_id)
            if child_node is not None:
                child_labels.append(_node_label(child_node))

        if not child_labels:
            continue

        parent_lbl = _node_label(parent_node)
        summary_snippet = parent_node.text[:60].strip().replace("\n", " ")

        builder.add_event(
            relation=summary_snippet,
            source=child_labels,
            target=[parent_lbl],
            chunk_id=parent_id,
        )

    # Also ensure orphan leaf nodes (not in any cluster) are present
    referenced = set()
    for edge in index.edges:
        referenced.add(edge.source)
        referenced.add(edge.target)
    for nid, node in index.nodes.items():
        if nid not in referenced:
            builder._get_or_create_node(_node_label(node))

    return builder


# ---------------------------------------------------------------------------
# Convenience: export everything at once
# ---------------------------------------------------------------------------

def export_all(
    index: RaptorIndex,
    output_dir: str | Path,
    *,
    min_dag_weight: float = 0.0,
) -> dict[str, Path]:
    """Export tree JSON, DAG JSON, hypergraph JSON, nodes JSON, and embeddings NPZ.

    Returns a dict of ``{name: path}`` for all exported files.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Convert RAPTOR clusters → hypergraph format
    hg_builder = raptor_to_hypergraph(index)
    hg_path = hg_builder.save(out_dir / "raptor_as_hypergraph.json")

    paths = {
        "nodes": export_nodes_json(index, out_dir / "raptor_nodes.json"),
        "tree": export_tree_json(index, out_dir / "raptor_tree.json"),
        "dag": export_dag_json(index, out_dir / "raptor_dag.json", min_weight=min_dag_weight),
        "hypergraph": hg_path,
        "embeddings": save_embeddings_npz(index, out_dir / "raptor_embeddings.npz"),
    }

    logger.info("All exports written to %s", out_dir)
    return paths
