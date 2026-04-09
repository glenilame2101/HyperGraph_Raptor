"""CLI entry point for building a RAPTOR hierarchical index.

Usage
-----
    python -m scripts.run_raptor_build --input doc.md --output-dir raptor_out/

Or from the repo root:
    python scripts/run_raptor_build.py --input doc.md
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Ensure repo root is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from GraphReasoning.llm_client import create_llm, create_embed_client
from GraphReasoning.raptor_tree import build_raptor_index
from GraphReasoning.raptor_export import export_all, raptor_to_hypergraph
from GraphReasoning.raptor_retrieval import build_faiss_index, query_raptor
from GraphReasoning.raptor_viz import visualize_raptor
from GraphReasoning.hypergraph_viz import visualize_hypergraph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM summarizer wrapper
# ---------------------------------------------------------------------------

def make_llm_call(**overrides) -> callable:
    """Return a ``llm_call(prompt) -> str`` function using the shared LLM."""
    llm = create_llm(**overrides)

    def call(prompt: str) -> str:
        response = llm.invoke(prompt)
        return response.content

    return call


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a RAPTOR hierarchical RAG index from a text/markdown file."
    )
    # Input
    p.add_argument("--input", "-i", required=True, help="Path to input text/markdown file")
    p.add_argument("--doc-id", default="", help="Document identifier for metadata")

    # Output
    p.add_argument("--output-dir", "-o", default="raptor_output", help="Output directory")

    # Chunking
    p.add_argument("--chunk-size", type=int, default=100, help="Target chunk size in tokens (paper: 100)")
    p.add_argument("--chunk-overlap", type=int, default=0, help="Overlap in tokens (paper: 0)")

    # Tree building
    p.add_argument("--max-depth", type=int, default=5, help="Max summarization levels")
    p.add_argument("--min-cluster", type=int, default=3, help="Min nodes to attempt clustering")
    p.add_argument("--max-k", type=int, default=20, help="Max GMM clusters per level")
    p.add_argument("--membership-threshold", type=float, default=0.1, help="Soft clustering threshold")
    p.add_argument("--max-context-tokens", type=int, default=4096, help="Max tokens per summarization call")

    # Override .env (optional)
    p.add_argument("--embed-url", default=None, help="Embedding server URL (overrides EMBED_URL env var)")
    p.add_argument("--embed-model", default=None, help="Embedding model name (overrides EMBED_MODEL env var)")
    p.add_argument("--llm-url", default=None, help="LLM server URL (overrides URL env var)")
    p.add_argument("--llm-model", default=None, help="LLM model name (overrides MODEL_NAME env var)")
    p.add_argument("--llm-temperature", type=float, default=None, help="LLM temperature (overrides LLM_TEMPERATURE env var)")

    # Query (optional demo)
    p.add_argument("--query", default=None, help="Optional query to run after building")

    return p.parse_args()


def main():
    args = parse_args()

    # Read input
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    text = input_path.read_text(encoding="utf-8")
    logger.info("Read %d characters from %s", len(text), input_path)

    doc_id = args.doc_id or input_path.stem

    # Output goes into a subfolder named after the input file
    output_dir = Path(args.output_dir).resolve() / input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize clients (from .env, with optional CLI overrides)
    embed_client = create_embed_client(base_url=args.embed_url, model=args.embed_model)

    llm_overrides = {}
    if args.llm_url:
        llm_overrides["base_url"] = args.llm_url
    if args.llm_model:
        llm_overrides["model"] = args.llm_model
    if args.llm_temperature is not None:
        llm_overrides["temperature"] = args.llm_temperature
    llm_call = make_llm_call(**llm_overrides)

    # Build RAPTOR index
    logger.info("Building RAPTOR index...")
    index = build_raptor_index(
        text=text,
        embed_client=embed_client,
        llm_call=llm_call,
        doc_id=doc_id,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_depth=args.max_depth,
        min_cluster_input=args.min_cluster,
        max_k=args.max_k,
        membership_threshold=args.membership_threshold,
        max_context_tokens=args.max_context_tokens,
    )

    logger.info(
        "Index built: %d nodes, %d edges, %d levels",
        index.node_count, index.edge_count, index.max_level,
    )

    # Export everything
    paths = export_all(index, output_dir)
    for name, p in paths.items():
        logger.info("  %s -> %s", name, p)

    # Optional: demo query
    overlay = None
    if args.query:
        logger.info("Running demo query: %s", args.query)
        try:
            faiss_idx = build_faiss_index(index)
        except ImportError:
            logger.warning("FAISS not installed — using brute-force search")
            faiss_idx = None

        results = query_raptor(
            args.query, index, embed_client,
            method="collapsed",
            max_tokens=args.max_context_tokens,
            faiss_index=faiss_idx,
        )

        logger.info("Retrieved %d nodes:", len(results))
        for node, score in results:
            logger.info("  [%.4f] %s: %s", score, node.id, node.text[:80])

        overlay = {
            "retrieved_node_ids": [n.id for n, _ in results],
            "scores": [s for _, s in results],
        }

    # Visualization — original RAPTOR tree/DAG view
    viz_path = visualize_raptor(
        index,
        output_dir / "raptor_viz.html",
        retrieval_overlay=overlay,
    )
    logger.info("RAPTOR tree viz -> %s", viz_path)

    # Visualization — hypergraph view (for 1-1 comparison with hypergraph pipeline)
    hg_builder = raptor_to_hypergraph(index)
    hg_viz_path = visualize_hypergraph(
        hg_builder,
        output_dir / "raptor_as_hypergraph_viz.html",
    )
    logger.info("RAPTOR-as-hypergraph viz -> %s", hg_viz_path)
    logger.info("Done!")


if __name__ == "__main__":
    main()
