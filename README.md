# HyperGraph Raptor

An LLM-powered knowledge graph and hypergraph extraction system that builds hierarchical reasoning structures from unstructured text using large language models. Combines hypergraph modeling with RAPTOR (Retrieval-Augmented Pretrained Transformer Organization) for multi-level document understanding, semantic search, and interactive visualization.

Designed by Markus J. Buehler and Isabella Stewart at MIT.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Web Interface (Streamlit)](#web-interface-streamlit)
  - [CLI Scripts](#cli-scripts)
- [Module Reference](#module-reference)
- [Sample Data](#sample-data)
- [Troubleshooting](#troubleshooting)

---

## Overview

HyperGraph Raptor extracts n-ary relationships (hyperedges) from text documents using LLMs, building knowledge structures where a single relationship can connect multiple entities simultaneously. Unlike traditional knowledge graphs limited to binary (subject-predicate-object) triples, hypergraphs capture the full complexity of real-world relationships.

**Typical workflow:**

```
Documents (MD/PDF)
      |
[1] Text Splitting (chunking)
      |
[2] Embedding Generation (BGE-M3)
      |
[3a] RAPTOR Path: Clustering --> Summarization --> Hierarchical Tree
[3b] Hypergraph Path: LLM Extraction --> Structured JSON --> HypergraphBuilder
      |
[4] Node Deduplication & Cross-Document Merging
      |
[5] Storage (JSON + pickle)
      |
[6] Visualization (PyVis, Plotly, HTML)
      |
[7] Retrieval (FAISS index, similarity search)
```

---

## Key Features

### Hypergraph Extraction
- Extracts n-ary relationships from text where multiple entities connect through a single relation
- LLM-driven structured output: `{source: [entities], relation: string, target: [entities]}`
- Handles tables, role assignments, causal chains, and narrative plot structures
- Label-based node deduplication and entity consistency across documents

### RAPTOR Hierarchical Indexing
- Bottom-up tree construction: chunking, embedding, GMM clustering (with BIC model selection), and LLM summarization
- Soft clusters (DAG structure) where nodes can have multiple parents
- Multi-level summaries for hierarchical reasoning over documents
- Supports both tree and DAG serialization formats

### Embeddings and Semantic Search
- Node-level embeddings via BGE-M3 (multilingual model)
- Embedding caching with incremental updates
- UMAP/PCA projection to 2D/3D for visualization
- HDBSCAN/KMeans clustering for exploration
- FAISS-backed similarity search

### Interactive Web Interface (Streamlit)
- **Documents Tab:** Upload and scan markdown files
- **Pipeline Tab:** Configure LLM/embedding settings, run hypergraph extraction
- **Graph Explorer Tab:** Interactive ego-graph visualization (PyVis), hyperedge inspection
- **Embeddings Tab:** 2D/3D projection plots, cluster inspection, similarity search
- **Overview Tab:** Global statistics and merged graph information

### Graph Analysis
- Community detection (Louvain method)
- Shortest path finding with embedding-guided heuristics
- Ego-graph extraction (N-hop neighborhoods)
- Power-law analysis for scale-free network properties

### Visualization
- PyVis bipartite graphs (entities + hyperedge connectors)
- Plotly interactive 3D/2D scatter plots
- D3.js-compatible JSON exports
- Standalone HTML reports with embedded graphs

---

## Architecture

### Key Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `HypergraphBuilder` | `GraphReasoning/graph_generation.py` | Incremental hypergraph construction with label-based deduplication |
| `RaptorIndex` | `GraphReasoning/raptor_tree.py` | Hierarchical index data structure for RAPTOR |
| `RaptorNode` / `RaptorEdge` | `GraphReasoning/raptor_tree.py` | Node and edge models for the RAPTOR tree |
| `LocalBGEClient` | `GraphReasoning/llm_client.py` | Embedding client with retry logic and token-aware truncation |
| `Hypergraph` / `HyperNode` / `HyperEdge` | `GraphReasoning/graph_generation.py` | Pydantic models for JSON schema validation |

### LLM Integration
- Uses LangChain's `ChatOpenAI` wrapper for any OpenAI-compatible endpoint
- Structured output via Pydantic models and `trustcall` for reliable extraction
- All prompts are configurable via `prompt_config.json`

---

## Project Structure

```
HyperGraph_Raptor/
|-- app.py                          # Streamlit web interface (main entry point)
|-- run_make_new_hypergraph.py      # CLI wrapper for hypergraph generation
|-- pdf_to_markdown.py              # PDF conversion utility
|-- pyproject.toml                  # Python project metadata
|-- requirements.txt                # Python dependencies
|-- uv.lock                         # Locked dependency tree (UV)
|-- prompt_config.json              # LLM prompt templates and configuration
|-- .env                            # Environment variables (API keys, URLs)
|
|-- GraphReasoning/                 # Core library package
|   |-- __init__.py                 # Public API exports
|   |-- llm_client.py              # LLM and embedding client factory
|   |-- prompt_config.py           # Prompt template loader and manager
|   |-- graph_generation.py        # Hypergraph extraction from text
|   |-- graph_tools.py             # Core graph processing utilities
|   |-- graph_analysis.py          # Community detection, path finding
|   |-- hypergraph_store.py        # JSON-based hypergraph persistence
|   |-- hypergraph_viz.py          # Hypergraph visualization
|   |-- raptor_tree.py             # RAPTOR hierarchical index builder
|   |-- raptor_export.py           # RAPTOR serialization/export
|   |-- raptor_retrieval.py        # RAPTOR query and retrieval
|   |-- raptor_viz.py              # RAPTOR tree visualization
|   |-- utils.py                   # Utility functions
|
|-- scripts/                       # Standalone CLI scripts
|   |-- run_make_new_hypergraph.py  # Hypergraph generation (multi-doc)
|   |-- run_raptor_build.py        # RAPTOR index builder
|   |-- run_hypergraph_to_viz.py   # Hypergraph to HTML visualization
|   |-- pdf2markdown.py            # PDF extraction utility
|
|-- Data/                          # Sample input documents
|   |-- Automate the Boring Stuff with Python.md
|   |-- Cars_Movie.md
|   |-- Finding_Nemo.md
|
|-- artifacts/                     # Generated outputs
|   |-- sg/
|       |-- graphs/                # Individual hypergraph JSONs
|       |-- integrated/            # Merged/integrated hypergraphs
|       |-- html/                  # HTML visualizations
|       |-- cache/chunks/          # Text chunk and embedding cache
|
|-- raptor_output/                 # RAPTOR-specific outputs
```

---

## Prerequisites

- **Python 3.10+**
- **UV** (recommended) or pip for package management
- **OpenAI-compatible LLM endpoint** (OpenAI API, local vLLM/Ollama server, etc.)
- **BGE Embeddings server** running locally on port 8080, or any compatible embedding endpoint
  - Model: `BAAI/bge-m3` (multilingual, recommended)

---

## Installation

### Using UV (recommended)

```bash
git clone <repository-url>
cd HyperGraph_Raptor

# Create virtual environment and install dependencies
uv venv
uv pip install -r requirements.txt
uv pip install -e .
```

### Using pip

```bash
git clone <repository-url>
cd HyperGraph_Raptor

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

---

## Configuration

### Environment Variables

Create a `.env` file in the project root (or edit the existing one):

```env
# ── LLM Configuration ──────────────────────────────────
URL=https://api.openai.com/v1          # OpenAI-compatible API base URL
MODEL_NAME=gpt-4o-2025-03-05          # Model identifier
OPENAI_API_KEY=sk-...                  # API key
LLM_TEMPERATURE=0                     # Sampling temperature (0 = deterministic)
LLM_MAX_TOKENS=20000                  # Max generation length
LLM_TIMEOUT=120.0                     # HTTP timeout in seconds

# ── Embeddings ──────────────────────────────────────────
EMBED_URL=http://127.0.0.1:8080       # Embedding server URL
EMBED_MODEL=BAAI/bge-m3               # Embedding model name
EMBED_MAX_CHARS=8192                  # Max input size for embeddings

# ── Optional Overrides ──────────────────────────────────
GRAPH_REASONING_CACHE_DIR=./cache     # Cache directory for chunks
GRAPH_REASONING_PROMPT_CONFIG=./prompt_config.json  # Custom prompt config path
```

### Prompt Configuration

All LLM prompts are defined in `prompt_config.json` and organized by domain:

| Domain | Purpose |
|--------|---------|
| `graph` | Binary knowledge graph extraction |
| `hypergraph` | N-ary hypergraph extraction (primary) |
| `graph_tools` | Node renaming, community summarization, keyword extraction |
| `raptor` | Text summarization for hierarchical levels |
| `runtime` | Figure description, fallback prompts |

You can customize prompts by editing `prompt_config.json` or providing a custom config via the `GRAPH_REASONING_PROMPT_CONFIG` environment variable.

---

## Usage

### Web Interface (Streamlit)

Launch the interactive dashboard:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. The interface provides:

1. **Documents** - Select and manage input documents from the `Data/` directory
2. **Pipeline** - Configure chunk size, LLM parameters, and run extraction
3. **Graph Explorer** - Browse the generated hypergraph interactively
4. **Embeddings** - Explore 2D/3D projections and run similarity searches
5. **Overview** - View aggregate statistics across all processed documents

### CLI Scripts

#### Generate Hypergraphs

Process markdown documents into hypergraphs:

```bash
python scripts/run_make_new_hypergraph.py \
  --doc-data-dir Data \
  --artifacts-root artifacts/sg \
  --chunk-size 10000
```

**Arguments:**
- `--doc-data-dir` - Directory containing `.md` input files
- `--artifacts-root` - Output directory for graphs, caches, and HTML
- `--chunk-size` - Text chunk size in characters (default: 10000)

#### Build RAPTOR Index

Build a hierarchical RAPTOR index from a document:

```bash
python scripts/run_raptor_build.py \
  --input Data/Cars_Movie.md \
  --output-dir raptor_output/Cars_Movie
```

**Arguments:**
- `--input` - Path to the input markdown file
- `--output-dir` - Directory for RAPTOR output (JSON, embeddings, visualizations)

#### Generate Visualizations

Create HTML visualizations from existing hypergraph JSONs:

```bash
python scripts/run_hypergraph_to_viz.py \
  --doc-data-dir Data \
  --json-out-dir artifacts/sg/graphs \
  --html-out-dir artifacts/sg/html
```

#### Convert PDF to Markdown

Pre-process PDF documents into markdown for ingestion:

```bash
python scripts/pdf2markdown.py \
  --input Data/document.pdf \
  --output Data/document.md
```

---

## Module Reference

### `GraphReasoning` Package

| Module | Description |
|--------|-------------|
| `llm_client` | Factory for LLM chat clients and embedding clients. Configures `ChatOpenAI` and `LocalBGEClient` from environment variables. |
| `prompt_config` | Loads and manages prompt templates from `prompt_config.json`. Supports domain-scoped template lookup. |
| `graph_generation` | Core hypergraph extraction logic. Chunks text, calls LLM for structured extraction, builds `HypergraphBuilder` objects. |
| `graph_tools` | Graph processing utilities: node merging, ego-graph extraction, network statistics, embedding projection, clustering. |
| `graph_analysis` | Higher-level analysis: community detection (Louvain), shortest paths with embedding heuristics, centrality metrics. |
| `hypergraph_store` | JSON-based persistence layer for hypergraphs. Handles serialization/deserialization of nodes, edges, and metadata. |
| `hypergraph_viz` | Visualization generators for hypergraphs. Produces PyVis HTML graphs and Plotly interactive plots. |
| `raptor_tree` | RAPTOR index builder. Implements bottom-up clustering (GMM + BIC), LLM summarization, and tree/DAG construction. |
| `raptor_export` | Serialization of RAPTOR trees to JSON. Exports node hierarchies, embeddings, and metadata. |
| `raptor_retrieval` | Query interface for RAPTOR indices. Supports tree traversal and FAISS-backed similarity retrieval. |
| `raptor_viz` | RAPTOR tree visualization. Generates hierarchical tree plots and level-by-level summaries. |
| `utils` | Shared utility functions (text cleaning, token counting, etc.). |

---

## Sample Data

The `Data/` directory includes sample documents for testing:

| File | Description |
|------|-------------|
| `Automate the Boring Stuff with Python.md` | Technical book content (programming) |
| `Cars_Movie.md` | Narrative content (movie plot) |
| `Finding_Nemo.md` | Narrative content (movie plot) |

These demonstrate the system's ability to handle both technical documentation and narrative text.

---

## Troubleshooting

### Common Issues

**Embedding server not reachable**
```
ConnectionError: Cannot connect to http://127.0.0.1:8080
```
Ensure your BGE embedding server is running. You can start one with:
```bash
# Example using a compatible embedding server
python -m sentence_transformers.server --model BAAI/bge-m3 --port 8080
```

**LLM timeout errors**
Increase the timeout in `.env`:
```env
LLM_TIMEOUT=300.0
```

**Out of memory during embedding generation**
Reduce `EMBED_MAX_CHARS` or process fewer documents at a time.

**Missing API key**
Ensure `OPENAI_API_KEY` is set in your `.env` file or exported as an environment variable.

**PDF conversion issues**
Install system dependencies for PDF processing:
```bash
# Ubuntu/Debian
sudo apt-get install pdftohtml tesseract-ocr

# macOS
brew install pdftohtml tesseract

# Windows
# Install from: https://github.com/oschwartz10612/poppler-windows/releases
```

---

## License

See repository for license information.
