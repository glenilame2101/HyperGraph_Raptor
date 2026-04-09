"""
HyperGraph Streamlit App
========================
Interactive GUI for processing documents into hypergraphs,
generating embeddings, and visualizing results.

Run with:  streamlit run app.py
"""

import base64
import glob
import os
import pickle
import random
import re
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import List

import hypernetx as hnx
import networkx as nx
import numpy as np
import streamlit as st
import torch
from dotenv import load_dotenv
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so GraphReasoning is importable
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from GraphReasoning.graph_generation import (
    add_new_hypersubgraph_from_text,
    make_hypergraph_from_text,
)
from GraphReasoning.graph_tools import (
    generate_hypernode_embeddings,
    load_embeddings,
    save_embeddings,
    update_hypernode_embeddings,
)
from GraphReasoning.prompt_config import get_prompt
from GraphReasoning.llm_client import create_llm, create_embed_client, LocalBGEClient

# ---------------------------------------------------------------------------
# Pydantic models for structured LLM output (same as scripts/)
# ---------------------------------------------------------------------------

class Event(BaseModel):
    source: List[str]
    target: List[str]
    relation: str


class HypergraphJSON(BaseModel):
    events: List[Event]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def image_to_base64_data_uri(file_path: str) -> str:
    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def resolve_docs(doc_dir: str) -> list[str]:
    """Find markdown files in a directory (flat or nested)."""
    root = Path(doc_dir)
    if not root.exists():
        return []
    docs = sorted(str(p) for p in root.glob("*.md"))
    if docs:
        return docs
    for folder in sorted(root.iterdir()):
        if folder.is_dir():
            candidate = folder / f"{folder.name}.md"
            if candidate.exists():
                docs.append(str(candidate))
    return sorted(docs)


def _load_hypergraph_from_disk(integrated_dir: Path):
    """Load the most recent integrated hypergraph pickle from disk."""
    pkl_files = sorted(
        glob.glob(str(integrated_dir / "*_integrated.pkl")),
        key=os.path.getmtime,
    )
    if not pkl_files:
        return None
    try:
        with open(pkl_files[-1], "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _ensure_hypergraph():
    """Return hypergraph from session or disk; cache in session_state."""
    if st.session_state.hypergraph is not None:
        return st.session_state.hypergraph
    G = _load_hypergraph_from_disk(integrated_path)
    if G is not None:
        st.session_state.hypergraph = G
    return G


def _ensure_embeddings():
    """Return embeddings from session or disk; cache in session_state."""
    if st.session_state.node_embeddings:
        return st.session_state.node_embeddings
    emb_disk = graphs_path / embedding_file
    if emb_disk.exists():
        try:
            embs = load_embeddings(str(emb_disk))
            st.session_state.node_embeddings = embs
            return embs
        except Exception:
            pass
    return None


def build_ego_hypergraph(H: hnx.Hypergraph, seed_node: str, hops: int = 1):
    """
    Collect hyperedges within N hops of seed_node.
    Returns (sub_incidence_dict, triples) where triples is a list of
    dicts with keys: edge_id, members (set of node names).
    """
    visited_nodes = {seed_node}
    visited_edges = set()

    for _ in range(hops):
        frontier_edges = set()
        for eid, members in H.incidence_dict.items():
            if eid in visited_edges:
                continue
            if visited_nodes & set(members):
                frontier_edges.add(eid)
        if not frontier_edges:
            break
        visited_edges |= frontier_edges
        for eid in frontier_edges:
            visited_nodes |= set(H.incidence_dict[eid])

    sub_incidence = {eid: H.incidence_dict[eid] for eid in visited_edges}
    return sub_incidence, visited_nodes


def render_ego_pyvis(sub_incidence: dict, seed_node: str, height: str = "650px") -> str:
    """
    Render ego subgraph as a bipartite PyVis graph.
    Entity nodes are circles; hyperedge connector nodes are small diamonds.
    """
    from pyvis.network import Network

    net = Network(height=height, width="100%", bgcolor="#0e1117", font_color="#fafafa",
                  directed=False, cdn_resources="remote")

    # Collect node degrees for sizing
    node_degrees: dict[str, int] = {}
    for members in sub_incidence.values():
        for m in members:
            node_degrees[m] = node_degrees.get(m, 0) + 1

    max_deg = max(node_degrees.values()) if node_degrees else 1

    # Add entity nodes
    for node, deg in node_degrees.items():
        size = 10 + 25 * (deg / max_deg)
        color = "#f59e0b" if node == seed_node else "#3b82f6"
        border = "#ffffff" if node == seed_node else "#1e3a5f"
        net.add_node(
            node, label=str(node)[:40], size=size, color=color,
            borderWidth=3 if node == seed_node else 1,
            borderWidthSelected=4, font={"size": 11},
            title=f"{node}\ndegree: {deg}",
            shape="dot",
        )

    # Add hyperedge connector nodes and link to members
    for eid, members in sub_incidence.items():
        he_id = f"__he__{eid}"
        label = str(eid)[:30]
        net.add_node(
            he_id, label=label, size=6, color="#94a3b8",
            shape="diamond", font={"size": 9, "color": "#94a3b8"},
            title=f"Hyperedge: {eid}\nMembers: {len(members)}",
        )
        for member in members:
            net.add_edge(str(member), he_id, color="#475569", width=1)

    net.repulsion(node_distance=160, spring_length=200, spring_strength=0.04)
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 150}
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "zoomView": true
      }
    }
    """)

    tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w")
    net.save_graph(tmp.name)
    tmp.close()
    with open(tmp.name, "r", encoding="utf-8") as f:
        html = f.read()
    os.unlink(tmp.name)
    return html


def compute_embedding_projection(embs: dict, method: str = "umap", n_components: int = 2):
    """
    Project embeddings to 2D or 3D and cluster them.
    Returns (points, cluster_labels, labels, actual_method).
    Cached via st.session_state to avoid recomputation.
    """
    cache_key = f"_emb_proj_{method}_{n_components}d_{len(embs)}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    labels = list(embs.keys())
    X = np.vstack([np.asarray(embs[k], dtype=float) for k in labels])

    # Projection
    if method == "umap":
        try:
            from umap import UMAP
            reducer = UMAP(n_components=n_components, n_neighbors=min(15, len(X) - 1),
                           min_dist=0.1, metric="cosine", random_state=42)
            pts = reducer.fit_transform(X)
        except ImportError:
            from sklearn.decomposition import PCA
            pts = PCA(n_components=n_components, random_state=42).fit_transform(X)
            method = "pca_fallback"
    else:
        from sklearn.decomposition import PCA
        pts = PCA(n_components=n_components, random_state=42).fit_transform(X)

    # Clustering
    cluster_labels = np.zeros(len(X), dtype=int)
    if len(X) >= 5:
        try:
            from hdbscan import HDBSCAN
            clusterer = HDBSCAN(min_cluster_size=max(3, len(X) // 50),
                                min_samples=2, metric="euclidean")
            cluster_labels = clusterer.fit_predict(pts)
        except ImportError:
            from sklearn.cluster import KMeans
            n_clusters = min(8, max(2, len(X) // 20))
            cluster_labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X)

    result = (pts, cluster_labels, labels, method)
    st.session_state[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="HyperGraph", page_icon="", layout="wide")
st.title("HyperGraph Pipeline")

# Load .env if present
load_dotenv(dotenv_path=_REPO_ROOT / ".env")

# ---------------------------------------------------------------------------
# Sidebar: configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("LLM Configuration")
    llm_url = st.text_input("LLM Base URL", value=os.getenv("URL", "http://127.0.0.1:8080/v1"))
    llm_api_key = st.text_input("API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    llm_model = st.text_input("Model Name", value=os.getenv("MODEL_NAME", ""))
    llm_max_tokens = st.number_input("Max Tokens", value=20000, min_value=1000, step=1000)

    st.header("Embedding Server")
    bge_url = st.text_input("Embedding URL", value="http://127.0.0.1:8080")
    bge_model = st.text_input("Embedding Model", value="BAAI/bge-m3")

    st.header("Pipeline Settings")
    chunk_size = st.number_input("Chunk Size", value=10000, min_value=500, step=500)
    chunk_overlap = st.number_input("Chunk Overlap", value=0, min_value=0, step=100)
    similarity_threshold = st.slider("Similarity Threshold", 0.5, 1.0, 0.9, 0.05)
    merge_every = st.number_input("Simplify Every N Docs", value=100, min_value=1)

    st.header("Paths")
    artifacts_root = st.text_input("Artifacts Root", value="artifacts/sg")

# Derived paths
artifacts_path = (_REPO_ROOT / artifacts_root).resolve()
graphs_path = artifacts_path / "graphs"
integrated_path = artifacts_path / "integrated"
cache_path = (_REPO_ROOT / "artifacts" / "cache" / "chunks").resolve()
embedding_file = "hypergraph_embeedings.pkl"

for p in [graphs_path, integrated_path, cache_path]:
    p.mkdir(parents=True, exist_ok=True)
os.environ["GRAPH_REASONING_CACHE_DIR"] = str(cache_path)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "pipeline_log" not in st.session_state:
    st.session_state.pipeline_log = []
if "hypergraph" not in st.session_state:
    st.session_state.hypergraph = None
if "node_embeddings" not in st.session_state:
    st.session_state.node_embeddings = None
if "doc_list" not in st.session_state:
    st.session_state.doc_list = []

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_docs, tab_pipeline, tab_explorer, tab_embeddings, tab_overview = st.tabs(
    ["Documents", "Pipeline", "Graph Explorer", "Embeddings", "Overview"]
)

# ========================== TAB 1: DOCUMENTS ==============================

with tab_docs:
    st.subheader("Load Documents")

    col_upload, col_dir = st.columns(2)

    with col_upload:
        st.markdown("**Upload Markdown Files**")
        uploaded_files = st.file_uploader(
            "Drop .md files here",
            type=["md"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            upload_dir = _REPO_ROOT / "Data"
            upload_dir.mkdir(parents=True, exist_ok=True)
            for uf in uploaded_files:
                dest = upload_dir / uf.name
                dest.write_bytes(uf.getvalue())
            st.success(f"Saved {len(uploaded_files)} file(s) to Data/")

    with col_dir:
        st.markdown("**Or Load from Directory**")
        doc_dir = st.text_input("Document Directory", value="Data")
        if st.button("Scan Directory"):
            full_dir = str((_REPO_ROOT / doc_dir).resolve())
            st.session_state.doc_list = resolve_docs(full_dir)

    # Always re-scan if we have uploads
    if uploaded_files and not st.session_state.doc_list:
        st.session_state.doc_list = resolve_docs(str(_REPO_ROOT / "Data"))

    if st.session_state.doc_list:
        st.markdown(f"**{len(st.session_state.doc_list)} document(s) found:**")
        for doc in st.session_state.doc_list:
            st.text(f"  {Path(doc).name}")
    else:
        st.info("No documents loaded yet. Upload files or scan a directory.")

# ========================== TAB 2: PIPELINE ===============================

with tab_pipeline:
    st.subheader("Run Pipeline")

    if not st.session_state.doc_list:
        st.warning("Load documents in the Documents tab first.")
    elif not llm_url or not llm_model or not llm_api_key:
        st.warning("Configure LLM URL, Model Name, and API Key in the sidebar.")
    else:
        st.markdown(f"Ready to process **{len(st.session_state.doc_list)}** document(s).")

        if st.button("Process Documents", type="primary"):
            log = st.session_state.pipeline_log = []
            progress_bar = st.progress(0)
            status_area = st.empty()
            log_area = st.container()

            def log_msg(msg: str):
                log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
                with log_area:
                    st.code("\n".join(log[-30:]), language="text")

            # --- Build LLM client ---
            try:
                client = create_llm(
                    base_url=llm_url,
                    model=llm_model,
                    api_key=llm_api_key,
                    max_tokens=llm_max_tokens,
                    verify_ssl=False,
                )
                log_msg(f"LLM client ready: {llm_model} @ {llm_url}")
            except Exception as e:
                st.error(f"Failed to create LLM client: {e}")
                st.stop()

            # --- Build generate functions ---
            try:
                from trustcall import create_extractor
            except ImportError:
                st.error("Missing dependency: `uv pip install trustcall`")
                st.stop()

            default_system_prompt = get_prompt("runtime", "default_system_prompt")

            def generate(
                system_prompt=None,
                prompt="",
                temperature=0.333,
                max_tokens=None,
                response_model=HypergraphJSON,
                **_,
            ):
                effective_system = system_prompt or default_system_prompt
                messages = [
                    {"role": "system", "content": effective_system},
                    {"role": "user", "content": prompt},
                ]
                extractor = create_extractor(client, tools=[response_model])
                retries, delay = 6, 2.0
                last_exc = None
                for attempt in range(1, retries + 1):
                    try:
                        result = extractor.invoke({"messages": messages})
                        responses = result.get("responses", [])
                        if responses:
                            return responses[0]
                        raise ValueError("trustcall returned no responses")
                    except Exception as exc:
                        last_exc = exc
                        if attempt < retries:
                            log_msg(f"LLM retry {attempt}/{retries}: {exc!r}")
                            time.sleep(delay + random.uniform(0, 0.5))
                            delay = min(30, delay * 2)
                if last_exc:
                    raise last_exc

            def generate_figure(image, system_prompt=None, prompt="", temperature=0.0, **_):
                try:
                    image_path = Path(image)
                    if not image_path.exists():
                        image_path = next(Path(".").glob(f"**/{Path(image).name}"))
                    image_uri = image_to_base64_data_uri(str(image_path))
                    response = client.invoke([
                        {"role": "system", "content": system_prompt or get_prompt("runtime", "figure_system_prompt")},
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt or get_prompt("runtime", "figure_user_prompt")},
                            {"type": "image_url", "image_url": {"url": image_uri}},
                        ]},
                    ])
                    return getattr(response, "content", "") or ""
                except Exception:
                    return ""

            # --- Embedding client ---
            embedding_tokenizer = None
            embedding_model = LocalBGEClient(base_url=bge_url, model=bge_model)
            log_msg(f"Embedding client ready: {bge_model} @ {bge_url}")

            # --- Load or create embeddings ---
            emb_path = graphs_path / embedding_file
            if emb_path.exists():
                node_embeddings = load_embeddings(str(emb_path))
                log_msg(f"Loaded {len(node_embeddings)} existing embeddings")
            else:
                node_embeddings = generate_hypernode_embeddings([], embedding_tokenizer, embedding_model)
                save_embeddings(node_embeddings, str(emb_path))
                log_msg("Initialized empty embeddings")

            # --- Load existing integrated graph ---
            G = None
            int_prefix = re.compile(r"^(\d+)_")

            def extract_idx(path: str) -> int:
                match = int_prefix.match(os.path.basename(path))
                return int(match.group(1)) if match else -1

            merged_list = sorted(
                glob.glob(str(integrated_path / "*_integrated.pkl")),
                reverse=True,
                key=extract_idx,
            )
            current_merged_i = 0
            if merged_list:
                try:
                    with open(merged_list[0], "rb") as f:
                        G = pickle.load(f)
                    current_merged_i = extract_idx(merged_list[0])
                    log_msg(f"Loaded existing integrated graph (doc index {current_merged_i})")
                except Exception as exc:
                    log_msg(f"Could not load integrated graph: {exc!r}")

            # --- Process documents ---
            doc_list = st.session_state.doc_list
            total = len(doc_list)

            with torch.no_grad():
                for i, doc in enumerate(doc_list):
                    progress_bar.progress((i + 1) / total)
                    status_area.text(f"Processing document {i + 1}/{total}...")

                    if i < current_merged_i:
                        log_msg(f"[{i}] Skipping (already merged)")
                        continue

                    title = os.path.basename(doc).rsplit(".md", 1)[0]
                    graph_root = f"{i}_{title[:100]}"
                    graph_pkl = graphs_path / f"{graph_root}.pkl"
                    integrated_pkl = integrated_path / f"{graph_root}_integrated.pkl"

                    with open(doc, "r", encoding="utf-8") as f:
                        txt = f.read()

                    # --- Generate per-doc hypergraph ---
                    needs_gen = not graph_pkl.exists()
                    if not needs_gen:
                        try:
                            with open(graph_pkl, "rb") as f:
                                existing = pickle.load(f)
                            if existing is None or not hasattr(existing, "incidence_dict"):
                                graph_pkl.unlink(missing_ok=True)
                                needs_gen = True
                        except Exception:
                            graph_pkl.unlink(missing_ok=True)
                            needs_gen = True

                    if needs_gen:
                        log_msg(f"[{i}] Generating KG: {title}")
                        try:
                            t0 = datetime.now()
                            graph_pkl, _, _, _ = make_hypergraph_from_text(
                                txt,
                                generate,
                                generate_figure,
                                image_list="",
                                graph_root=graph_root,
                                do_distill=False,
                                do_relabel=False,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                                repeat_refine=0,
                                verbatim=False,
                                data_dir=str(graphs_path),
                            )
                            log_msg(f"[{i}] Generated in {datetime.now() - t0}")
                        except Exception as exc:
                            log_msg(f"[{i}] Generation failed: {exc!r}")
                            continue
                    else:
                        log_msg(f"[{i}] Using cached KG: {title}")

                    # --- Skip if already integrated ---
                    if integrated_pkl.exists():
                        log_msg(f"[{i}] Already integrated, skipping")
                        try:
                            with open(integrated_pkl, "rb") as f:
                                G = pickle.load(f)
                        except Exception:
                            pass
                        continue

                    # --- Load sub-graph ---
                    try:
                        with open(str(graph_pkl), "rb") as f:
                            H0 = pickle.load(f)
                        if H0 is None or not hasattr(H0, "incidence_dict"):
                            log_msg(f"[{i}] Invalid graph pickle, skipping")
                            continue
                        H_doc = hnx.Hypergraph(
                            H0.incidence_dict,
                            edge_attr={"DOI": {eid: title for eid in H0.incidence_dict}},
                        )
                    except Exception as exc:
                        log_msg(f"[{i}] Failed loading sub-graph: {exc!r}")
                        continue

                    # --- Merge ---
                    if G is None:
                        G = H_doc
                        with open(str(integrated_pkl), "wb") as f:
                            pickle.dump(G, f)
                        node_embeddings = update_hypernode_embeddings(
                            node_embeddings, G, embedding_tokenizer, embedding_model,
                        )
                        save_embeddings(node_embeddings, str(emb_path))
                        log_msg(f"[{i}] Initialized integrated graph")
                    else:
                        do_simplify = (i % merge_every == 0)
                        try:
                            _, G, _, node_embeddings, _ = add_new_hypersubgraph_from_text(
                                txt="",
                                node_embeddings=node_embeddings,
                                tokenizer=embedding_tokenizer,
                                model=embedding_model,
                                original_graph=G,
                                data_dir_output=str(integrated_path),
                                graph_root=graph_root,
                                do_simplify_graph=do_simplify,
                                do_relabel=False,
                                size_threshold=10 if do_simplify else 0,
                                do_update_node_embeddings=do_simplify,
                                repeat_refine=0,
                                similarity_threshold=similarity_threshold,
                                do_Louvain_on_new_graph=False,
                                return_only_giant_component=False,
                                save_common_graph=False,
                                G_to_add=H_doc,
                                graph_pkl_to_add=None,
                                sub_dfs=[],
                                verbatim=False,
                            )
                            save_embeddings(node_embeddings, str(emb_path))
                            log_msg(f"[{i}] Merged successfully")
                        except Exception as exc:
                            log_msg(f"[{i}] Merge failed: {exc!r}")

            # --- Done ---
            st.session_state.hypergraph = G
            st.session_state.node_embeddings = node_embeddings
            progress_bar.progress(1.0)
            status_area.empty()
            if G is not None:
                log_msg(f"Pipeline complete: {len(G.nodes)} nodes, {len(G.edges)} edges")
                st.success("Pipeline complete!")
            else:
                log_msg("Pipeline finished but no graph was produced.")
                st.warning("No graph produced. Check the log for errors.")

    # Show previous log if exists
    if st.session_state.pipeline_log and not st.session_state.get("_processing"):
        with st.expander("Previous Run Log", expanded=False):
            st.code("\n".join(st.session_state.pipeline_log[-50:]), language="text")

# ========================== TAB 3: GRAPH EXPLORER =========================

with tab_explorer:
    st.subheader("Graph Explorer")

    G_display = _ensure_hypergraph()

    if G_display is None:
        st.info("No hypergraph available. Run the pipeline or check artifacts directory.")
    else:
        node_list = sorted(str(n) for n in G_display.nodes)

        col_search, col_hops = st.columns([3, 1])
        with col_search:
            seed = st.selectbox("Select a node to explore", options=[""] + node_list,
                                index=0, placeholder="Type to search...")
        with col_hops:
            hops = st.slider("Hops", 1, 3, 1)

        if seed:
            sub_inc, sub_nodes = build_ego_hypergraph(G_display, seed, hops=hops)

            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Nodes in view", len(sub_nodes))
            col_m2.metric("Hyperedges in view", len(sub_inc))
            col_m3.metric("Hops from seed", hops)

            # Render bipartite ego graph
            if sub_inc:
                html = render_ego_pyvis(sub_inc, seed)
                st.components.v1.html(html, height=680, scrolling=True)
            else:
                st.warning(f"Node '{seed}' has no hyperedges.")

            # Hyperedge table
            st.markdown("---")
            st.markdown(f"**Hyperedges involving '{seed}'**")

            import pandas as pd

            rows = []
            for eid, members in sub_inc.items():
                members_list = sorted(str(m) for m in members)
                rows.append({
                    "Edge ID": str(eid)[:60],
                    "Members": ", ".join(members_list),
                    "Size": len(members_list),
                })
            if rows:
                df_edges = pd.DataFrame(rows)
                st.dataframe(df_edges, use_container_width=True, hide_index=True)
        else:
            st.info("Select a node above to explore its neighborhood.")

# ========================== TAB 4: EMBEDDINGS =============================

with tab_embeddings:
    st.subheader("Node Embeddings")

    embs = _ensure_embeddings()

    if not embs:
        st.info("No embeddings available. Run the pipeline first.")
    else:
        import plotly.express as px
        import pandas as pd

        st.metric("Embedded Nodes", len(embs))

        # Projection method selector
        col_proj, col_dim = st.columns(2)
        with col_proj:
            proj_method = st.radio("Projection method", ["UMAP", "PCA"], horizontal=True)
        with col_dim:
            dim_choice = st.radio("Dimensions", ["2D", "3D"], horizontal=True)
        method_key = proj_method.lower()
        n_components = 3 if dim_choice == "3D" else 2

        pts, cluster_labels, labels, actual_method = compute_embedding_projection(
            embs, method=method_key, n_components=n_components
        )

        if actual_method == "pca_fallback":
            st.caption("UMAP not installed (`uv pip install umap-learn`). Falling back to PCA.")

        # Build dataframe for plotting
        cluster_strs = [f"Cluster {c}" if c >= 0 else "Noise" for c in cluster_labels]
        axis_label = "UMAP" if "umap" in actual_method else "PC"
        method_display = actual_method.upper().replace('_FALLBACK', ' (fallback)')

        if n_components == 3:
            df_plot = pd.DataFrame({
                "x": pts[:, 0],
                "y": pts[:, 1],
                "z": pts[:, 2],
                "node": [str(l)[:60] for l in labels],
                "cluster": cluster_strs,
            })
            fig = px.scatter_3d(
                df_plot, x="x", y="y", z="z",
                color="cluster",
                hover_name="node",
                title=f"Node Embeddings ({method_display} 3D)",
                labels={"x": f"{axis_label} 1", "y": f"{axis_label} 2", "z": f"{axis_label} 3"},
                height=750,
            )
            fig.update_traces(marker=dict(size=3, opacity=0.75))
        else:
            df_plot = pd.DataFrame({
                "x": pts[:, 0],
                "y": pts[:, 1],
                "node": [str(l)[:60] for l in labels],
                "cluster": cluster_strs,
            })
            fig = px.scatter(
                df_plot, x="x", y="y",
                color="cluster",
                hover_name="node",
                title=f"Node Embeddings ({method_display} 2D)",
                labels={"x": f"{axis_label} 1", "y": f"{axis_label} 2"},
                height=650,
            )
            fig.update_traces(marker=dict(size=5, opacity=0.75))

        fig.update_layout(
            legend_title_text="Cluster",
            plot_bgcolor="#fafafa",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Cluster summary
        n_clusters = len(set(cluster_labels) - {-1})
        if n_clusters > 0:
            with st.expander(f"Cluster Summary ({n_clusters} clusters)"):
                for cid in sorted(set(cluster_labels)):
                    if cid < 0:
                        continue
                    members = [labels[i] for i in range(len(labels)) if cluster_labels[i] == cid]
                    st.markdown(f"**Cluster {cid}** ({len(members)} nodes)")
                    # Show top 5 by embedding norm (proxy for centrality)
                    norms = [(m, float(np.linalg.norm(np.asarray(embs[m], dtype=float)))) for m in members]
                    norms.sort(key=lambda x: x[1], reverse=True)
                    for name, _ in norms[:5]:
                        st.text(f"  {name}")

        # Similarity search
        st.markdown("---")
        st.markdown("**Similarity Search**")
        query = st.text_input("Enter a keyword or phrase:")
        n_results = st.slider("Number of results", 1, 20, 5)

        if query:
            try:
                emb_client = LocalBGEClient(base_url=bge_url, model=bge_model)
                query_vec = emb_client.encode(query).flatten()

                sims = {}
                for node, vec in embs.items():
                    vec_flat = np.asarray(vec, dtype=float).flatten()
                    dot = np.dot(vec_flat, query_vec)
                    norm = np.linalg.norm(vec_flat) * np.linalg.norm(query_vec)
                    sims[node] = float(dot / norm) if norm > 0 else 0.0
                top = sorted(sims.items(), key=lambda x: x[1], reverse=True)[:n_results]

                results_df = pd.DataFrame(top, columns=["Node", "Cosine Similarity"])
                st.dataframe(results_df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Search failed (is the embedding server running?): {e}")

# ========================== TAB 5: OVERVIEW ================================

with tab_overview:
    st.subheader("Graph Overview")

    G_ov = _ensure_hypergraph()

    if G_ov is None:
        st.info("No hypergraph available. Run the pipeline or check artifacts directory.")
    else:
        import plotly.express as px
        import plotly.graph_objects as go
        import pandas as pd

        # --- Metrics row ---
        node_count = len(G_ov.nodes)
        edge_count = len(G_ov.edges)
        edge_sizes = [len(G_ov.incidence_dict[e]) for e in G_ov.incidence_dict]
        avg_edge_size = sum(edge_sizes) / len(edge_sizes) if edge_sizes else 0
        degrees = {str(n): G_ov.degree(n) for n in G_ov.nodes}
        avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Nodes", f"{node_count:,}")
        c2.metric("Hyperedges", f"{edge_count:,}")
        c3.metric("Avg Edge Size", f"{avg_edge_size:.1f}")
        c4.metric("Avg Node Degree", f"{avg_degree:.1f}")
        try:
            n_comp = sum(1 for _ in G_ov.s_connected_components(s=1))
        except Exception:
            n_comp = "?"
        c5.metric("Components (s=1)", n_comp)

        st.markdown("---")

        # --- Distribution plots ---
        col_left, col_right = st.columns(2)

        with col_left:
            # Degree distribution
            deg_vals = sorted(degrees.values(), reverse=True)
            df_deg = pd.DataFrame({"degree": deg_vals})
            log_scale = st.checkbox("Log scale (degree)", value=True, key="log_deg")
            fig_deg = px.histogram(
                df_deg, x="degree", nbins=50,
                title="Node Degree Distribution",
                labels={"degree": "Degree", "count": "Count"},
                log_y=log_scale,
            )
            fig_deg.update_layout(bargap=0.05, height=350)
            st.plotly_chart(fig_deg, use_container_width=True)

        with col_right:
            # Hyperedge size distribution
            df_es = pd.DataFrame({"size": edge_sizes})
            fig_es = px.histogram(
                df_es, x="size", nbins=30,
                title="Hyperedge Size Distribution",
                labels={"size": "Members per Hyperedge", "count": "Count"},
            )
            fig_es.update_layout(bargap=0.05, height=350)
            st.plotly_chart(fig_es, use_container_width=True)

        # --- Top nodes bar chart ---
        st.markdown("---")
        top_n = st.slider("Top N nodes by degree", 10, 50, 20, key="top_n_overview")
        top_sorted = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:top_n]
        df_top = pd.DataFrame(top_sorted, columns=["Node", "Degree"])
        df_top = df_top.iloc[::-1]  # reverse for horizontal bar (highest at top)

        fig_top = px.bar(
            df_top, x="Degree", y="Node", orientation="h",
            title=f"Top {top_n} Nodes by Degree",
            height=max(400, top_n * 22),
        )
        fig_top.update_layout(yaxis=dict(tickfont=dict(size=10)))
        st.plotly_chart(fig_top, use_container_width=True)
