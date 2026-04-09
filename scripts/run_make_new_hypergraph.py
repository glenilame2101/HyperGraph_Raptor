import argparse
import base64
import glob
import os
import pickle
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
from dotenv import load_dotenv
from pydantic import BaseModel


class Event(BaseModel):
    source: List[str]
    target: List[str]
    relation: str


class HypergraphJSON(BaseModel):
    events: List[Event]


def resolve_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and merge hypergraphs from markdown files.")
    parser.add_argument("--doc-data-dir", default="Data")
    parser.add_argument("--artifacts-root", default="artifacts/sg")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--data-dir-output", default=None)
    parser.add_argument("--cache-dir", default="artifacts/cache/chunks")
    parser.add_argument("--embedding-file", default="hypergraph_embeedings.pkl")
    parser.add_argument("--thread-index", type=int, default=0)
    parser.add_argument("--total-threads", type=int, default=1)
    parser.add_argument("--merge-every", type=int, default=100)
    parser.add_argument("--bge-url", default="http://127.0.0.1:8080")
    parser.add_argument("--bge-model", default="BAAI/bge-m3")
    parser.add_argument("--max-tokens", type=int, default=20000)
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--chunk-overlap", type=int, default=0)
    parser.add_argument("--similarity-threshold", type=float, default=0.9)
    parser.add_argument("--llm-timeout", type=float, default=120.0)
    parser.add_argument("--llm-retries", type=int, default=6)
    parser.add_argument("--llm-retry-delay", type=float, default=2.0)
    parser.add_argument("--llm-retry-backoff", type=float, default=2.0)
    parser.add_argument("--llm-max-delay", type=float, default=30.0)
    parser.add_argument("--prompt-config", default=None)
    parser.add_argument("--no-ssl-verify", action="store_true")
    parser.add_argument("--no-proxy", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    return parser.parse_args()


def resolve_docs(doc_data_dir: str, workspace_root: Path) -> list[str]:
    root = resolve_path(doc_data_dir, workspace_root)
    docs = sorted(str(path) for path in root.glob("*.md") if path.exists())
    if docs:
        return docs
    for folder in sorted(root.iterdir() if root.exists() else []):
        if folder.is_dir():
            candidate = folder / f"{folder.name}.md"
            if candidate.exists():
                docs.append(str(candidate))
    return sorted(docs)


def image_to_base64_data_uri(file_path: str) -> str:
    with open(file_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def main() -> None:
    args = parse_args()
    workspace_root = Path(__file__).resolve().parent.parent
    load_dotenv(dotenv_path=workspace_root / ".env")

    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))

    from GraphReasoning.prompt_config import get_prompt
    from GraphReasoning.llm_client import create_llm, create_embed_client

    artifacts_root_path = resolve_path(args.artifacts_root, workspace_root)
    data_dir_path = resolve_path(args.data_dir, workspace_root) if args.data_dir else (artifacts_root_path / "graphs").resolve()
    data_dir_output_path = resolve_path(args.data_dir_output, workspace_root) if args.data_dir_output else (artifacts_root_path / "integrated").resolve()
    cache_dir_path = resolve_path(args.cache_dir, workspace_root)
    os.environ["GRAPH_REASONING_CACHE_DIR"] = str(cache_dir_path)
    if args.prompt_config:
        os.environ["GRAPH_REASONING_PROMPT_CONFIG"] = str(resolve_path(args.prompt_config, workspace_root))

    from GraphReasoning.graph_generation import make_hypergraph_from_text, add_new_hypersubgraph_from_text
    from GraphReasoning.graph_tools import (
        generate_hypernode_embeddings,
        load_embeddings,
        save_embeddings,
        update_hypernode_embeddings,
    )
    from GraphReasoning.hypergraph_store import HypergraphBuilder

    try:
        from trustcall import create_extractor  # noqa: F401 — kept as optional fallback
    except ImportError:
        pass  # not required when using with_structured_output

    client = create_llm(
        timeout=args.llm_timeout,
        max_tokens=args.max_tokens,
        verify_ssl=not args.no_ssl_verify,
        trust_env=not args.no_proxy,
    )

    if not args.skip_preflight:
        import httpx as _httpx
        try:
            base_url = os.getenv("URL", "").rstrip("/")
            api_key = os.getenv("OPENAI_API_KEY", "")
            preflight_url = base_url if base_url.endswith("/v1") else f"{base_url}/v1"
            preflight_client = _httpx.Client(timeout=min(10.0, args.llm_timeout))
            response = preflight_client.get(f"{preflight_url}/models", headers={"Authorization": f"Bearer {api_key}"})
            print(f"[preflight] GET {preflight_url}/models -> {response.status_code}")
            if response.status_code >= 500:
                print("[preflight] Endpoint is reachable but returned server error.")
        except Exception as exc:
            print(f"[preflight] Connection check failed before generation: {exc!r}")
            print("[preflight] Try --no-proxy or --no-ssl-verify if your environment uses proxy interception or self-signed certs.")

    os.makedirs(data_dir_path, exist_ok=True)
    os.makedirs(data_dir_output_path, exist_ok=True)
    os.makedirs(cache_dir_path, exist_ok=True)

    print(f"[paths] workspace_root={workspace_root}")
    print(f"[paths] doc_data_dir={resolve_path(args.doc_data_dir, workspace_root)}")
    print(f"[paths] data_dir={data_dir_path}")
    print(f"[paths] data_dir_output={data_dir_output_path}")
    print(f"[paths] cache_dir={cache_dir_path}")

    embedding_tokenizer = None
    embedding_model = create_embed_client(base_url=args.bge_url, model=args.bge_model)

    embedding_path = data_dir_path / args.embedding_file
    if os.path.exists(embedding_path):
        print(f"Found existing embedding file: {embedding_path}")
        node_embeddings = load_embeddings(str(embedding_path))
    else:
        node_embeddings = generate_hypernode_embeddings([], embedding_tokenizer, embedding_model)
        save_embeddings(node_embeddings, str(embedding_path))

    doc_list = resolve_docs(args.doc_data_dir, workspace_root)
    if not doc_list:
        raise FileNotFoundError(f"No markdown docs found in: {resolve_path(args.doc_data_dir, workspace_root)}")

    int_prefix = re.compile(r"^(\d+)_")

    def extract_idx(path: str) -> int:
        match = int_prefix.match(os.path.basename(path))
        return int(match.group(1)) if match else -1

    current_merged_i = 0
    merged_graph_list = []
    if args.total_threads == 1:
        merged_graph_list = sorted(
            glob.glob(str(data_dir_output_path / "*_integrated.json")),
            reverse=True,
            key=extract_idx,
        )
        current_merged_i = extract_idx(merged_graph_list[0]) if merged_graph_list else 0

    print(f"[config] model={os.getenv('MODEL_NAME')} | url={os.getenv('URL')}")
    print(f"[config] thread={args.thread_index}/{args.total_threads} | merge_every={args.merge_every}")
    print(f"[config] current_merged_i={current_merged_i} | doc_count={len(doc_list)}")
    print(
        "[config] llm: "
        f"timeout={args.llm_timeout}s, retries={args.llm_retries}, "
        f"retry_delay={args.llm_retry_delay}s, backoff={args.llm_retry_backoff}"
    )

    default_system_prompt = get_prompt("runtime", "default_system_prompt")

    def save_embeddings_with_retry(payload, file_path: Path, retries: int = 6, base_delay: float = 1.0) -> bool:
        for attempt in range(1, retries + 1):
            try:
                save_embeddings(payload, str(file_path))
                return True
            except PermissionError as exc:
                if attempt >= retries:
                    print(
                        f"[embeddings] permission denied after {retries} attempts: {exc!r}; "
                        f"path={file_path}"
                    )
                    return False
                wait_s = base_delay * attempt
                print(
                    f"[embeddings] permission error (attempt {attempt}/{retries}): {exc!r}; "
                    f"path={file_path}; retrying in {wait_s:.1f}s"
                )
                time.sleep(wait_s)
            except Exception as exc:
                print(f"[embeddings] failed to save: {exc!r}; path={file_path}")
                return False
        return False

    _generate_call_count = 0

    def generate(
        system_prompt: str | None = None,
        prompt: str = "",
        temperature: float = 0.333,
        max_tokens: int | None = None,
        response_model=HypergraphJSON,
        **_: dict,
    ):
        nonlocal _generate_call_count
        _generate_call_count += 1
        call_id = _generate_call_count
        effective_system_prompt = system_prompt or default_system_prompt
        messages = [
            {"role": "system", "content": effective_system_prompt},
            {"role": "user", "content": prompt},
        ]
        prompt_chars = len(prompt)
        print(
            f"[llm] call #{call_id} | model={os.getenv('MODEL_NAME')} | "
            f"prompt_len={prompt_chars} chars | schema={response_model.__name__}"
        )
        structured_llm = client.with_structured_output(response_model)
        delay_seconds = max(0.0, args.llm_retry_delay)
        total_attempts = max(1, args.llm_retries + 1)
        last_exception: Exception | None = None
        for attempt in range(1, total_attempts + 1):
            t0 = time.time()
            try:
                resp = structured_llm.invoke(messages)
                elapsed = time.time() - t0
                if resp is None:
                    raise ValueError("with_structured_output returned None")
                detail = ""
                if hasattr(resp, "events"):
                    detail = f" | events={len(resp.events)}"
                elif hasattr(resp, "nodes"):
                    detail = f" | nodes={len(resp.nodes)}, edges={len(resp.edges)}"
                print(
                    f"[llm] call #{call_id} OK "
                    f"(attempt {attempt}/{total_attempts}, {elapsed:.1f}s){detail}"
                )
                return resp
            except Exception as exc:
                elapsed = time.time() - t0
                last_exception = exc
                if attempt >= total_attempts:
                    print(
                        f"[llm] call #{call_id} FAILED after {total_attempts} attempts "
                        f"({elapsed:.1f}s): {exc!r}"
                    )
                    break
                print(
                    f"[llm] call #{call_id} attempt {attempt}/{total_attempts} failed "
                    f"({elapsed:.1f}s): {exc!r} | retrying in {delay_seconds:.1f}s"
                )
                time.sleep(delay_seconds + random.uniform(0, 0.5))
                delay_seconds = min(args.llm_max_delay, max(0.1, delay_seconds * args.llm_retry_backoff))

        if last_exception is not None:
            raise last_exception
        raise RuntimeError("LLM generation failed without a captured exception")

    def generate_figure(
        image: str,
        system_prompt: str | None = None,
        prompt: str = "",
        temperature: float = 0.0,
        **_: dict,
    ):
        try:
            image_path = Path(image)
            if not image_path.exists():
                image_path = next(Path(".").glob(f"**/{Path(image).name}"))
            image_uri = image_to_base64_data_uri(str(image_path))
            delay_seconds = max(0.0, args.llm_retry_delay)
            total_attempts = max(1, args.llm_retries + 1)
            for attempt in range(1, total_attempts + 1):
                try:
                    response = client.invoke([
                        {
                            "role": "system",
                            "content": system_prompt or get_prompt("runtime", "figure_system_prompt"),
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt or get_prompt("runtime", "figure_user_prompt")},
                                {"type": "image_url", "image_url": {"url": image_uri}},
                            ],
                        },
                    ])
                    return getattr(response, "content", "") or ""
                except Exception as exc:
                    if attempt >= total_attempts:
                        raise
                    print(
                        f"[llm-vision] request failed (attempt {attempt}/{total_attempts}): {exc!r}; "
                        f"retrying in {delay_seconds:.1f}s"
                    )
                    time.sleep(delay_seconds + random.uniform(0, 0.5))
                    delay_seconds = min(args.llm_max_delay, max(0.1, delay_seconds * args.llm_retry_backoff))
        except Exception:
            return ""

    G = None
    if merged_graph_list:
        try:
            G = HypergraphBuilder.load(merged_graph_list[0])
            print(f"Loaded existing integrated graph: {merged_graph_list[0]} — {G.node_count} nodes, {G.edge_count} edges")
        except Exception as exc:
            print(f"Could not load existing integrated graph: {exc!r}")

    with torch.no_grad():
        for i, doc in enumerate(doc_list):
            if i % args.total_threads != args.thread_index:
                continue
            if i < current_merged_i:
                continue

            title = os.path.basename(doc).rsplit(".md", 1)[0]
            graph_root = f"{i}_{title[:100]}"
            graph_path = data_dir_path / f"{graph_root}.json"
            integrated_path = data_dir_output_path / f"{graph_root}_integrated.json"

            with open(doc, "r", encoding="utf-8") as handle:
                txt = handle.read()

            needs_generation = not os.path.exists(graph_path)
            if not needs_generation:
                try:
                    existing_builder = HypergraphBuilder.load(graph_path)
                    if existing_builder.node_count == 0:
                        print(f"[generation] Found empty hypergraph JSON, regenerating: {graph_path}")
                        os.remove(graph_path)
                        needs_generation = True
                except Exception:
                    print(f"[generation] Could not read hypergraph JSON, regenerating: {graph_path}")
                    try:
                        os.remove(graph_path)
                    except Exception:
                        pass
                    needs_generation = True

            if needs_generation:
                print(f"[generation] Building KG for doc {i}: {title}")
                try:
                    t0 = datetime.now()
                    graph_path, _, _, _ = make_hypergraph_from_text(
                        txt,
                        generate,
                        generate_figure,
                        image_list="",
                        graph_root=graph_root,
                        do_distill=False,
                        do_relabel=False,
                        chunk_size=args.chunk_size,
                        chunk_overlap=args.chunk_overlap,
                        repeat_refine=0,
                        verbatim=False,
                        data_dir=str(data_dir_path),
                    )
                    print(f"[generation] done in {datetime.now() - t0}")
                except Exception as exc:
                    print(f"[generation] failed for doc {i}: {exc!r}")
                    time.sleep(2)
                    continue

            if os.path.exists(integrated_path):
                print(f"[merge] already exists, skipping: {integrated_path}")
                try:
                    G = HypergraphBuilder.load(integrated_path)
                except Exception:
                    pass
                continue

            try:
                H_doc = HypergraphBuilder.load(graph_path)
                if H_doc.node_count == 0:
                    print(f"[merge] doc graph empty after generation: {graph_path}")
                    continue
            except Exception as exc:
                print(f"[merge] failed loading doc graph {graph_path!r}: {exc!r}")
                continue

            if G is None:
                G = H_doc
                G.save(integrated_path)
                node_embeddings = update_hypernode_embeddings(node_embeddings, G, embedding_tokenizer, embedding_model)
                save_embeddings_with_retry(node_embeddings, embedding_path)
                print(f"[merge] initialized integrated graph with doc {i}; embeddings saved")
                continue

            do_simplify_graph = i % args.merge_every == 0
            size_threshold = 10 if do_simplify_graph else 0

            merge_attempts = 6
            merge_ok = False
            for merge_attempt in range(1, merge_attempts + 1):
                try:
                    integrated_path, G, _, node_embeddings, _ = add_new_hypersubgraph_from_text(
                        txt="",
                        node_embeddings=node_embeddings,
                        tokenizer=embedding_tokenizer,
                        model=embedding_model,
                        original_graph=G,
                        data_dir_output=str(data_dir_output_path),
                        graph_root=graph_root,
                        do_simplify_graph=do_simplify_graph,
                        do_relabel=False,
                        size_threshold=size_threshold,
                        do_update_node_embeddings=do_simplify_graph,
                        repeat_refine=0,
                        similarity_threshold=args.similarity_threshold,
                        do_Louvain_on_new_graph=False,
                        return_only_giant_component=False,
                        save_common_graph=False,
                        G_to_add=H_doc,
                        graph_pkl_to_add=None,
                        sub_dfs=[],
                        verbatim=True,
                    )
                    embeddings_saved = save_embeddings_with_retry(node_embeddings, embedding_path)
                    if embeddings_saved:
                        print(f"[merge] merged doc {i}; embeddings saved")
                    else:
                        print(f"[merge] merged doc {i}; embeddings save skipped due to lock")
                    merge_ok = True
                    break
                except PermissionError as exc:
                    lock_path = getattr(exc, "filename", None)
                    if merge_attempt >= merge_attempts:
                        print(
                            f"[merge] failed for doc {i} after {merge_attempts} attempts: {exc!r}. "
                            f"graph_root={graph_root}, output_dir={data_dir_output_path}, locked_path={lock_path}"
                        )
                        break
                    wait_s = 2 * merge_attempt
                    print(
                        f"[merge] doc {i} permission error (attempt {merge_attempt}/{merge_attempts}): {exc!r}; "
                        f"locked_path={lock_path}; retrying in {wait_s}s"
                    )
                    time.sleep(wait_s)
                except Exception as exc:
                    print(f"[merge] failed for doc {i}: {exc!r}")
                    break

            if not merge_ok:
                continue

    print("Pipeline complete")


if __name__ == "__main__":
    main()
