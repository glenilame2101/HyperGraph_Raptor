import argparse
import os
import sys
from pathlib import Path
from typing import List

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


def collect_markdown_files(input_dir: Path) -> list[Path]:
    docs = sorted(input_dir.glob("*.md"))
    if docs:
        return docs

    collected: list[Path] = []
    for folder in sorted(input_dir.iterdir() if input_dir.exists() else []):
        if folder.is_dir():
            candidate = folder / f"{folder.name}.md"
            if candidate.exists():
                collected.append(candidate)
    return collected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate hypergraph JSON and HTML visualization from Markdown documents."
    )
    parser.add_argument("--doc-data-dir", default="Data", help="Folder containing .md files")
    parser.add_argument("--json-out-dir", default="artifacts/sg/graphs", help="Output folder for hypergraph JSON")
    parser.add_argument("--html-out-dir", default="artifacts/sg/html", help="Output folder for visualization HTML")
    parser.add_argument("--prompt-config", default=None)
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--chunk-overlap", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate even if json/html already exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace_root = Path(__file__).resolve().parent.parent

    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))

    from GraphReasoning.llm_client import create_llm
    from GraphReasoning.graph_generation import make_hypergraph_from_text
    from GraphReasoning.hypergraph_store import HypergraphBuilder
    from GraphReasoning.hypergraph_viz import visualize_hypergraph
    from GraphReasoning.prompt_config import get_prompt

    doc_dir = resolve_path(args.doc_data_dir, workspace_root)
    json_out_dir = resolve_path(args.json_out_dir, workspace_root)
    html_out_dir = resolve_path(args.html_out_dir, workspace_root)

    if args.prompt_config:
        os.environ["GRAPH_REASONING_PROMPT_CONFIG"] = str(resolve_path(args.prompt_config, workspace_root))

    os.makedirs(json_out_dir, exist_ok=True)
    os.makedirs(html_out_dir, exist_ok=True)

    docs = collect_markdown_files(doc_dir)
    if not docs:
        raise FileNotFoundError(f"No markdown docs found in: {doc_dir}")

    client = create_llm()

    def generate(
        system_prompt: str | None = None,
        prompt: str = "",
        response_model=HypergraphJSON,
        **_: dict,
    ):
        messages = [
            {"role": "system", "content": system_prompt or get_prompt("runtime", "viz_system_prompt")},
            {"role": "user", "content": prompt},
        ]
        return client.with_structured_output(response_model).invoke(messages)

    for i, doc_path in enumerate(docs):
        title = doc_path.stem
        graph_root = f"{i}_{title[:100]}"
        json_path = json_out_dir / f"{graph_root}.json"
        html_path = html_out_dir / f"{graph_root}.html"

        if not args.overwrite and json_path.exists() and html_path.exists():
            print(f"[skip] {title} (json+html already exist)")
            continue

        txt = doc_path.read_text(encoding="utf-8")
        print(f"[build] {title}")

        out_json, builder, _, _ = make_hypergraph_from_text(
            txt,
            generate,
            generate_figure=None,
            image_list=None,
            graph_root=graph_root,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            do_distill=False,
            do_relabel=False,
            repeat_refine=0,
            verbatim=False,
            data_dir=str(json_out_dir),
            force_rebuild=args.overwrite,
        )

        if not isinstance(builder, HypergraphBuilder):
            builder = HypergraphBuilder.load(out_json)

        visualize_hypergraph(builder, output_html=html_path)
        print(f"[ok] json={out_json} | html={html_path}")

    print("Done: hypergraph JSON + HTML visualization generated.")


if __name__ == "__main__":
    main()
