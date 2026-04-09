import json
import os
from pathlib import Path
from typing import Any


DEFAULT_PROMPTS: dict[str, Any] = {
    "graph": {
        "distill_system": "You are provided with a context chunk (delimited by ```). Respond with a concise heading, summary, and a bulleted list of key facts. Omit author names, references, and citations.",
        "distill_user": "Rewrite this text so it stands alone with all necessary context. Extract and organize any table data. Focus on factual content.\n\n```{input}```",
        "figure_system": "You analyze figures and diagrams. Report the factual content in detail. If the image is not informational, return an empty string. Include the full image path.",
        "figure_user": "Describe this figure factually. Extract data, labels, relationships, and structure.\n\n```{input}```",
        "graphmaker_system": (
            "You are a domain-agnostic knowledge-graph extractor.\n\n"
            "Given a text chunk delimited by ```, extract entities and binary relationships.\n\n"
            "Identify entities of any type relevant to the text, such as: people, characters, "
            "organizations, locations, dates, products, systems, processes, concepts, materials, "
            "events, metrics, rules, or constraints.\n\n"
            "Relation labels should be specific and descriptive in snake_case. Prefer precise verbs over generic ones:\n"
            "- Good: founded_by, located_in, causes, regulates, composed_of, defeated_by\n"
            "- Bad: related_to, associated_with, has_property, involves\n\n"
            "Rules:\n"
            "- Keep technical terms and abbreviations exactly as written\n"
            "- Each node needs an id (unique string) and a type (entity category)\n"
            "- Each edge needs source, target, and relation\n"
            "- Omit author names, citations, and generic filler\n"
            "- When an entity appears in multiple relationships, reuse the same node id\n"
            "- Resolve pronouns to named entities whenever possible\n\n"
            "Return a JSON object with keys \"nodes\" and \"edges\"."
        ),
        "graphmaker_user": "Context: ```{input}```\n\nExtract the knowledge graph. Return only a valid JSON object with keys \"nodes\" and \"edges\".",
    },
    "hypergraph": {
        "distill_system": "You are provided with a context chunk (delimited by ```). Respond with a concise heading, summary, and a bulleted list of key facts. Omit author names, references, and citations.",
        "distill_user": "Rewrite this text so it stands alone with all necessary context. Extract and organize any table data. Focus on factual content.\n\n```{input}```",
        "figure_system": "You analyze figures and diagrams. Report the factual content in detail. If the image is not informational, return an empty string. Include the full image path.",
        "figure_user": "Describe this figure factually. Extract data, labels, relationships, and structure.\n\n```{input}```",
        "graphmaker_system": (
            "You are a domain-agnostic hypergraph relationship extractor. Read a text chunk and extract "
            "n-ary relationships (hyperedges) that capture how multiple entities participate in one fact, "
            "event, process, role assignment, property assignment, or causal chain.\n\n"
            "A hyperedge connects multiple source entities to multiple target entities through one specific relation. "
            "Use this for technical manuals, scientific text, policies, narratives, movie summaries, tables, and bullet lists.\n\n"
            "Extraction strategy:\n"
            "1. Read the full chunk before extracting. Identify canonical entities such as documents, products, films, "
            "characters, organizations, people, places, dates, themes, systems, constraints, and outcomes.\n"
            "2. Convert each meaningful statement into one hyperedge:\n"
            "   - source: the main subject(s), actor(s), input(s), cause(s), or owner(s)\n"
            "   - relation: a specific normalized verb phrase in snake_case\n"
            "   - target: the object(s), value(s), recipient(s), effect(s), or result(s)\n"
            "3. For tables, treat each row as facts. Example: a movie title can be linked to release date, runtime, studio, rating, and box office.\n"
            "4. For cast lists or role lists, connect the character/entity to the actor or role with a specific relation such as voiced_by, played_by, works_as, owns, rules, helps, opposes.\n"
            "5. For plot summaries, resolve pronouns to named entities whenever possible and extract concrete story events.\n"
            "6. When a section heading is only organizational (for example: Main Characters, Act I, Key Themes), do NOT extract it as an entity unless the text explicitly treats it as a real concept.\n"
            "7. Do NOT create vague nodes like \"details\", \"overview\", \"section\", \"movie information\", or \"plot summary\" unless they are actual semantic entities in the text.\n\n"
            "Rules:\n"
            "- Be concrete, not generic. Prefer directed_by over related_to. Prefer has_release_date over has_property.\n"
            "- Preserve proper nouns and literal values exactly as written when they are meaningful targets.\n"
            "- Reuse the exact same entity string across events for the same entity.\n"
            "- Include metadata facts, role assignments, world-building facts, and causal story events when present.\n"
            "- Each event must have at least one source and one target.\n"
            "- Return only JSON.\n\n"
            "Return a JSON object with one key \"events\". Each event has:\n"
            "- \"source\": list[str]\n"
            "- \"relation\": str\n"
            "- \"target\": list[str]\n\n"
            "Examples:\n"
            "1. {\"source\": [\"The SpongeBob SquarePants Movie\"], \"relation\": \"has_release_date\", \"target\": [\"November 19, 2004\"]}\n"
            "2. {\"source\": [\"SpongeBob SquarePants\"], \"relation\": \"voiced_by\", \"target\": [\"Tom Kenny\"]}\n"
            "3. {\"source\": [\"Plankton\"], \"relation\": \"frames\", \"target\": [\"Mr. Krabs\"]}\n"
            "4. {\"source\": [\"TMS\", \"Munich warehouse\", \"DHL\"], \"relation\": \"routes_shipments_to\", \"target\": [\"customers in Austria\", \"48-hour SLA\"]}"
        ),
        "graphmaker_user": (
            "Context: ```{input}```\n\n"
            "Extract all meaningful hyperedges. Include factual metadata, table rows, list items, role assignments, "
            "narrative events, causal links, constraints, and dependencies. Ignore formatting-only headings and generic "
            "document labels. Return only a valid JSON object with key \"events\"."
        ),
    },
    "graph_tools": {
        "node_rename_system": (
            "You are an ontological graph maker. You rename nodes in complex networks to be clearer and more descriptive.\n\n"
            "Rules:\n"
            "- Return ONLY the new name as a single phrase, nothing else.\n"
            "- Do not include explanations, quotes, or punctuation beyond what the name needs.\n"
            "- Keep the name concise (1-5 words).\n"
            "- Preserve domain-specific terms and abbreviations."
        ),
        "node_rename_user": "Rename this network node to be more descriptive of its role:\n\nCurrent name: {node_name}\n\nNew name:",
        "community_summary_system": (
            "You are a domain expert who summarizes groups of related entities and relationships from a knowledge graph.\n\n"
            "Write a structured summary with:\n"
            "1. A one-sentence overview of what this community of nodes represents.\n"
            "2. The key entities and their roles.\n"
            "3. The most important relationships and patterns.\n"
            "4. Any notable constraints, dependencies, or hierarchies.\n\n"
            "Be specific and factual. Use the exact entity names from the data. Do not invent information not present in the relationships."
        ),
        "extract_keywords_system": (
            "You are a strict scientific keyword extractor.\n\n"
            "Your job is to extract ONLY the concrete scientific entities mentioned "
            "in the text (e.g., materials, chemicals, biological entities, properties). "
            "Do NOT extract abstract concepts, verbs, or relational words.\n\n"
            "Rules:\n"
            "- Output ONLY valid JSON.\n"
            '- Format: {"keywords": ["keyword1", "keyword2", ...]}\n'
            "- Extract ONLY MATERIALS / SUBSTANCES / SPECIFIC ENTITIES.\n"
            "- DO NOT extract:\n"
            "    - verbs (e.g., relate, interact, form, behave)\n"
            "    - question words (how, why, what)\n"
            "    - abstract concepts (mechanistic relation, mechanism, relationship)\n"
            "    - adjectives or descriptors (mechanistic, structural, functional)\n"
            "- Keep acronyms in original case (e.g., PCL, PLA, PEG).\n"
            "- Otherwise, lowercase all extracted words.\n"
            "- No explanations, no markdown, no code fences.\n\n"
            "Examples:\n\n"
            "Context: What is the capital of the United States?\n"
            '{"keywords": ["united states"]}\n\n'
            "Context: How can silk mechanistically relate to PCL?\n"
            '{"keywords": ["silk", "PCL"]}\n\n'
            "Context: What technology is Taiwan famous for?\n"
            '{"keywords": ["taiwan", "technology", "semiconductor"]}\n\n'
            "Context: What is CVD uniformity and etching uniformity?\n"
            '{"keywords": ["cvd", "etching", "uniformity"]}'
        ),
        "extract_keywords_user": "Context: ```{question}```",
        "extract_material_keywords_system": (
            "You are a strict keyword extractor.\n\n"
            "Rules:\n"
            '- Output EXACTLY one JSON object: {"keywords": [<strings>]} with no extra text.\n'
            "- If any materials/chemicals/compounds are present, RETURN ONLY those (lowercased, deduplicated).\n"
            "- Otherwise, include domain nouns (processes, properties, entities), but never verbs, stopwords, or question words.\n"
            "- Preserve common acronyms (e.g., CVD, PLA) in their original case; otherwise lowercase.\n"
            "- No explanations.\n\n"
            "Example:\n"
            "Context: ```What is a formulation for a composite design that can combine chitosan and silk?```\n"
            '{"keywords": ["chitosan", "silk"]}'
        ),
        "extract_material_keywords_user": "Context: ```{question}```",
        "local_search_system": (
            "You answer questions using information retrieved from a knowledge graph.\n\n"
            "You will receive a report containing entities and relationships found by traversing "
            "shortest paths between relevant nodes in the graph.\n\n"
            "Rules:\n"
            "- Synthesize the report into a clear, detailed answer to the user's question.\n"
            "- Use only the facts provided in the report. Do not hallucinate or add external knowledge.\n"
            "- If the report does not contain enough information to answer, say so explicitly and explain what is missing.\n"
            "- Cite specific entities and relationships from the report to support your answer."
        ),
        "local_search_user": (
            "Question: {question}\n\n"
            "Knowledge graph report (entities and relationships from shortest-path traversal):\n"
            "{information}\n\n"
            "Provide a detailed answer based on the report above."
        ),
        "query_validation_system": (
            "You validate whether an answer adequately addresses a question.\n\n"
            "Respond with EXACTLY this format:\n"
            "Line 1: YES or NO\n"
            "Line 2+: (only if NO) A brief explanation of what is missing or incorrect.\n\n"
            "Do not include any other text."
        ),
        "query_validation_user": "Question: {question}\n\nAnswer to validate:\n{response}",
        "global_search_system": (
            "You answer questions by synthesizing information from multiple knowledge graph communities.\n\n"
            "You will receive:\n"
            "- A community summary (a cluster of related entities)\n"
            "- Supporting information from graph traversal\n"
            "- Your current working answer from previous communities (if any)\n\n"
            "Rules:\n"
            "- Integrate new information into your current answer. Do not discard prior findings unless contradicted.\n"
            "- If the new community adds nothing relevant, keep your current answer unchanged.\n"
            "- Use only the facts provided. Do not hallucinate.\n"
            "- Build toward a comprehensive, well-structured final answer."
        ),
        "global_search_user": (
            "Question: {question}\n\n"
            "Community summary:\n{summary}\n\n"
            "Supporting graph information:\n{information}\n\n"
            "Current working answer:\n{last_response}\n\n"
            "Update your answer by integrating any new relevant information from this community."
        ),
    },
    "raptor": {
        "summarize_user": (
            "You are an expert summarizer. Your task is to condense the following passages "
            "into a single, information-dense paragraph.\n\n"
            "Rules:\n"
            "- Preserve ALL key facts, named entities, numbers, and relationships.\n"
            "- Do not add information not present in the source text.\n"
            "- Do not include preamble like \"This passage discusses...\" -- go straight to the content.\n"
            "- Merge overlapping information rather than repeating it.\n"
            "- Prioritize concrete facts over general statements.\n\n"
            "Passages:\n{text}\n\n"
            "Summary:"
        ),
    },
    "runtime": {
        "default_system_prompt": "You extract structured relationships from a text chunk. Return a JSON object with one key \"events\". Each event has: source (list[str]), relation (str), target (list[str]). Be thorough and specific.",
        "figure_system_prompt": "You are an assistant who describes figures and diagrams in factual detail.",
        "figure_user_prompt": "Describe this figure in detail. Include all data, labels, axes, legends, and relationships shown.",
        "viz_system_prompt": (
            "You extract structured hypergraph events from text. Return a JSON object with one key \"events\". "
            "Each event has: source (list[str]), relation (str), target (list[str]). "
            "Use specific snake_case relation labels. Be thorough -- extract all meaningful relationships."
        ),
    },
}


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_config_path(config_path: str | None = None) -> Path:
    if config_path:
        return Path(config_path).expanduser().resolve()
    env_path = os.getenv("GRAPH_REASONING_PROMPT_CONFIG")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return (Path(__file__).resolve().parent.parent / "prompt_config.json").resolve()


def load_prompt_config(config_path: str | None = None) -> dict[str, Any]:
    resolved = _resolve_config_path(config_path)
    if not resolved.exists():
        return DEFAULT_PROMPTS

    try:
        with resolved.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            return DEFAULT_PROMPTS
        return _deep_merge(DEFAULT_PROMPTS, loaded)
    except Exception:
        return DEFAULT_PROMPTS


def get_prompt(section: str, key: str, config_path: str | None = None, **kwargs) -> str:
    prompts = load_prompt_config(config_path=config_path)
    section_data = prompts.get(section, {}) if isinstance(prompts, dict) else {}
    template = section_data.get(key, "") if isinstance(section_data, dict) else ""
    if not isinstance(template, str):
        return ""
    if kwargs:
        try:
            return template.format(**kwargs)
        except Exception:
            return template
    return template
