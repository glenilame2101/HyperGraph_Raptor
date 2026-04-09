"""D3.js visualization for RAPTOR tree / DAG.

Generates a self-contained HTML file with two switchable views:

1. **Tree view** — ``d3.tree()`` tidy dendrogram from the strict-tree JSON.
2. **DAG view** — force-directed layout with soft-clustering edges, nodes
   stratified by level via ``forceY``.

Features:
- Node color by level (ordinal palette)
- Node radius by token_count
- Link thickness / opacity by membership weight
- Retrieval overlay highlighting
- Search filter
- Weight threshold slider
- Click-to-expand subtree
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Union

from .raptor_export import export_dag_json, export_tree_json
from .raptor_tree import RaptorIndex

logger = logging.getLogger(__name__)


def visualize_raptor(
    index: RaptorIndex,
    output_html: Union[str, Path] = "raptor_viz.html",
    *,
    height: str = "100vh",
    bgcolor: str = "#0f172a",
    retrieval_overlay: Optional[dict] = None,
) -> Path:
    """Render a RAPTOR index to an interactive D3.js HTML file.

    Parameters
    ----------
    index : RaptorIndex
        The built RAPTOR index.
    output_html : str or Path
        Where to write the HTML file.
    height : str
        CSS height for the canvas.
    bgcolor : str
        Background color.
    retrieval_overlay : dict, optional
        ``{"retrieved_node_ids": [...], "scores": [...]}`` to highlight.
    """
    # Build data inline
    # -- DAG data (nodes + links) ---
    nodes_data = []
    for node in index.all_nodes():
        nodes_data.append({
            "id": node.id,
            "level": node.level,
            "type": node.type,
            "name": node.text[:200] + ("..." if len(node.text) > 200 else ""),
            "full_text": node.text,
            "token_count": node.token_count,
        })

    links_data = []
    for edge in index.edges:
        links_data.append({
            "source": edge.source,
            "target": edge.target,
            "weight": round(edge.weight, 4),
        })

    dag_data = {"nodes": nodes_data, "links": links_data}

    # -- Tree data (nested) ---
    from .raptor_export import _build_tree_edges, _build_nested

    child_to_parent = _build_tree_edges(index)
    parent_to_children: dict[str, list[str]] = {}
    for child, parent in child_to_parent.items():
        parent_to_children.setdefault(parent, []).append(child)

    all_children_set = set(child_to_parent.keys())
    all_parents_set = set(child_to_parent.values())
    roots = all_parents_set - all_children_set
    orphans = set(index.nodes.keys()) - all_children_set - all_parents_set

    root_list = sorted(roots | orphans)
    if len(root_list) == 1:
        tree_data = _build_nested(root_list[0], index, parent_to_children)
    elif root_list:
        tree_data = {
            "id": "raptor_root", "name": "RAPTOR Root",
            "level": index.max_level + 1, "type": "root", "token_count": 0,
            "children": [_build_nested(r, index, parent_to_children) for r in root_list],
        }
    else:
        tree_data = {"id": "raptor_root", "name": "Empty", "level": 0, "type": "root", "token_count": 0}

    overlay_data = retrieval_overlay or {"retrieved_node_ids": [], "scores": []}

    html = _TEMPLATE.format(
        bgcolor=bgcolor,
        height=height,
        dag_json=json.dumps(dag_data, ensure_ascii=False),
        tree_json=json.dumps(tree_data, ensure_ascii=False),
        overlay_json=json.dumps(overlay_data, ensure_ascii=False),
        max_level=index.max_level,
    )

    out = Path(output_html).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    logger.info("RAPTOR visualization saved -> %s", out)
    return out


# ---------------------------------------------------------------------------
# HTML template with embedded D3.js
# ---------------------------------------------------------------------------

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>RAPTOR Tree Visualization</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    html,body{{height:100%;background:{bgcolor};color:#e2e8f0;font-family:'Segoe UI',system-ui,sans-serif;overflow:hidden}}
    #canvas{{position:absolute;inset:0;width:100%;height:{height}}}
    svg{{width:100%;height:100%}}

    .panel{{
      position:absolute;z-index:20;background:rgba(15,23,42,0.92);
      border:1px solid rgba(148,163,184,0.18);border-radius:12px;
      padding:14px 16px;backdrop-filter:blur(10px);
      box-shadow:0 12px 30px rgba(0,0,0,0.35);
    }}
    #controls{{top:16px;left:16px;width:320px}}
    #details{{left:16px;bottom:16px;width:450px;max-height:280px;overflow:auto}}
    #stats{{top:16px;right:16px;min-width:200px}}

    h2{{font-size:17px;margin-bottom:8px;color:#e2e8f0}}
    .muted{{color:#94a3b8;font-size:11px;line-height:1.4}}
    .row{{margin:5px 0;font-size:12px}}
    .mono{{font-family:Consolas,monospace;word-break:break-word;font-size:11px}}
    label{{font-size:12px;color:#cbd5e1;display:block;margin-top:8px}}
    input[type=text]{{width:100%;padding:7px 10px;border:1px solid rgba(148,163,184,0.24);border-radius:8px;background:rgba(30,41,59,0.9);color:#fff;font-size:12px;margin:4px 0}}
    input[type=range]{{width:100%;margin:4px 0}}
    select{{width:100%;padding:6px 8px;border:1px solid rgba(148,163,184,0.24);border-radius:8px;background:rgba(30,41,59,0.9);color:#fff;font-size:12px;margin:4px 0}}

    .btn-row{{display:flex;gap:8px;margin:10px 0}}
    .btn{{padding:6px 14px;border:1px solid rgba(148,163,184,0.3);border-radius:8px;
          background:rgba(30,41,59,0.8);color:#e2e8f0;cursor:pointer;font-size:12px;
          transition:background 0.15s}}
    .btn:hover{{background:rgba(59,130,246,0.3)}}
    .btn.active{{background:rgba(59,130,246,0.5);border-color:#3b82f6}}

    .legend{{display:flex;gap:12px;margin:8px 0;font-size:11px;flex-wrap:wrap}}
    .dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:middle}}

    .node-label{{fill:#e2e8f0;font-size:10px;pointer-events:none}}
    .link{{fill:none}}
    .tree-link{{fill:none;stroke:#475569;stroke-width:1.5}}
  </style>
</head>
<body>
  <div id="canvas"><svg id="svg"></svg></div>

  <div id="controls" class="panel">
    <h2>RAPTOR Viewer</h2>
    <div class="muted">Hierarchical RAG index. Leaves=chunks, internal nodes=LLM summaries.<br>
    Soft clustering: a node may have multiple parents (DAG view).</div>

    <div class="btn-row">
      <button class="btn active" id="btnDAG">DAG View</button>
      <button class="btn" id="btnTree">Tree View</button>
    </div>

    <label>Search nodes</label>
    <input id="searchInput" type="text" placeholder="Type to filter by text..."/>

    <label>Min edge weight: <span id="weightVal">0.00</span></label>
    <input id="weightSlider" type="range" min="0" max="1" step="0.01" value="0"/>

    <label>Show level</label>
    <select id="levelFilter"><option value="all">All levels</option></select>

    <div class="legend" id="legendBox"></div>
    <div class="muted" style="margin-top:8px">Drag nodes &middot; Scroll to zoom &middot; Hover for details</div>
  </div>

  <div id="stats" class="panel"></div>
  <div id="details" class="panel">Hover over a node for details.</div>

  <script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
  <script>
  (function(){{

    /* ====== DATA ====== */
    const DAG    = {dag_json};
    const TREE   = {tree_json};
    const OVERLAY= {overlay_json};
    const MAX_LVL= {max_level};

    /* ====== PALETTE ====== */
    const LEVEL_COLORS = [
      "#3b82f6","#10b981","#f59e0b","#ef4444","#a855f7",
      "#ec4899","#06b6d4","#84cc16","#f97316","#6366f1"
    ];
    function lvlColor(lvl){{ return LEVEL_COLORS[lvl % LEVEL_COLORS.length]; }}

    /* ====== SETUP ====== */
    const W = window.innerWidth, H = window.innerHeight;
    const svg = d3.select("#svg").attr("viewBox",[0,0,W,H]);
    const gMain = svg.append("g");

    // zoom
    const zoom = d3.zoom().scaleExtent([0.05,8]).on("zoom", e => gMain.attr("transform",e.transform));
    svg.call(zoom);

    // layers
    const gLinks = gMain.append("g").attr("class","links-layer");
    const gNodes = gMain.append("g").attr("class","nodes-layer");
    const gLabels= gMain.append("g").attr("class","labels-layer");

    /* ====== POPULATE CONTROLS ====== */
    const levelSel = document.getElementById("levelFilter");
    for(let l=0;l<=MAX_LVL;l++){{
      const o=document.createElement("option");
      o.value=l; o.textContent=`Level ${{l}} (${{l===0?"leaves":"summaries"}})`;
      levelSel.appendChild(o);
    }}

    // legend
    const legendBox = document.getElementById("legendBox");
    for(let l=0;l<=MAX_LVL;l++){{
      legendBox.innerHTML += `<div><span class="dot" style="background:${{lvlColor(l)}}"></span>L${{l}}</div>`;
    }}

    /* ====== OVERLAY SET ====== */
    const overlaySet = new Set(OVERLAY.retrieved_node_ids||[]);
    const overlayScores = {{}};
    (OVERLAY.retrieved_node_ids||[]).forEach((id,i) => {{
      overlayScores[id] = (OVERLAY.scores||[])[i] || 0;
    }});

    /* ====== STATE ====== */
    let currentView = "dag"; // "dag" | "tree"
    let sim; // d3 force simulation

    /* ====== DAG VIEW ====== */
    function renderDAG(){{
      gLinks.selectAll("*").remove();
      gNodes.selectAll("*").remove();
      gLabels.selectAll("*").remove();

      const nodeData = DAG.nodes.map(d => ({{...d}}));
      const nodeMap  = {{}};
      nodeData.forEach(d => nodeMap[d.id]=d);

      const linkData = DAG.links.map(d => ({{...d}}));

      // token_count → radius
      const maxTok = d3.max(nodeData, d=>d.token_count)||1;
      const rScale = d3.scaleSqrt().domain([0,maxTok]).range([5,22]);

      // force sim
      const layerSpacing = H / (MAX_LVL + 3);
      sim = d3.forceSimulation(nodeData)
        .force("link", d3.forceLink(linkData).id(d=>d.id).distance(80).strength(d=>d.weight*0.25))
        .force("charge", d3.forceManyBody().strength(-180))
        .force("y", d3.forceY(d => (MAX_LVL - d.level + 1) * layerSpacing).strength(0.35))
        .force("x", d3.forceX(W/2).strength(0.04))
        .force("collision", d3.forceCollide(d => rScale(d.token_count)+4));

      // links
      const linkSel = gLinks.selectAll("line").data(linkData).enter().append("line")
        .attr("class","link")
        .attr("stroke","#475569")
        .attr("stroke-opacity", d => 0.15 + d.weight*0.7)
        .attr("stroke-width", d => 0.5 + d.weight*3);

      // nodes
      const nodeSel = gNodes.selectAll("g.node").data(nodeData).enter().append("g").attr("class","node");

      nodeSel.each(function(d){{
        const g = d3.select(this);
        const r = rScale(d.token_count);
        if(d.type==="leaf"){{
          g.append("circle").attr("r",r).attr("fill",lvlColor(d.level)).attr("stroke","#0f172a").attr("stroke-width",1.5);
        }} else {{
          // diamond for summary nodes
          const s = r*1.3;
          g.append("rect").attr("x",-s).attr("y",-s).attr("width",s*2).attr("height",s*2)
           .attr("transform","rotate(45)").attr("rx",3)
           .attr("fill",lvlColor(d.level)).attr("stroke","#0f172a").attr("stroke-width",1.5);
        }}
        // overlay glow
        if(overlaySet.has(d.id)){{
          g.append("circle").attr("r",r+6).attr("fill","none")
           .attr("stroke","#fbbf24").attr("stroke-width",3).attr("stroke-dasharray","4,2");
        }}
      }});

      nodeSel.style("cursor","pointer")
        .on("mouseover", (e,d) => showDetail(d))
        .on("mouseout", clearDetail)
        .call(d3.drag()
          .on("start",(e,d)=>{{if(!e.active)sim.alphaTarget(0.3).restart();d.fx=d.x;d.fy=d.y;}})
          .on("drag",(e,d)=>{{d.fx=e.x;d.fy=e.y;}})
          .on("end",(e,d)=>{{if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null;}}));

      // labels
      const labelSel = gLabels.selectAll("text").data(nodeData).enter().append("text")
        .attr("class","node-label").attr("text-anchor","middle").attr("dy","0.35em")
        .text(d => d.name.slice(0,30) + (d.name.length>30?"...":""));

      sim.on("tick",()=>{{
        linkSel.attr("x1",d=>d.source.x).attr("y1",d=>d.source.y)
               .attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);
        nodeSel.attr("transform",d=>`translate(${{d.x}},${{d.y}})`);
        labelSel.attr("x",d=>d.x).attr("y",d=>d.y - rScale(d.token_count) - 6);
      }});

      // store refs for filtering
      window._dagRefs = {{ nodeSel, linkSel, labelSel, nodeData, linkData, nodeMap }};
    }}

    /* ====== TREE VIEW ====== */
    function renderTree(){{
      gLinks.selectAll("*").remove();
      gNodes.selectAll("*").remove();
      gLabels.selectAll("*").remove();
      if(sim) sim.stop();

      const root = d3.hierarchy(TREE);
      const treeW = Math.max(W - 200, 600);
      const treeH = Math.max(root.descendants().length * 22, H - 100);
      const layout = d3.tree().size([treeH, treeW - 200]);
      layout(root);

      // links
      gLinks.selectAll("path").data(root.links()).enter().append("path")
        .attr("class","tree-link")
        .attr("d", d3.linkHorizontal().x(d=>d.y+100).y(d=>d.x));

      // nodes
      const nodeSel = gNodes.selectAll("g.tnode").data(root.descendants()).enter().append("g")
        .attr("class","tnode")
        .attr("transform", d => `translate(${{d.y+100}},${{d.x}})`);

      nodeSel.append("circle")
        .attr("r", d => d.data.type==="leaf" ? 5 : 8)
        .attr("fill", d => lvlColor(d.data.level||0))
        .attr("stroke","#0f172a").attr("stroke-width",1.5);

      // overlay glow
      nodeSel.filter(d => overlaySet.has(d.data.id)).append("circle")
        .attr("r",12).attr("fill","none").attr("stroke","#fbbf24").attr("stroke-width",2.5).attr("stroke-dasharray","4,2");

      // labels
      nodeSel.append("text")
        .attr("class","node-label")
        .attr("dx", d => d.children ? -12 : 12)
        .attr("dy","0.35em")
        .attr("text-anchor", d => d.children ? "end" : "start")
        .text(d => (d.data.name||"").slice(0,50) + ((d.data.name||"").length>50?"...":""));

      nodeSel.style("cursor","pointer")
        .on("mouseover",(e,d) => showDetail(d.data))
        .on("mouseout", clearDetail);

      // fit
      svg.call(zoom.transform, d3.zoomIdentity.translate(50,H/2 - treeH/2).scale(0.85));
    }}

    /* ====== DETAIL PANEL ====== */
    const detEl = document.getElementById("details");
    function showDetail(d){{
      const score = overlayScores[d.id];
      detEl.innerHTML = `
        <div class="row"><strong style="font-size:14px">${{d.name||d.id}}</strong></div>
        <div class="row"><strong>ID:</strong> <span class="mono">${{d.id}}</span></div>
        <div class="row"><strong>Level:</strong> ${{d.level}} &nbsp; <strong>Type:</strong> ${{d.type}}</div>
        <div class="row"><strong>Tokens:</strong> ${{d.token_count}}</div>
        ${{score !== undefined ? `<div class="row"><strong>Retrieval score:</strong> ${{score.toFixed(4)}}</div>` : ""}}
        ${{d.full_text ? `<div class="row" style="margin-top:8px;max-height:140px;overflow:auto"><span class="mono">${{d.full_text}}</span></div>` : ""}}
      `;
    }}
    function clearDetail(){{ detEl.innerHTML = "Hover over a node for details."; }}

    /* ====== STATS PANEL ====== */
    const statEl = document.getElementById("stats");
    function updateStats(){{
      const nLeaves = DAG.nodes.filter(n=>n.type==="leaf").length;
      const nSummaries = DAG.nodes.filter(n=>n.type==="summary").length;
      statEl.innerHTML = `
        <div class="row"><strong>Total nodes:</strong> ${{DAG.nodes.length}}</div>
        <div class="row"><strong>Leaves:</strong> ${{nLeaves}}</div>
        <div class="row"><strong>Summaries:</strong> ${{nSummaries}}</div>
        <div class="row"><strong>Edges:</strong> ${{DAG.links.length}}</div>
        <div class="row"><strong>Levels:</strong> ${{MAX_LVL+1}}</div>
        ${{overlaySet.size ? `<div class="row"><strong>Retrieved:</strong> ${{overlaySet.size}} nodes</div>` : ""}}
      `;
    }}
    updateStats();

    /* ====== VIEW SWITCHING ====== */
    document.getElementById("btnDAG").addEventListener("click", ()=>{{
      currentView="dag";
      document.getElementById("btnDAG").classList.add("active");
      document.getElementById("btnTree").classList.remove("active");
      renderDAG();
    }});
    document.getElementById("btnTree").addEventListener("click", ()=>{{
      currentView="tree";
      document.getElementById("btnTree").classList.add("active");
      document.getElementById("btnDAG").classList.remove("active");
      renderTree();
    }});

    /* ====== SEARCH FILTER ====== */
    document.getElementById("searchInput").addEventListener("input", e => {{
      const q = e.target.value.trim().toLowerCase();
      if(currentView !== "dag" || !window._dagRefs) return;
      const {{nodeSel, linkSel, labelSel, nodeData, linkData, nodeMap}} = window._dagRefs;
      if(!q){{
        nodeSel.attr("opacity",1); linkSel.attr("opacity",d=>0.15+d.weight*0.7); labelSel.attr("opacity",1);
        return;
      }}
      const match = new Set();
      nodeData.forEach(d => {{ if(d.name.toLowerCase().includes(q)) match.add(d.id); }});
      nodeSel.attr("opacity", d => match.has(d.id)?1:0.08);
      labelSel.attr("opacity", d => match.has(d.id)?1:0.05);
      linkSel.attr("opacity", d => {{
        const sid = typeof d.source==="object"?d.source.id:d.source;
        const tid = typeof d.target==="object"?d.target.id:d.target;
        return (match.has(sid)||match.has(tid)) ? 0.6 : 0.03;
      }});
    }});

    /* ====== WEIGHT SLIDER ====== */
    document.getElementById("weightSlider").addEventListener("input", e => {{
      const thresh = parseFloat(e.target.value);
      document.getElementById("weightVal").textContent = thresh.toFixed(2);
      if(currentView !== "dag" || !window._dagRefs) return;
      const {{linkSel}} = window._dagRefs;
      linkSel.attr("display", d => d.weight >= thresh ? null : "none");
    }});

    /* ====== LEVEL FILTER ====== */
    document.getElementById("levelFilter").addEventListener("change", e => {{
      const val = e.target.value;
      if(currentView !== "dag" || !window._dagRefs) return;
      const {{nodeSel, labelSel}} = window._dagRefs;
      if(val === "all"){{
        nodeSel.attr("opacity",1); labelSel.attr("opacity",1);
      }} else {{
        const lvl = parseInt(val);
        nodeSel.attr("opacity", d => d.level===lvl?1:0.1);
        labelSel.attr("opacity", d => d.level===lvl?1:0.05);
      }}
    }});

    /* ====== INITIAL RENDER ====== */
    renderDAG();

  }})();
  </script>
</body>
</html>
"""
