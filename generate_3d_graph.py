import json
from pathlib import Path

def build_3d_html():
    graph_json_path = Path("graphify-out/graph.json")
    if not graph_json_path.exists():
        print("graph.json not found!")
        return

    raw_data = json.loads(graph_json_path.read_text(encoding="utf-8"))
    nodes = raw_data.get("nodes", [])
    links = raw_data.get("links", [])

    # Calculate degree for node sizing
    degree_map = {}
    for link in links:
        s = link.get("source")
        t = link.get("target")
        degree_map[s] = degree_map.get(s, 0) + 1
        degree_map[t] = degree_map.get(t, 0) + 1

    for n in nodes:
        nid = n.get("id")
        deg = degree_map.get(nid, 0)
        n["degree"] = deg
        # Node radius based on connections - larger for God Nodes
        n["val"] = max(3.0, min(18.0, 3.0 + (deg ** 0.65) * 2.2))
        n["is_god"] = deg >= 15

    graph_payload = json.dumps({"nodes": nodes, "links": links}, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>FireGuard AI — 3D Cyber Knowledge Graph</title>
  <!-- 3D Force Graph & Three.js -->
  <script src="https://unpkg.com/three@0.160.0/build/three.min.js"></script>
  <script src="https://unpkg.com/3d-force-graph@1.73.4/dist/3d-force-graph.min.js"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    :root {{
      --bg-dark: #050711;
      --panel-bg: rgba(10, 15, 29, 0.82);
      --panel-border: rgba(0, 245, 255, 0.22);
      --panel-border-glow: rgba(0, 245, 255, 0.12);
      --accent-cyan: #00f5ff;
      --accent-fire: #ff3d00;
      --accent-green: #00ff88;
      --accent-purple: #b5179e;
      --accent-pink: #ff007f;
      --accent-gold: #ffbe0b;
      --text-main: #f3f7fd;
      --text-muted: #8ea3c7;
      --font-ui: 'Space Grotesk', -apple-system, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }}
    body {{
      background: radial-gradient(circle at 50% 50%, #0d1226 0%, #050711 75%, #020308 100%);
      color: var(--text-main);
      font-family: var(--font-ui);
      overflow: hidden;
      height: 100vh;
      width: 100vw;
    }}
    #3d-graph {{
      position: absolute;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      z-index: 1;
    }}

    /* Futuristic Header Glass HUD */
    .top-hud {{
      position: absolute;
      top: 20px;
      left: 24px;
      z-index: 10;
      display: flex;
      align-items: center;
      gap: 18px;
      background: var(--panel-bg);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      border: 1px solid var(--panel-border);
      border-radius: 14px;
      padding: 12px 22px;
      box-shadow: 0 10px 35px rgba(0, 0, 0, 0.6), 0 0 20px var(--panel-border-glow);
    }}
    .brand-title {{
      font-weight: 700;
      font-size: 17px;
      letter-spacing: 0.05em;
      display: flex;
      align-items: center;
      gap: 10px;
      background: linear-gradient(135deg, #ffffff 30%, #00f5ff 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      text-transform: uppercase;
    }}
    .brand-tag {{
      font-family: var(--font-mono);
      font-size: 10px;
      padding: 3px 8px;
      border-radius: 6px;
      background: linear-gradient(135deg, rgba(0, 245, 255, 0.2), rgba(255, 0, 127, 0.2));
      color: var(--accent-cyan);
      border: 1px solid var(--accent-cyan);
      box-shadow: 0 0 10px rgba(0, 245, 255, 0.3);
      font-weight: 700;
      letter-spacing: 0.08em;
      -webkit-text-fill-color: var(--accent-cyan);
    }}
    .stats-badge {{
      font-size: 12px;
      color: var(--text-muted);
      font-family: var(--font-mono);
    }}
    .stats-badge b {{
      color: #00f5ff;
      text-shadow: 0 0 8px rgba(0, 245, 255, 0.5);
    }}

    /* Controls Bar */
    .controls-bar {{
      position: absolute;
      bottom: 24px;
      left: 24px;
      z-index: 10;
      display: flex;
      gap: 10px;
      background: var(--panel-bg);
      backdrop-filter: blur(20px);
      border: 1px solid var(--panel-border);
      border-radius: 12px;
      padding: 8px;
      box-shadow: 0 10px 35px rgba(0,0,0,0.6), 0 0 15px var(--panel-border-glow);
    }}
    .btn {{
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.12);
      color: var(--text-main);
      padding: 8px 16px;
      border-radius: 8px;
      font-size: 12px;
      font-weight: 600;
      font-family: var(--font-ui);
      cursor: pointer;
      transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
      display: inline-flex;
      align-items: center;
      gap: 7px;
      letter-spacing: 0.03em;
    }}
    .btn:hover {{
      background: linear-gradient(135deg, rgba(0, 245, 255, 0.25), rgba(255, 0, 127, 0.25));
      border-color: var(--accent-cyan);
      box-shadow: 0 0 12px rgba(0, 245, 255, 0.4);
      transform: translateY(-1px);
    }}
    .btn.active {{
      background: linear-gradient(135deg, #00f5ff, #ff007f);
      border-color: #fff;
      color: #050711;
      font-weight: 700;
      box-shadow: 0 0 16px rgba(0, 245, 255, 0.6);
    }}

    /* Futuristic Floating Glass Sidebar */
    .sidebar {{
      position: absolute;
      top: 20px;
      right: 24px;
      bottom: 24px;
      width: 360px;
      z-index: 10;
      background: var(--panel-bg);
      backdrop-filter: blur(22px);
      -webkit-backdrop-filter: blur(22px);
      border: 1px solid var(--panel-border);
      border-radius: 18px;
      display: flex;
      flex-direction: column;
      box-shadow: 0 16px 45px rgba(0, 0, 0, 0.7), 0 0 25px var(--panel-border-glow);
      overflow: hidden;
    }}
    .sidebar-header {{
      padding: 18px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      background: linear-gradient(180deg, rgba(0, 245, 255, 0.06) 0%, transparent 100%);
    }}
    .search-input {{
      width: 100%;
      background: rgba(4, 7, 18, 0.7);
      border: 1px solid rgba(0, 245, 255, 0.3);
      border-radius: 10px;
      padding: 10px 14px;
      color: #fff;
      font-size: 13px;
      font-family: var(--font-ui);
      outline: none;
      transition: all 0.25s;
    }}
    .search-input:focus {{
      border-color: var(--accent-cyan);
      box-shadow: 0 0 15px rgba(0, 245, 255, 0.4);
      background: rgba(4, 7, 18, 0.95);
    }}
    .search-results {{
      max-height: 160px;
      overflow-y: auto;
      background: rgba(5, 8, 20, 0.9);
      border-bottom: 1px solid var(--panel-border);
      display: none;
    }}
    .search-item {{
      padding: 9px 16px;
      font-size: 12px;
      cursor: pointer;
      border-bottom: 1px solid rgba(255,255,255,0.04);
      display: flex;
      align-items: center;
      justify-content: space-between;
      transition: background 0.15s;
    }}
    .search-item:hover {{
      background: linear-gradient(90deg, rgba(0, 245, 255, 0.2), rgba(255, 0, 127, 0.15));
    }}

    /* Detail Inspector */
    .inspector {{
      padding: 18px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      background: rgba(0, 0, 0, 0.25);
    }}
    .inspector-title {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--accent-cyan);
      margin-bottom: 8px;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .inspector-title::before {{
      content: '';
      display: inline-block;
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--accent-cyan);
      box-shadow: 0 0 8px var(--accent-cyan);
    }}
    .node-name {{
      font-size: 18px;
      font-weight: 700;
      color: #fff;
      word-break: break-word;
      margin-bottom: 6px;
      font-family: var(--font-mono);
      text-shadow: 0 0 12px rgba(255, 255, 255, 0.3);
    }}
    .node-meta {{
      font-size: 12px;
      color: var(--text-muted);
      line-height: 1.6;
    }}
    .community-tag {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 700;
      margin-top: 8px;
      letter-spacing: 0.03em;
    }}
    .neighbors-box {{
      margin-top: 12px;
      max-height: 120px;
      overflow-y: auto;
    }}
    .neighbor-pill {{
      display: block;
      padding: 5px 10px;
      margin: 4px 0;
      border-radius: 6px;
      background: rgba(255, 255, 255, 0.06);
      font-size: 11px;
      font-family: var(--font-mono);
      cursor: pointer;
      color: var(--text-main);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      border: 1px solid rgba(255, 255, 255, 0.05);
      transition: all 0.15s;
    }}
    .neighbor-pill:hover {{
      background: linear-gradient(90deg, rgba(0, 245, 255, 0.25), transparent);
      border-color: var(--accent-cyan);
      color: #00f5ff;
      transform: translateX(4px);
    }}

    /* Communities Legend */
    .legend-container {{
      flex: 1;
      overflow-y: auto;
      padding: 18px;
    }}
    .legend-title {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--text-muted);
      margin-bottom: 12px;
      font-weight: 700;
      display: flex;
      justify-content: space-between;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 7px 10px;
      border-radius: 8px;
      font-size: 12px;
      cursor: pointer;
      transition: all 0.2s ease;
      margin-bottom: 3px;
      border: 1px solid transparent;
    }}
    .legend-item:hover {{
      background: rgba(255, 255, 255, 0.08);
      border-color: rgba(255, 255, 255, 0.15);
      transform: translateX(3px);
    }}
    .legend-item.active {{
      background: rgba(0, 245, 255, 0.15);
      border-color: var(--accent-cyan);
    }}
    .legend-dot {{
      width: 11px;
      height: 11px;
      border-radius: 50%;
      flex-shrink: 0;
    }}
    .legend-name {{
      flex: 1;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      font-weight: 500;
    }}
    .legend-count {{
      color: var(--text-muted);
      font-size: 11px;
      font-family: var(--font-mono);
      background: rgba(255, 255, 255, 0.07);
      padding: 2px 6px;
      border-radius: 4px;
    }}

    /* Custom Scrollbars */
    ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
    ::-webkit-scrollbar-track {{ background: rgba(0, 0, 0, 0.2); }}
    ::-webkit-scrollbar-thumb {{ background: rgba(0, 245, 255, 0.3); border-radius: 4px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: var(--accent-cyan); }}
  </style>
</head>
<body>

  <!-- Top Info HUD -->
  <div class="top-hud">
    <div class="brand-title">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ff3d00" stroke-width="2.5"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/><path d="M12 6c0 0-3 4-3 7a3 3 0 0 0 6 0c0-3-3-7-3-7z" fill="#ff3d00"/></svg>
      FireGuard AI
      <span class="brand-tag">3D CYBER GRAPH</span>
    </div>
    <div class="stats-badge" id="stats-hud">
      Nodes: <b id="stat-nodes">389</b> · Edges: <b id="stat-edges">721</b> · Clusters: <b id="stat-clusters">6</b>
    </div>
  </div>

  <!-- Bottom Action Bar -->
  <div class="controls-bar">
    <button class="btn" id="btn-rotate" title="Toggle Auto-Rotation">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/></svg>
      Auto-Rotate: OFF
    </button>
    <button class="btn active" id="btn-particles" title="Toggle Animated Particles">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="2"/><circle cx="6" cy="6" r="1.5"/><circle cx="18" cy="18" r="1.5"/></svg>
      Particles: ON
    </button>
    <button class="btn" id="btn-reset-cam" title="Reset Camera View">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>
      Reset View
    </button>
  </div>

  <!-- 3D Canvas -->
  <div id="3d-graph"></div>

  <!-- Sidebar -->
  <aside class="sidebar">
    <div class="sidebar-header">
      <input type="text" class="search-input" id="search-box" placeholder="⚡ Search classes, functions, modules..." />
    </div>
    <div class="search-results" id="search-results"></div>

    <!-- Inspector -->
    <div class="inspector" id="inspector-panel">
      <div class="inspector-title">Active Node Inspector</div>
      <div class="node-name" id="insp-name">Click any 3D node</div>
      <div class="node-meta" id="insp-meta">Select a glowing node or use search to explore connections, source code location, and architectural cluster relations.</div>
      <div id="insp-badge" style="display:none;"></div>
      <div class="neighbors-box" id="insp-neighbors"></div>
    </div>

    <!-- Community Clusters -->
    <div class="legend-container">
      <div class="legend-title">
        <span>Architecture Clusters</span>
        <span style="font-size:10px; color:var(--accent-cyan); font-weight:normal;">Click to Focus</span>
      </div>
      <div id="legend-list"></div>
    </div>
  </aside>

  <!-- Raw Graph Payload -->
  <script id="graph-data" type="application/json">
    {graph_payload}
  </script>

  <script>
    "use strict";

    // Vibrant Electric Neon Palette
    const PALETTE = [
      "#00f5ff", // 0: Electric Cyan (Core Detection Engine)
      "#ff3d00", // 1: Solar Flame Orange (Hazard Analysis & State Machine)
      "#00ff88", // 2: Neon Emerald (Live Video & Camera Capture)
      "#b5179e", // 3: Plasma Purple (LocateAnything & AI Models)
      "#ff007f", // 4: Cyber Magenta (Web Dashboard & Socket Streaming)
      "#ffbe0b", // 5: Cyber Gold (Test Suite & Verification)
      "#7209b7", // 6: Deep Neon Indigo
      "#4cc9f0", // 7: Laser Azure
      "#f72585", // 8: Hot Pink
      "#3a0ca3", // 9: Royal Velvet
      "#fb5607", // 10: Orange Spark
      "#4361ee", // 11: Electric Cobalt
    ];

    const rawData = JSON.parse(document.getElementById("graph-data").textContent);
    const container = document.getElementById("3d-graph");

    // UI Elements
    const searchBox = document.getElementById("search-box");
    const searchResults = document.getElementById("search-results");
    const inspName = document.getElementById("insp-name");
    const inspMeta = document.getElementById("insp-meta");
    const inspBadge = document.getElementById("insp-badge");
    const inspNeighbors = document.getElementById("insp-neighbors");
    const legendList = document.getElementById("legend-list");

    const btnRotate = document.getElementById("btn-rotate");
    const btnParticles = document.getElementById("btn-particles");
    const btnResetCam = document.getElementById("btn-reset-cam");

    let isRotating = false;
    let particlesOn = true;
    let highlightNodes = new Set();
    let highlightLinks = new Set();
    let selectedNode = null;
    let focusedCommunity = null;

    // Cross-link nodes for fast neighbor lookup
    const neighborMap = new Map();
    rawData.links.forEach(link => {{
      const s = typeof link.source === 'object' ? link.source.id : link.source;
      const t = typeof link.target === 'object' ? link.target.id : link.target;
      if (!neighborMap.has(s)) neighborMap.set(s, []);
      if (!neighborMap.has(t)) neighborMap.set(t, []);
      neighborMap.get(s).push({{ id: t, link }});
      neighborMap.get(t).push({{ id: s, link }});
    }});

    // Build Community Counts
    const communityMap = new Map();
    rawData.nodes.forEach(n => {{
      const cId = n.community != null ? n.community : 0;
      const cName = n.community_name || `Cluster ${{cId}}`;
      if (!communityMap.has(cId)) {{
        communityMap.set(cId, {{ name: cName, count: 0, color: PALETTE[cId % PALETTE.length] }});
      }}
      communityMap.get(cId).count++;
    }});

    // Render Legend
    Array.from(communityMap.entries()).sort((a,b) => b[1].count - a[1].count).forEach(([cId, info]) => {{
      const item = document.createElement("div");
      item.className = "legend-item";
      item.dataset.cid = cId;
      item.innerHTML = `
        <span class="legend-dot" style="background:${{info.color}}; box-shadow: 0 0 10px ${{info.color}}"></span>
        <span class="legend-name">${{info.name}}</span>
        <span class="legend-count">${{info.count}}</span>
      `;
      item.addEventListener("click", () => {{
        focusOnCommunity(cId, item);
      }});
      legendList.appendChild(item);
    }});

    // Init 3D Force Graph
    const Graph = ForceGraph3D()
      (container)
      .graphData(rawData)
      .backgroundColor("rgba(5, 7, 17, 0)")
      .showNavInfo(false)
      .nodeId('id')
      .nodeLabel(n => `${{n.label || n.id}} [${{n.community_name || 'Cluster ' + n.community}}]`)
      .nodeVal(n => n.val || 3)
      .nodeColor(n => {{
        const baseColor = PALETTE[(n.community || 0) % PALETTE.length];
        if (focusedCommunity !== null) {{
          return n.community === focusedCommunity ? baseColor : "rgba(255,255,255,0.06)";
        }}
        if (highlightNodes.size > 0) {{
          return highlightNodes.has(n.id) ? baseColor : "rgba(255,255,255,0.08)";
        }}
        return baseColor;
      }})
      .nodeResolution(20)
      .linkSource('source')
      .linkTarget('target')
      .linkColor(l => {{
        if (focusedCommunity !== null) {{
          const s = typeof l.source === 'object' ? l.source : rawData.nodes.find(n => n.id === l.source);
          const t = typeof l.target === 'object' ? l.target : rawData.nodes.find(n => n.id === l.target);
          if (s && t && s.community === focusedCommunity && t.community === focusedCommunity) {{
            return PALETTE[focusedCommunity % PALETTE.length];
          }}
          return "rgba(255,255,255,0.02)";
        }}
        if (highlightLinks.size > 0) {{
          return highlightLinks.has(l) ? "#00f5ff" : "rgba(255,255,255,0.03)";
        }}
        return "rgba(0, 245, 255, 0.22)";
      }})
      .linkWidth(l => highlightLinks.has(l) ? 3.0 : 1.0)
      .linkDirectionalParticles(l => particlesOn ? (highlightLinks.size > 0 ? (highlightLinks.has(l) ? 5 : 0) : 2) : 0)
      .linkDirectionalParticleWidth(2.2)
      .linkDirectionalParticleSpeed(0.007)
      .linkDirectionalParticleColor(() => "#00f5ff")
      .onNodeClick(node => {{
        selectNode(node);
      }})
      .onBackgroundClick(() => {{
        clearSelection();
      }});

    // Tune 3D Physics
    Graph.d3Force('charge').strength(-140);
    Graph.d3Force('link').distance(45);

    // Node Selection & Focus
    function selectNode(node) {{
      selectedNode = node;
      focusedCommunity = null;
      document.querySelectorAll(".legend-item").forEach(el => el.classList.remove("active"));
      highlightNodes.clear();
      highlightLinks.clear();

      if (node) {{
        highlightNodes.add(node.id);
        const neighbors = neighborMap.get(node.id) || [];
        neighbors.forEach(n => {{
          highlightNodes.add(n.id);
          highlightLinks.add(n.link);
        }});

        // Smooth 3D Camera flight
        const distance = 90;
        const distRatio = 1 + distance/Math.hypot(node.x, node.y, node.z);
        Graph.cameraPosition(
          {{ x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio }},
          node,
          1400
        );

        // Update Inspector
        inspName.textContent = node.label || node.id;
        const fileLoc = node.source_file ? `${{node.source_file}} ${{node.source_location || ''}}` : "Internal module";
        inspMeta.innerHTML = `
          <div style="color:var(--text-main); margin-bottom:4px;"><b>Type:</b> ${{node.file_type || 'symbol'}} · <b>Origin:</b> ${{node._origin || 'ast'}}</div>
          <div style="font-family:var(--font-mono); font-size:11px; color:var(--accent-cyan); word-break:break-all;"><b>Location:</b> ${{fileLoc}}</div>
          <div style="margin-top:4px;"><b>Connections:</b> <span style="color:#00ff88;font-weight:700;">${{neighbors.length}}</span> edges</div>
        `;

        const cColor = PALETTE[(node.community || 0) % PALETTE.length];
        inspBadge.style.display = "inline-block";
        inspBadge.className = "community-tag";
        inspBadge.style.borderColor = cColor;
        inspBadge.style.color = "#fff";
        inspBadge.style.background = cColor + "33";
        inspBadge.style.boxShadow = `0 0 12px ${{cColor}}66`;
        inspBadge.textContent = node.community_name || `Cluster ${{node.community}}`;

        // Render Neighbors
        inspNeighbors.innerHTML = '<div style="font-size:10px; text-transform:uppercase; color:var(--text-muted); margin: 8px 0 4px 0;">Connected Neighbors:</div>';
        neighbors.forEach(n => {{
          const nNode = rawData.nodes.find(item => item.id === n.id);
          const pill = document.createElement("div");
          pill.className = "neighbor-pill";
          pill.textContent = nNode ? nNode.label || nNode.id : n.id;
          pill.addEventListener("click", () => {{
            if (nNode) selectNode(nNode);
          }});
          inspNeighbors.appendChild(pill);
        }});
      }}

      updateHighlight();
    }}

    function clearSelection() {{
      selectedNode = null;
      focusedCommunity = null;
      document.querySelectorAll(".legend-item").forEach(el => el.classList.remove("active"));
      highlightNodes.clear();
      highlightLinks.clear();
      inspName.textContent = "Click any 3D node";
      inspMeta.textContent = "Select a glowing node or use search to explore connections, source code location, and architectural cluster relations.";
      inspBadge.style.display = "none";
      inspNeighbors.innerHTML = "";
      updateHighlight();
    }}

    function updateHighlight() {{
      Graph.nodeColor(Graph.nodeColor())
           .linkColor(Graph.linkColor())
           .linkWidth(Graph.linkWidth())
           .linkDirectionalParticles(Graph.linkDirectionalParticles());
    }}

    function focusOnCommunity(cId, itemElement) {{
      if (focusedCommunity === cId) {{
        clearSelection();
        return;
      }}
      focusedCommunity = cId;
      document.querySelectorAll(".legend-item").forEach(el => el.classList.remove("active"));
      if (itemElement) itemElement.classList.add("active");

      const cNodes = rawData.nodes.filter(n => n.community === cId && n.x != null);
      if (cNodes.length === 0) return;
      let avgX = 0, avgY = 0, avgZ = 0;
      cNodes.forEach(n => {{ avgX += n.x; avgY += n.y; avgZ += n.z; }});
      avgX /= cNodes.length; avgY /= cNodes.length; avgZ /= cNodes.length;
      Graph.cameraPosition({{ x: avgX * 1.7, y: avgY * 1.7, z: avgZ * 1.7 + 140 }}, {{ x: avgX, y: avgY, z: avgZ }}, 1500);
      updateHighlight();
    }}

    // Search Box
    searchBox.addEventListener("input", (e) => {{
      const query = e.target.value.trim().toLowerCase();
      if (!query) {{
        searchResults.style.display = "none";
        searchResults.innerHTML = "";
        return;
      }}
      const matches = rawData.nodes.filter(n => {{
        const lbl = (n.label || "").toLowerCase();
        const id = (n.id || "").toLowerCase();
        const file = (n.source_file || "").toLowerCase();
        return lbl.includes(query) || id.includes(query) || file.includes(query);
      }}).slice(0, 10);

      if (matches.length > 0) {{
        searchResults.style.display = "block";
        searchResults.innerHTML = "";
        matches.forEach(m => {{
          const item = document.createElement("div");
          item.className = "search-item";
          item.innerHTML = `
            <span><b>${{m.label || m.id}}</b></span>
            <span style="font-size:10px; color:var(--accent-cyan)">${{m.community_name || ''}}</span>
          `;
          item.addEventListener("click", () => {{
            selectNode(m);
            searchResults.style.display = "none";
            searchBox.value = m.label || m.id;
          }});
          searchResults.appendChild(item);
        }});
      }} else {{
        searchResults.style.display = "none";
      }}
    }});

    // Controls
    btnRotate.addEventListener("click", () => {{
      isRotating = !isRotating;
      btnRotate.textContent = isRotating ? "Auto-Rotate: ON" : "Auto-Rotate: OFF";
      btnRotate.classList.toggle("active", isRotating);
    }});

    btnParticles.addEventListener("click", () => {{
      particlesOn = !particlesOn;
      btnParticles.textContent = particlesOn ? "Particles: ON" : "Particles: OFF";
      btnParticles.classList.toggle("active", particlesOn);
      updateHighlight();
    }});

    btnResetCam.addEventListener("click", () => {{
      clearSelection();
      Graph.cameraPosition({{ x: 0, y: 0, z: 290 }}, {{ x: 0, y: 0, z: 0 }}, 1300);
    }});

    // Auto-rotation loop
    let angle = 0;
    setInterval(() => {{
      if (isRotating) {{
        angle += Math.PI / 700;
        const distance = 270;
        Graph.cameraPosition({{
          x: distance * Math.sin(angle),
          z: distance * Math.cos(angle)
        }});
      }}
    }}, 25);

    // Initial Camera Intro
    setTimeout(() => {{
      Graph.cameraPosition({{ x: 0, y: 20, z: 290 }}, {{ x: 0, y: 0, z: 0 }}, 1500);
    }}, 300);

  </script>
</body>
</html>"""

    Path("graphify-out/graph.html").write_text(html, encoding="utf-8")
    print("Vibrant 3D Interactive Graph generated successfully!")

if __name__ == "__main__":
    build_3d_html()
