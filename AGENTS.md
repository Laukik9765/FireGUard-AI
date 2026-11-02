# FireGuard AI — Agent Architecture & Context Guidelines

> **Mandatory Rule for All AI Coding Models & Agents**:
> 1. Always consult the **Graphify Knowledge Graph** (`graphify-out/graph.json` and `graphify-out/GRAPH_REPORT.md`) or run `python -m graphify query "<question>"` before proposing architectural assumptions or editing core components.
> 2. After making any modifications, additions, or refactors, execute `python update_graph.py` (or `python -m graphify update .`) to keep the knowledge graph and 3D visualizer synchronized.

---

## 🗺️ Core System Components

- **Server Entrypoint**: [`app.py`](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/app.py) (Flask + SocketIO, `/stream`, `/graph`, `/api/livestock/verify-all`)
- **Detector**: [`models/locate_anything/mock_detector.py`](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/models/locate_anything/mock_detector.py) (Precision 5-gate fire detection & 4-gate smoke detection with flicker validation)
- **Hazard Scoring**: [`models/hazard_level/hazard_engine.py`](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/models/hazard_level/hazard_engine.py)
- **Web UI & Aesthetics**: [`dashboard/templates/index.html`](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/dashboard/templates/index.html), [`dashboard/static/js/dashboard.js`](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/dashboard/static/js/dashboard.js), [`dashboard/static/css/style.css`](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/dashboard/static/css/style.css)
- **Interactive 3D Graph**: [`graphify-out/graph.html`](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/graphify-out/graph.html)

---

## 🛠️ Graphify Commands

```bash
# Query the codebase:
python -m graphify query "<question>"
python -m graphify path "<nodeA>" "<nodeB>"

# Update after code changes:
python update_graph.py
```
