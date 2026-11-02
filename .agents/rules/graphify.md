---
trigger: always_on
description: Mandatory rule to consult the graphify knowledge graph before editing and auto-update after all code changes.
---

# Graphify Knowledge Graph Context & Auto-Update Protocol

This repository relies on the **Graphify Knowledge Graph** at `graphify-out/` as the single source of truth for architectural context, cross-module relationships, and system behavior.

## 🔴 Mandatory Directives for All AI Coding Models

1. **Mandatory Pre-Action Context Resolution**:
   - Before editing files or answering architecture questions, always consult the knowledge graph using:
     `python -m graphify query "<your question or feature goal>"`
     or inspect `graphify-out/GRAPH_REPORT.md` and `graphify-out/graph.json`.
   - Never guess module dependencies, God Nodes, or data flow paths — query the graph first.

2. **Mandatory Post-Edit Graph Synchronization**:
   - After completing any code modification, addition, or refactoring in the project, you **MUST** run:
     `python update_graph.py`
   - This keeps `graphify-out/graph.json`, `GRAPH_REPORT.md`, and the 3D WebGL visualizer (`graphify-out/graph.html`) fully updated so future models never lose context.
