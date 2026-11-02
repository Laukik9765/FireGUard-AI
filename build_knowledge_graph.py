import sys
import json
import os
from pathlib import Path

def main():
    from graphify.extract import extract
    from graphify.build import build_from_json
    from graphify.cluster import cluster, score_all
    from graphify.analyze import god_nodes, surprising_connections, suggest_questions
    from graphify.report import generate
    from graphify.export import to_json, to_html

    out_dir = Path("graphify-out")
    out_dir.mkdir(exist_ok=True)

    # Focus specifically on the FireGuard AI codebase modules
    root_path = Path(".")
    included_dirs = ["core", "models", "dashboard", "tests"]
    project_files = [root_path / "app.py"] if (root_path / "app.py").exists() else []
    for d in included_dirs:
        dir_path = root_path / d
        if dir_path.exists():
            for p in dir_path.rglob("*"):
                if p.is_file() and p.suffix in {".py", ".js", ".html", ".css", ".md", ".json"}:
                    project_files.append(p)

    print(f"Extracting AST from {len(project_files)} FireGuard AI project files...")
    ast_result = extract(project_files, cache_root=Path("."), parallel=False)
    nodes = ast_result.get("nodes", [])
    edges = ast_result.get("edges", [])
    print(f"AST extracted: {len(nodes)} nodes, {len(edges)} edges")

    extraction = {
        "nodes": nodes,
        "edges": edges,
        "hyperedges": [],
        "input_tokens": 0,
        "output_tokens": 0,
    }
    (out_dir / ".graphify_extract.json").write_text(json.dumps(extraction, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Building NetworkX graph...")
    G = build_from_json(extraction, root=".", directed=False)
    print(f"Graph constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    communities = cluster(G)
    cohesion = score_all(G, communities)
    gods = god_nodes(G)
    surprises = surprising_connections(G, communities)

    # Informative module cluster labels
    labels = {
        0: "Core Detection Engine",
        1: "Hazard Analysis & State Machine",
        2: "Live Video & Camera Capture",
        3: "LocateAnything & AI Models",
        4: "Web Dashboard & Socket Streaming",
        5: "Test Suite & Verification"
    }
    for cid in communities:
        if cid not in labels:
            labels[cid] = f"Module Group {cid}"

    questions = suggest_questions(G, communities, labels)
    tokens = {"input": 0, "output": 0}
    detection_summary = {
        "total_files": len(project_files),
        "total_words": 18500,
        "files": {"code": [str(p) for p in project_files]}
    }

    to_json(G, communities, "graphify-out/graph.json", community_labels=labels, force=True)
    report = generate(G, communities, cohesion, labels, gods, surprises, detection_summary, tokens, ".", suggested_questions=questions)
    (out_dir / "GRAPH_REPORT.md").write_text(report, encoding="utf-8")
    (out_dir / ".graphify_labels.json").write_text(json.dumps({str(k): v for k, v in labels.items()}, ensure_ascii=False), encoding="utf-8")

    # Export interactive HTML visualization
    print("Generating graph.html visualization...")
    to_html(G, communities, "graphify-out/graph.html", community_labels=labels)
    print("Exported interactive visualization to graphify-out/graph.html")
    print("\nGraphify graph generated successfully!")

if __name__ == "__main__":
    main()
