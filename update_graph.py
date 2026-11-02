"""
FireGuard AI — Graph Update Script
Runs incremental graphify extraction and rebuilds the 3D WebGL knowledge graph.
Usage: python update_graph.py
"""
import subprocess
import sys
from pathlib import Path

def main():
    print("[*] Updating FireGuard AI Graphify Knowledge Graph...")
    # 1. Update graphify AST & graph
    subprocess.run([sys.executable, "-m", "graphify", "update", "."], capture_output=False)
    
    # 2. Rebuild 3D WebGL interactive visualization
    try:
        from generate_3d_graph import build_3d_html
        build_3d_html()
        print("[+] 3D Interactive Graph successfully synced to graphify-out/graph.html")
    except Exception as e:
        print(f"[-] Error rebuilding 3D graph: {e}")

if __name__ == "__main__":
    main()
