import sys
import json

def query_graph(term):
    with open("graphify-out/graph.json", encoding="utf-8") as f:
        data = json.load(f)
    
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    
    term_lower = term.lower()
    matched = [n for n in nodes if term_lower in n.get("id", "").lower() or term_lower in n.get("label", "").lower()]
    
    print(f"=== Graphify Query: '{term}' (Matches: {len(matched)}) ===")
    for n in matched[:5]:
        nid = n.get("id")
        lbl = n.get("label")
        comm = n.get("community_name", n.get("community", "?"))
        src = n.get("source_location", n.get("file", "unknown"))
        print(f"\nNode: {lbl} (ID: {nid})")
        print(f"  Community: {comm}")
        print(f"  Source: {src}")
        
        # In/out edges
        connected = [e for e in edges if e.get("source") == nid or e.get("target") == nid]
        print(f"  Edges ({len(connected)}):")
        for e in connected[:8]:
            direction = "->" if e.get("source") == nid else "<-"
            other = e.get("target") if e.get("source") == nid else e.get("source")
            rel = e.get("relation", e.get("type", "rel"))
            print(f"    {direction} [{rel}] {other}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query_graph(sys.argv[1])
    else:
        print("Usage: python scripts/graph_query.py <term>")
