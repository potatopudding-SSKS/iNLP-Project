from kg2 import load_graph
from collections import Counter

def save_entities(graph, output_file):
    v_type = graph.vp["type"]
    v_id = graph.vp["id"]
    with open(output_file, 'w') as f:
        for node in graph.vertices():
            if v_type[node] == "entity":
                f.write(f"{v_id[node]}\n")

def save_edge_types(graph, output_file):
    e_type = graph.ep["relation"]
    etypes = Counter()
    for edge in graph.edges():
        etypes[e_type[edge]] += 1
    with open(output_file, 'w') as f:
        for etype, count in etypes.items():
            f.write(f"{etype}\t{count}\n")

if __name__ == "__main__":
    graph = load_graph("knowledge_graph/bioasq_kg.gt")
    save_entities(graph, "entities.txt")
    save_edge_types(graph, "edge_types.txt")