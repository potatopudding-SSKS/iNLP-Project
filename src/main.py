from kg2 import load_graph, bfs

g = load_graph("knowledge_graph/bioasq_kg.gt")
result = bfs(g, start_id="thyroxine", max_steps=2)

for edge in result:
    print(edge)