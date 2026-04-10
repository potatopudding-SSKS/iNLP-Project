from kg2 import load_graph, extract_entities_from_question, find_node_by_id, bfs
from graph_tool.all import shortest_path
import itertools

# result = bfs(g, start_id="thyroxine", max_steps=2)

# for edge in result:
#     print(edge)

# def traverse_graph(graph, entities):

#     eset = set()
#     ep_type = graph.ep["relation"]
#     for i, j in itertools.combinations(entities, 2):
#         i_v = find_node_by_id(graph, i)
#         j_v = find_node_by_id(graph, j)
#         if i_v is None or j_v is None:
#             print(f"One of the entities '{i}' or '{j}' not found in the graph.")
#             continue
#         print(f"Finding paths between {i} and {j}...")
#         try:
#             vlist, elist = shortest_path(graph, graph.vertex(i_v), graph.vertex(j_v))
#             eset.update((int(e.source()),int(e.target()),str(ep_type[e])) for e in elist)
#         except Exception as e:
#             print(f"No path found between {i} and {j}: {e}")
#     return eset

import heapq
from graph_tool.all import shortest_path, GraphView


def yen_k_shortest_paths(graph, source, target, k):
    ep_type = graph.ep["relation"]

    if k <= 0:
        return []

    def path_cost(edges_uv):
        return len(edges_uv)

    def to_edge_pairs(vlist):
        return [(int(vlist[i]), int(vlist[i + 1])) for i in range(len(vlist) - 1)]

    # First shortest path
    vlist, elist = shortest_path(graph, source, target)
    if not elist:
        return []

    first_vlist = list(vlist)
    first_edges_uv = to_edge_pairs(first_vlist)

    A = [(first_vlist, first_edges_uv)]  # accepted paths
    B = []  # candidate paths (min-heap)
    accepted_keys = {tuple(first_edges_uv)}
    queued_keys = set()

    for k_i in range(1, k):
        last_vlist, last_edges_uv = A[-1]

        for i in range(len(last_vlist) - 1):
            spur_node = last_vlist[i]
            root_path_vertices = last_vlist[:i+1]
            root_path_edges_uv = last_edges_uv[:i]

            # 🚫 Block edges that would recreate same prefix
            blocked_edges = set()

            for vlist_p, edges_uv_p in A:
                if len(vlist_p) > i and vlist_p[:i+1] == root_path_vertices:
                    blocked_edges.add(edges_uv_p[i])

            # Create edge filter
            efilt = graph.new_edge_property("bool", val=True)
            for e in graph.edges():
                uv = (int(e.source()), int(e.target()))
                vu = (uv[1], uv[0])
                if uv in blocked_edges or vu in blocked_edges:
                    efilt[e] = False

            # Yen's algorithm removes root-path vertices except the spur node.
            vfilt = graph.new_vertex_property("bool", val=True)
            for rv in root_path_vertices[:-1]:
                vfilt[rv] = False

            gv = GraphView(graph, efilt=efilt, vfilt=vfilt)

            spur_vlist, spur_elist = shortest_path(
                gv,
                spur_node,
                target,
            )

            if not spur_elist:
                continue

            spur_vlist = list(spur_vlist)
            total_vlist = root_path_vertices[:-1] + spur_vlist
            total_edges_uv = root_path_edges_uv + to_edge_pairs(spur_vlist)
            path_key = tuple(total_edges_uv)

            if path_key in accepted_keys or path_key in queued_keys:
                continue

            cost = path_cost(total_edges_uv)

            heapq.heappush(B, (cost, total_vlist, total_edges_uv))
            queued_keys.add(path_key)

        if not B:
            break

        _, vlist_new, edges_uv_new = heapq.heappop(B)
        queued_keys.discard(tuple(edges_uv_new))
        A.append((vlist_new, edges_uv_new))
        accepted_keys.add(tuple(edges_uv_new))

    # Convert to your format
    paths = []
    for _, edges_uv in A:
        elist = []
        for u, v in edges_uv:
            e = graph.edge(graph.vertex(u), graph.vertex(v))
            if e is None:
                # Undirected graph fallback.
                e = graph.edge(graph.vertex(v), graph.vertex(u))
            if e is None:
                continue
            elist.append(e)

        path = [
            (int(e.source()), int(e.target()), str(ep_type[e]))
            for e in elist
        ]
        paths.append(path)

    return paths

def traverse_graph(graph, entities, k=3):

    if len(entities) < 2:
        return bfs(graph, start_id=entities[0], max_steps=2)

    eset = set()

    for i, j in itertools.combinations(entities, 2):
        i_v = find_node_by_id(graph, i)
        j_v = find_node_by_id(graph, j)

        if i_v is None or j_v is None:
            print(f"One of the entities '{i}' or '{j}' not found.")
            continue

        print(f"Finding top-{k} paths between {i} and {j}...")

        # try:
        paths = yen_k_shortest_paths(
            graph,
            i_v,
            j_v,
            k
        )

        for path in paths:
            eset.update(path)

        # except Exception as e:
        #     print(f"No path found between {i} and {j}: {e}")

    return eset

def create_json(eset, graph):
    structure = {
        "edges": {},
        "documents": {}
    }
    v_id = graph.vp["id"]
    v_type = graph.vp["type"]
    v_title = graph.vp["title"]
    v_abstract = graph.vp["abstract"]
    for source, target, relation in eset:
        src_v = graph.vertex(source)
        tgt_v = graph.vertex(target)
        source_name = v_id[src_v]
        target_name = v_id[tgt_v]
        if v_type[src_v] == "document" and source_name not in structure["documents"]:
            doc = {
                "title": v_title[src_v],
                "abstract": v_abstract[src_v]
            }
            structure["documents"][source_name] = doc
        if v_type[tgt_v] == "document" and target_name not in structure["documents"]:
            doc = {
                "title": v_title[tgt_v],
                "abstract": v_abstract[tgt_v]
            }
            structure["documents"][target_name] = doc
        if source_name not in structure["edges"]:
            structure["edges"][source_name] = set()
        structure["edges"][source_name].add((relation, target_name))

    for k in structure["edges"]:
        structure["edges"][k] = [
        {"relation": r, "connected_to": t}
            for (r, t) in structure["edges"][k]
        ]
    return structure

def process_question(question, graph):
    entities = extract_entities_from_question(question, top_k=5)
    eset = traverse_graph(graph, entities, k=5)
    structure = create_json(eset, graph)
    return structure

def main():

    g = load_graph("knowledge_graph/bioasq_kg.gt")

    # while True:
    #     q_type = input("""
    # Select a question type:
    # 1 - yes/no
    # 2 - factoid
    # 3 - list
    # 4 - summary
    # Enter choice number: """)
    #     if q_type not in ["1", "2", "3", "4"]:
    #         print("Invalid choice. Please enter a number between 1 and 4.")
    #         break
    #     question = input("Enter your question: ")
    #     entities = extract_entities_from_question(question, top_k=5)
    #     subgraph = traverse_graph(g, entities, max_steps=2)

    entities = ["mendelian disorder"]
    eset = traverse_graph(g, entities, k=5)
    structure = create_json(eset, g)
    print(structure)

if __name__ == "__main__":
    main()

