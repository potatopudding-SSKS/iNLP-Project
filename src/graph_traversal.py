from kg2 import load_graph, extract_entities_from_question, find_node_by_id
from graph_tool.all import shortest_path, GraphView
import itertools
import heapq
import math
import re
from collections import defaultdict


_RELATION_COUNT_CACHE = None
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "for", "on", "with",
    "is", "are", "was", "were", "be", "by", "as", "from", "that", "this",
    "which", "what", "who", "when", "where", "how", "does", "do", "did",
    "can", "could", "should", "would", "it", "its",
}


def _load_relation_counts(path="edge_types.txt"):
    global _RELATION_COUNT_CACHE
    if _RELATION_COUNT_CACHE is not None:
        return _RELATION_COUNT_CACHE

    counts = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 2:
                    continue
                rel = parts[0].strip().lower()
                try:
                    counts[rel] = int(parts[1])
                except ValueError:
                    continue
    except FileNotFoundError:
        counts = {}

    _RELATION_COUNT_CACHE = counts
    return counts


def _question_terms(question):
    tokens = re.findall(r"[a-z0-9\-]+", question.lower())
    return {t for t in tokens if len(t) >= 3 and t not in _STOPWORDS}


def _edge_from_pair(graph, u, v):
    e = graph.edge(graph.vertex(u), graph.vertex(v))
    if e is None:
        e = graph.edge(graph.vertex(v), graph.vertex(u))
    return e


def _to_edge_pairs(vlist):
    return [(int(vlist[i]), int(vlist[i + 1])) for i in range(len(vlist) - 1)]


def _to_vertex_list(edges_uv):
    if not edges_uv:
        return []
    vlist = [edges_uv[0][0]]
    for _, v in edges_uv:
        vlist.append(v)
    return vlist


def _build_edge_cost_property(graph):
    relation_counts = _load_relation_counts()
    ep_type = graph.ep["relation"]
    vp_type = graph.vp["type"]
    weight = graph.new_edge_property("double")

    max_count = max(relation_counts.values()) if relation_counts else 1
    log_max = math.log1p(max_count)

    for e in graph.edges():
        rel = str(ep_type[e]).strip().lower()
        count = relation_counts.get(rel, 1)

        src_idx = int(e.source())
        tgt_idx = int(e.target())
        src_v = graph.vertex(src_idx)
        tgt_v = graph.vertex(tgt_idx)

        src_deg = src_v.out_degree()
        tgt_deg = tgt_v.out_degree()

        if rel == "related-to":
            # related-to is expected to be high-frequency by design.
            # Treat it as important and only use document hubness to separate signal from noise.
            doc_deg = None
            if str(vp_type[src_v]) == "document":
                doc_deg = src_deg
            elif str(vp_type[tgt_v]) == "document":
                doc_deg = tgt_deg

            if doc_deg is None:
                doc_deg = max(src_deg, tgt_deg)

            weight[e] = max(0.25, 0.75 + 0.22 * math.log1p(doc_deg))
            continue

        rel_cost = 1.0 + (math.log1p(count) / max(1e-9, log_max))
        if count <= 10:
            rel_cost -= 0.2

        hub_penalty = 0.12 * (math.log1p(src_deg) + math.log1p(tgt_deg))
        weight[e] = max(0.25, rel_cost + hub_penalty)

    return weight


def _path_cost_from_pairs(graph, edge_cost, edges_uv):
    cost = 0.0
    for u, v in edges_uv:
        e = _edge_from_pair(graph, u, v)
        if e is None:
            cost += 1000.0
        else:
            cost += float(edge_cost[e])
    return cost


def _document_overlap_score(graph, doc_vertex_idx, question_tokens):
    if not question_tokens:
        return 0.0

    v = graph.vertex(doc_vertex_idx)
    v_title = graph.vp["title"]
    v_abstract = graph.vp["abstract"]
    text_parts = [str(v_title[v]), str(v_abstract[v])]
    if "mesh_terms" in graph.vp:
        text_parts.extend(str(x) for x in list(graph.vp["mesh_terms"][v]))

    doc_tokens = set(re.findall(r"[a-z0-9\-]+", " ".join(text_parts).lower()))
    if not doc_tokens:
        return 0.0

    overlap = len(question_tokens & doc_tokens)
    return overlap / max(1.0, math.sqrt(len(question_tokens)))


def _best_paths_from_single_entity(graph, start_v, edge_cost, max_docs=10, max_hops=3):
    ep_type = graph.ep["relation"]
    v_type = graph.vp["type"]
    visited = {}
    heap = [(0.0, [int(start_v)], [])]
    paths = []

    while heap and len(paths) < max_docs:
        cost, vertices, edges = heapq.heappop(heap)
        current = vertices[-1]
        state_key = (current, len(edges))

        if state_key in visited and visited[state_key] <= cost:
            continue
        visited[state_key] = cost

        if len(edges) >= max_hops:
            continue

        current_v = graph.vertex(current)
        for e in current_v.all_edges():
            nxt = int(e.target()) if int(e.source()) == current else int(e.source())
            if nxt in vertices:
                continue

            rel = str(ep_type[e])
            new_cost = cost + float(edge_cost[e])
            new_vertices = vertices + [nxt]
            new_edges = edges + [(current, nxt, rel)]

            if v_type[graph.vertex(nxt)] == "document":
                paths.append(
                    {
                        "vertices": new_vertices,
                        "edges": new_edges,
                        "cost": new_cost,
                        "hops": len(new_edges),
                        "pair": None,
                    }
                )

            heapq.heappush(heap, (new_cost, new_vertices, new_edges))

    return paths


def yen_k_shortest_paths(graph, source, target, k, edge_cost):
    ep_type = graph.ep["relation"]

    if k <= 0:
        return []

    vlist, elist = shortest_path(graph, source, target, weights=edge_cost)
    if not elist:
        return []

    first_vlist = list(vlist)
    first_edges_uv = _to_edge_pairs(first_vlist)

    accepted = [(first_vlist, first_edges_uv)]
    candidates = []
    accepted_keys = {tuple(first_edges_uv)}
    queued_keys = set()

    for _ in range(1, k):
        last_vlist, last_edges_uv = accepted[-1]

        for i in range(len(last_vlist) - 1):
            spur_node = last_vlist[i]
            root_vertices = last_vlist[: i + 1]
            root_edges = last_edges_uv[:i]

            blocked_edges = set()
            for pvlist, pedges in accepted:
                if len(pvlist) > i and pvlist[: i + 1] == root_vertices:
                    blocked_edges.add(pedges[i])

            efilt = graph.new_edge_property("bool", val=True)
            for e in graph.edges():
                uv = (int(e.source()), int(e.target()))
                vu = (uv[1], uv[0])
                if uv in blocked_edges or vu in blocked_edges:
                    efilt[e] = False

            vfilt = graph.new_vertex_property("bool", val=True)
            for rv in root_vertices[:-1]:
                vfilt[rv] = False

            gv = GraphView(graph, efilt=efilt, vfilt=vfilt)
            spur_vlist, spur_elist = shortest_path(gv, spur_node, target, weights=edge_cost)

            if not spur_elist:
                continue

            spur_vlist = list(spur_vlist)
            total_vlist = root_vertices[:-1] + spur_vlist
            total_edges_uv = root_edges + _to_edge_pairs(spur_vlist)
            path_key = tuple(total_edges_uv)

            if path_key in accepted_keys or path_key in queued_keys:
                continue

            cost = _path_cost_from_pairs(graph, edge_cost, total_edges_uv)
            heapq.heappush(candidates, (cost, total_vlist, total_edges_uv))
            queued_keys.add(path_key)

        if not candidates:
            break

        _, vlist_new, edges_uv_new = heapq.heappop(candidates)
        queued_keys.discard(tuple(edges_uv_new))
        accepted.append((vlist_new, edges_uv_new))
        accepted_keys.add(tuple(edges_uv_new))

    paths = []
    for vlist_path, edges_uv in accepted:
        edges = []
        for u, v in edges_uv:
            e = _edge_from_pair(graph, u, v)
            if e is None:
                continue
            edges.append((u, v, str(ep_type[e])))

        if edges:
            paths.append(
                {
                    "vertices": [int(v) for v in vlist_path],
                    "edges": edges,
                    "cost": _path_cost_from_pairs(graph, edge_cost, edges_uv),
                    "hops": len(edges),
                }
            )

    return paths


def _rank_documents(graph, path_infos, question):
    v_type = graph.vp["type"]
    question_tokens = _question_terms(question)
    doc_stats = defaultdict(lambda: {"path_hits": 0, "min_cost": float("inf"), "entities": set(), "relations": set()})

    for p in path_infos:
        path_vertices = p.get("vertices", [])
        path_edges = p.get("edges", [])
        path_cost = float(p.get("cost", 0.0))
        pair = p.get("pair")

        docs = [v for v in path_vertices if v_type[graph.vertex(v)] == "document"]
        if not docs:
            continue

        for dv in docs:
            st = doc_stats[dv]
            st["path_hits"] += 1
            st["min_cost"] = min(st["min_cost"], path_cost)
            st["relations"].update(rel for _, _, rel in path_edges)
            if pair is not None:
                st["entities"].update(pair)

    ranked = []
    for dv, st in doc_stats.items():
        overlap = _document_overlap_score(graph, dv, question_tokens)
        score = (
            2.0 * len(st["entities"])
            + 1.3 * st["path_hits"]
            + 0.8 * len(st["relations"])
            + 2.2 * overlap
            + 1.5 * (1.0 / (1.0 + st["min_cost"]))
        )
        ranked.append((dv, score, st))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def traverse_graph(graph, entities, question, k=3, max_docs=10):
    if not entities:
        return []

    entities = [e for e in entities if find_node_by_id(graph, e) is not None]
    if not entities:
        return []

    edge_cost = _build_edge_cost_property(graph)

    if len(entities) == 1:
        start_v = find_node_by_id(graph, entities[0])
        if start_v is None:
            return []
        return _best_paths_from_single_entity(
            graph,
            start_v,
            edge_cost=edge_cost,
            max_docs=max_docs,
            max_hops=3,
        )

    path_infos = []
    for i, j in itertools.combinations(entities, 2):
        i_v = find_node_by_id(graph, i)
        j_v = find_node_by_id(graph, j)
        if i_v is None or j_v is None:
            continue

        paths = yen_k_shortest_paths(graph, i_v, j_v, k, edge_cost=edge_cost)
        for path in paths:
            path["pair"] = (i, j)
            if not path.get("vertices"):
                path["vertices"] = _to_vertex_list([(u, v) for u, v, _ in path.get("edges", [])])
            path_infos.append(path)

    if not path_infos:
        return []

    ranked_docs = _rank_documents(graph, path_infos, question)
    selected_docs = {dv for dv, _, _ in ranked_docs[:max_docs]}

    filtered = []
    for p in sorted(path_infos, key=lambda x: x.get("cost", float("inf"))):
        vertices = p.get("vertices", [])
        if any(v in selected_docs for v in vertices):
            filtered.append(p)

    if not filtered:
        filtered = sorted(path_infos, key=lambda x: x.get("cost", float("inf")))[: max(1, k)]

    return filtered


def create_json(path_infos, graph, question, entities, top_docs=10):
    structure = {
        "question": question,
        "entities": entities,
        "documents": [],
        "subgraph": {
            "nodes": [],
            "edges": [],
            "paths": [],
        },
        "adjacency": {},
    }

    if not path_infos:
        return structure

    v_id = graph.vp["id"]
    v_type = graph.vp["type"]
    v_title = graph.vp["title"]
    v_abstract = graph.vp["abstract"]
    edge_cost = _build_edge_cost_property(graph)

    ranked_docs = _rank_documents(graph, path_infos, question)
    doc_rank = {dv: idx + 1 for idx, (dv, _, _) in enumerate(ranked_docs)}

    for dv, score, stats in ranked_docs[:top_docs]:
        v = graph.vertex(dv)
        mesh_vals = list(graph.vp["mesh_terms"][v]) if "mesh_terms" in graph.vp else []
        structure["documents"].append(
            {
                "pmid": str(v_id[v]),
                "rank": doc_rank[dv],
                "score": round(score, 4),
                "evidence_path_count": stats["path_hits"],
                "supporting_entities": sorted(stats["entities"]),
                "title": str(v_title[v]),
                "abstract": str(v_abstract[v]),
                "mesh": mesh_vals,
            }
        )

    node_seen = set()
    edge_seen = set()
    for idx, p in enumerate(sorted(path_infos, key=lambda x: x.get("cost", float("inf"))), start=1):
        path_edges = p.get("edges", [])
        if not path_edges:
            continue

        docs_on_path = set()
        for u, v, rel in path_edges:
            for vv in (u, v):
                if vv in node_seen:
                    continue
                node_seen.add(vv)
                gv = graph.vertex(vv)
                n_type = str(v_type[gv])
                node = {
                    "vertex": int(vv),
                    "id": str(v_id[gv]),
                    "type": n_type,
                }
                if n_type == "document":
                    mesh_vals = list(graph.vp["mesh_terms"][gv]) if "mesh_terms" in graph.vp else []
                    node["title"] = str(v_title[gv])
                    node["abstract"] = str(v_abstract[gv])
                    node["mesh"] = mesh_vals
                    if vv in doc_rank:
                        node["rank"] = doc_rank[vv]
                    docs_on_path.add(str(v_id[gv]))

                structure["subgraph"]["nodes"].append(node)

            edge_key = (u, v, rel)
            if edge_key not in edge_seen:
                edge_seen.add(edge_key)
                e = _edge_from_pair(graph, u, v)
                structure["subgraph"]["edges"].append(
                    {
                        "source_vertex": u,
                        "target_vertex": v,
                        "source_id": str(v_id[graph.vertex(u)]),
                        "target_id": str(v_id[graph.vertex(v)]),
                        "relation": rel,
                        "cost": round(float(edge_cost[e]), 4) if e is not None else None,
                    }
                )

                source_id = str(v_id[graph.vertex(u)])
                structure["adjacency"].setdefault(source_id, []).append(
                    {
                        "relation": rel,
                        "connected_to": str(v_id[graph.vertex(v)]),
                    }
                )

        structure["subgraph"]["paths"].append(
            {
                "path_id": idx,
                "pair": p.get("pair"),
                "cost": round(float(p.get("cost", 0.0)), 4),
                "hops": int(p.get("hops", len(path_edges))),
                "node_ids": [str(v_id[graph.vertex(vx)]) for vx in p.get("vertices", [])],
                "document_ids": sorted(docs_on_path),
                "edges": [
                    {
                        "source_id": str(v_id[graph.vertex(u)]),
                        "target_id": str(v_id[graph.vertex(v)]),
                        "relation": rel,
                    }
                    for u, v, rel in path_edges
                ],
            }
        )

    structure["documents"].sort(key=lambda x: (-x["score"], x["pmid"]))
    structure["subgraph"]["nodes"].sort(key=lambda x: (x["type"], x["id"]))
    structure["subgraph"]["edges"].sort(key=lambda x: (x["source_id"], x["target_id"], x["relation"]))
    for src in structure["adjacency"]:
        structure["adjacency"][src].sort(key=lambda x: (x["relation"], x["connected_to"]))

    return structure


def process_question(question, graph):
    entities = extract_entities_from_question(question, top_k=5)
    path_infos = traverse_graph(graph, entities, question=question, k=5, max_docs=10)
    structure = create_json(path_infos, graph, question=question, entities=entities, top_docs=10)
    return structure

