# BioASQ Knowledge Graph Structure

## Overview

The knowledge graph is stored in **graph-tool's `.gt` format** - a binary format optimized for fast load/save and memory efficiency.

```python
import graph_tool
g = graph_tool.load_graph("knowledge_graph/bioasq_kg.gt")
```

---

## Nodes (Vertices)

Two types of nodes, distinguished by the `type` vertex property:

### 1. Document Nodes (`type="document"`)

| Property | Type | Description |
|----------|------|-------------|
| `id` | string | PubMed ID (PMID), e.g., `"15858239"` |
| `title` | string | Article title |
| `abstract` | string | Article abstract text |
| `mesh_terms` | vector\<string\> | MeSH (Medical Subject Headings) terms |

### 2. Entity Nodes (`type="entity"`)

| Property | Type | Description |
|----------|------|-------------|
| `id` | string | Normalized entity name (lowercase), e.g., `"hirschsprung disease"` |
| `title` | string | Empty (`""`) |
| `abstract` | string | Empty (`""`) |
| `mesh_terms` | vector\<string\> | Empty (`[]`) |

---

## Edges

Two types of edges, distinguished by the `relation` edge property:

### 1. Document-Entity Edges (`relation="related-to"`)

- Connects document nodes to entity nodes extracted from the associated BioASQ question
- **Undirected**
- Example: `Document:15858239 <--related-to--> Entity:"hirschsprung disease"`

### 2. Entity-Entity Edges (`relation=<label>`)

- Connects two entity nodes with a labeled relationship
- Relations extracted from question text using REBEL model
- **Undirected**
- Example: `Entity:"RET" <--encodes--> Entity:"receptor tyrosine kinase"`

---

## Property Maps

### Vertex Properties (`g.vp`)

```python
g.vp["type"]       # string: "document" or "entity"
g.vp["id"]         # string: PMID (docs) or entity name
g.vp["title"]      # string: Article title (docs only)
g.vp["abstract"]   # string: Abstract text (docs only)
g.vp["mesh_terms"] # vector<string>: MeSH terms (docs only)
```

### Edge Properties (`g.ep`)

```python
g.ep["relation"]   # string: "related-to" or extracted relation label
```

---

## Usage Examples

### Load and Inspect Graph

```python
import graph_tool

g = graph_tool.load_graph("knowledge_graph/bioasq_kg.gt")

print(f"Vertices: {g.num_vertices()}")
print(f"Edges: {g.num_edges()}")

# Count by type
vp_type = g.vp["type"]
docs = sum(1 for v in g.vertices() if vp_type[v] == "document")
entities = sum(1 for v in g.vertices() if vp_type[v] == "entity")
print(f"Documents: {docs}, Entities: {entities}")
```

### Find a Node by ID

```python
def find_node_by_id(g, target_id):
    """Find vertex by its ID (PMID or entity name)."""
    vp_id = g.vp["id"]
    for v in g.vertices():
        if vp_id[v] == target_id:
            return v
    return None

# Find a document
doc = find_node_by_id(g, "15858239")

# Find an entity
entity = find_node_by_id(g, "hirschsprung disease")
```

### Get Neighbors

```python
def get_neighbors(g, vertex):
    """Get all neighbors of a vertex with edge relations."""
    vp_id = g.vp["id"]
    vp_type = g.vp["type"]
    ep_rel = g.ep["relation"]
    
    neighbors = []
    for e in vertex.all_edges():
        other = e.target() if e.source() == vertex else e.source()
        neighbors.append({
            "id": vp_id[other],
            "type": vp_type[other],
            "relation": ep_rel[e]
        })
    return neighbors
```

### BFS Traversal

```python
from graph_tool.search import bfs_search, BFSVisitor

class BFSCollector(BFSVisitor):
    def __init__(self):
        self.visited = []
    
    def discover_vertex(self, v):
        self.visited.append(int(v))

# Run BFS from a starting vertex
visitor = BFSCollector()
bfs_search(g, source=start_vertex, visitor=visitor)
print(f"Visited {len(visitor.visited)} nodes")
```

### Filter Subgraph

```python
# Get only entity nodes
vp_type = g.vp["type"]
entity_filter = g.new_vertex_property("bool")
for v in g.vertices():
    entity_filter[v] = (vp_type[v] == "entity")

# Create filtered view
entity_subgraph = graph_tool.GraphView(g, vfilt=entity_filter)
```

---

## BFS Implementation (Custom)

For more control, see `bfs_from_node()` in `kg2.py`:

```python
from kg2 import load_graph, bfs_from_node

g = load_graph("knowledge_graph/bioasq_kg.gt")
result = bfs_from_node(g, start_id="hirschsprung disease", max_depth=2)

for node_id, depth in result:
    print(f"Depth {depth}: {node_id}")
```
