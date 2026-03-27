# Beyond Cosine Similarity: Why Graph Topology is the Future of RAG Evaluation

*Published on the release of Domain-Specific RAG Evaluation & MLOps Platform v1.1.0*

---

Every RAG practitioner eventually hits the same wall. You carefully tune your retriever,
optimise your chunking strategy, and run your evaluation suite — and cosine similarity
gives you a respectable 0.78. Then you read the model's actual output and find a
hallucinated answer stitched together from three loosely-connected passages about
entirely different processes.

The problem is not your retriever. The problem is your metric.

## The Fundamental Blind Spot of Embedding-Based Evaluation

Cosine similarity measures the angle between two vectors. It is a excellent proxy for
lexical and semantic relatedness, but it is fundamentally a **pairwise, context-free
measure**. It can tell you that a retrieved chunk *mentions* the right topic. It cannot
tell you whether the *set* of retrieved chunks forms a coherent, interconnected body of
knowledge capable of supporting a factual answer.

Imagine retrieving five documents about "steel plate inspection" from a manufacturing
knowledge base. Three are from the same process standard and are tightly inter-referenced.
Two are from unrelated quality reports that happen to share a keyword. Cosine similarity
between each chunk and the question might score 0.75 across the board. But the retrieval
set is structurally fragmented — and that fragmentation is exactly what causes a model to
confabulate connections that do not exist.

## Enter Graph Context Relevance

Our **GCR evaluator** models the retrieved context as a subgraph of a domain knowledge
graph and computes three complementary topological scores:

$$\mathrm{GCR} = \mathrm{clip}\!\left(\;0.4 \cdot S_e \;+\; 0.4 \cdot S_c \;-\; 0.2 \cdot P_h,\; 0,\; 1\right)$$

### $S_c$ — Structural Connectivity: Rewarding Coherent Retrieval

$S_c$ answers the question: *how connected is what we retrieved?*

$$S_c = \frac{|V_{LCC}|}{|V_R|}$$

where $V_{LCC}$ is the largest connected component in the undirected projection of the
retrieved subgraph, and $V_R$ is the full set of retrieved nodes. A retrieval set where
all nodes are reachable from one another scores $S_c = 1.0$. A set of five isolated
fragments scores $S_c = 0.2$. The intuition is that a model can only reason coherently
over a context that is itself coherent — connectivity is a prerequisite for faithful
multi-hop reasoning.

### $P_h$ — Hub Noise Penalty: Discouraging Generic Over-Retrieval

$P_h$ answers the question: *how much of what we retrieved is generic noise?*

$$P_h = \frac{|\{v \in V_R : \deg(v) > \mu_d + 2\sigma_d\}|}{|V_R|}$$

High-degree "hub" nodes in a knowledge graph are typically high-coverage but low-precision
— they connect to everything and therefore carry little discriminative signal. A hub about
"quality control" in a manufacturing KG might appear relevant to almost any question in
the domain, but including it dilutes the retrieved context with noise. The $P_h$ term
applies a $-0.2$ penalty proportional to the hub fraction, directly disincentivising
retrievers that pad context with popular-but-vague nodes.

## Why This Is Hard — and Why We Solved It

The engineering challenge is that rigorous graph-topological evaluation requires a
persistent, content-addressed knowledge graph that survives across pipeline runs without
manual invalidation. We engineered a **SHA-256-keyed SQLite GraphStore** that
serialises full `networkx.Graph` objects — including all node and edge attributes — and
achieves deterministic cache hits with O(1) lookup. Every metric is computed in O(N + E)
time. No model calls. No network I/O.

## 100% Containerisation: AI Evaluation as Rigorous Software Engineering

The second axis of innovation is operational. AI evaluation pipelines have a reputation
for being fragile research code. We challenged that assumption directly.

The platform ships with a **733-test parallel suite** (`pytest-xdist` across 369
`eval-pipeline` tests and 364 `services` tests), deterministic offline fallbacks for every
LLM-dependent path, and a `Dockerfile.test` that reproduces the exact CI environment
locally. Every developer operation — from running the full test suite to generating OpenAPI
specs to rebuilding the Insights Portal — is available as a Docker Compose service.
**Zero host Python, Node.js, or CLI tools required.**

The result is an AI evaluation platform that satisfies the same engineering contract as
traditional software: reproducible, independently testable, and deployable anywhere
containers run.

## What This Means for RAG Practitioners

Graph-topological evaluation is not a replacement for cosine similarity — it is a
complement. $S_e$ (entity overlap) still captures semantic relevance. $S_c$ and $P_h$
add the structural dimension that text-based metrics cannot see. Together they produce a
composite score that correlates more strongly with human judgements of retrieval quality
for domain-specific corpora than any single metric alone.

The future of RAG evaluation is not a single number — it is a multidimensional view of
*what knowledge was retrieved, how coherently it hangs together, and whether it was precise
enough to matter*. Graph topology gives us that view.

---

*The platform is open-source. See the full technical specification in
[RELEASE_NOTES_v1.0.0.md](../RELEASE_NOTES_v1.0.0.md) and the implementation in
`eval-pipeline/src/evaluation/graph_context_relevance.py`.*
