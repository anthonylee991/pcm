# Peripheral Cognitive Mesh (PCM): A Biologically-Inspired Memory Architecture for Autonomous AI Agents

**Technical Specification & Whitepaper**  
*Version 2.0 — October 2026*  
*SkillVault Research & Engineering*

---

## Abstract

Most agent memory systems treat persistent state as a document search problem: embed past messages, retrieve the nearest chunks, and place them in the prompt. Search quality matters enormously, and it is the first thing any memory system must get right. But search alone has no lifecycle (it cannot tell a replaced decision from a current one, and it either grows without bound or forgets destructively), no attention (it does not tell the agent what is out of the ordinary), and no sense of absence (it cannot surface a routine that has stopped).

The **Peripheral Cognitive Mesh (PCM)** is the memory architecture behind SkillVault's MemVault. It combines high-quality retrieval (dense vectors, a neural reranker, knowledge-graph links and episode context) with a cognitive layer:

1. **Ebbinghaus decay with a savings effect**, where recall reinforces a memory and slows its future decay.
2. **Pinned guardrails** ($S = 1.0$) that never decay and are always delivered.
3. **Recall-time spreading activation** that keeps the neighbours of recalled memories alive.
4. **Consolidation and demotion** instead of deletion.
5. **Surprisal and absence detection**: memories that break the pattern of their routine, and routines that have gone quiet, are raised as anomaly flags.
6. **Peripheral Attention Engineering (PAE)**: memory delivered in structured slots (`[ANOMALY FLAGS]`, `[ASKER CONTEXT]`, `[SITUATIONAL CONTEXT]`).

On the held-out LoCoMo test split (885 questions), MemVault scores **91.9%** with its default high-context recall (about 2,800 context tokens), against 93.7% with the entire conversation in context. Run through the same harness, Mem0's cloud product scores 89.9% at 6,771 tokens (p = 0.11, not significant). In a 180-day lifecycle simulation, forgetting by demotion preserves accuracy (86.0% vs 85.7% for a fresh store) where deletion loses a quarter of it (60.6%). On a synthetic test of personal routines, anomaly flags raise the share of answers that account for an out-of-the-ordinary event from 35.4% to 60.4% (p = 0.012) and for a stopped routine from 4.5% to 54.5% (p = 0.001).

---

## 1. Introduction: The Document Retrieval Fallacy

Autonomous agents (Claude Code, Cursor, Windsurf, custom harnesses, personal assistants) need continuous context across sessions and devices. The industry default is to adapt **Retrieval-Augmented Generation (RAG)** as "agent memory". RAG was designed for question answering over static corpora. Applied unchanged to an agent's evolving experience, it has three structural problems.

### 1.1 Temporal blindness
If a team decides in January:
> *"Use UUIDv4 for all public API endpoints."*

and refactors in March:
> *"Migrated from UUIDv4 to ULIDs for B-tree index locality."*

a query in April about *"What ID format should I generate?"* is similar to **both** statements. Similarity search has no model of supersession, reinforcement or relevance over time.

### 1.2 No attention
A list of retrieved memories is flat. An out-of-the-ordinary entry (a much longer commute, a symptom during a run, a bill twice the usual size) looks like every other line, and agents frequently overlook it even when it is in their context. Nor can a list show what is missing: a weekly routine that stopped a month ago leaves nothing to retrieve.

### 1.3 No lifecycle
A store that keeps everything at full weight accumulates noise in every recall; a store that deletes old entries loses evidence that may be needed later. Neither behaves like memory.

PCM keeps retrieval as its foundation and adds the missing layers.

---

## 2. Architecture

```
            INGEST                                     RECALL
  ┌──────────────────────────┐          ┌────────────────────────────────────┐
  │ exact / near-duplicate    │          │ query embedding  ║  graph lookup   │
  │ check (same figures only) │          │ (768-d)          ║  (entity-linked │
  │ importance tier           │          │        │         ║   memories)     │
  │ embedding (768-d halfvec) │          │        ▼         ║                 │
  │ async graph extraction    │          │ HNSW candidates (2k, >= 50)        │
  └────────────┬─────────────┘          │ + graph-linked memories            │
               ▼                         │        │                           │
  ┌──────────────────────────┐          │        ▼                           │
  │ BACKGROUND (cron)         │          │ neural reranker, relevance floor  │
  │ • decay sweep             │          │        │                           │
  │ • consolidation           │          │        ▼                           │
  │ • demotion (not deletion) │          │ cognitive score: relevance,        │
  │ • vector + graph repair   │          │ cue-reactivated strength,          │
  │ • surprisal sweep (2 min) │          │ graph links, project scope         │
  └──────────────────────────┘          │        │                           │
                                         │        ▼                           │
                                         │ top k + episode context            │
                                         │ + surprisal / absence flags        │
                                         │        │                           │
                                         │        ▼                           │
                                         │ PAE slots → agent                  │
                                         │ (side effects: reinforcement,      │
                                         │  spreading activation)             │
                                         └────────────────────────────────────┘
```

### 2.1 Ebbinghaus decay and the savings effect
Every non-pinned memory carries a strength $S \in (0, 1]$ that decays exponentially with time. Each recall reinforces it and lengthens its effective half-life (the savings effect), so memories that keep mattering stay strong and transient noise fades.

### 2.2 Pinned guardrails
Architectural invariants, security policies, stack constraints and core preferences are **pinned**: their strength is fixed at 1.0, they are cached in memory, and they are delivered in `[ASKER CONTEXT]` on every recall whether or not the query mentions them.

### 2.3 Spreading activation
When memories are recalled, their nearest associates are reinforced in the background: up to four similarity neighbours ($\cos \ge 0.65$) of each of the top five recalled memories, plus memories from the same episode (saved within 60 minutes, weighted 0.7). Associates are computed at recall time, so no edge table is stored. Spreading activation does not change the current answer; it keeps related evidence alive against decay.

### 2.4 Consolidation and demotion
Clusters of related short-term memories are consolidated into a long-term record that recall can use as a whole episode. Memories whose strength falls below threshold and that have not been recalled for 90 days are **demoted** (state `archived`): they no longer compete by default but remain searchable. Nothing is deleted by forgetting.

### 2.5 Episode context
Single memories rarely answer a question on their own. For the top recalled memories (one per ten requested), recall adds up to three neighbouring memories on each side from the same episode (saved within 60 minutes), in time order and de-duplicated.

### 2.6 Knowledge graph
Each memory is processed by an extraction model into typed entities (13 types: Person, Organization, Place, Event, Activity, Work, Product, Animal, Technology, Project, Concept, Decision, Time) and relations from a closed set of 25 families (KNOWS, FAMILY_OF, DOES, VISITED, LIKES, USES, DEPENDS_ON, REPLACES, DECIDED, ...). Facts are stored in Kùzu, one physically separate database per tenant. At recall, memories linked to the query's entities receive a score bonus proportional to the strength of the link. A repair job re-extracts any memory whose facts are missing from the graph.

### 2.7 Surprisal and absence
A **family** is the set of earlier memories that share a memory's statistical shape: up to 12 earlier memories with $\cos \ge 0.45$, at least four of them. A background sweep asks a small model whether the new memory departs from its family's pattern (an unusual value, a different place, person or product, or something qualitatively different) and stores a score, a note on what is different, the usual pattern, and the family members. At recall:

- a surprising memory ($\text{score} \ge 0.7$) is flagged when it, or any member of its family, was recalled, ranked by the best recalled position of the memory or its family;
- a regular routine of a top recalled memory that has gone quiet is flagged with the date of its last entry and its usual interval.

### 2.8 Peripheral Attention Engineering (PAE)
Recall returns structured slots instead of raw chunks:

- `[ANOMALY FLAGS — PAY ATTENTION]`: surprising memories and stopped routines, each with what is out of the ordinary and what is usual (up to five).
- `[ASKER CONTEXT]`: pinned rules and preferences (up to five).
- `[SITUATIONAL CONTEXT]`: the recalled memories with their dates.

An empty result is reported explicitly ("No relevant memories found") rather than padded with weak matches.

---

## 3. Mathematical Formulation

### 3.1 Memory strength
Initial strength by importance:

$$S_0 = \begin{cases} 1.0 & \text{pinned} \\ 0.8 & \text{high} \\ 0.4 & \text{default} \end{cases}$$

For non-pinned memories, after $t$ days without reinforcement:

$$S(t) = \max\left(0.01, \; S_0 \cdot \exp\left(-\frac{t}{\tau}\right)\right), \qquad \tau = \frac{T_{\text{eff}}}{3}, \qquad T_{\text{eff}} = 90 \cdot \left(1 + 0.25 \cdot \min(n, 20)\right) \text{ days}$$

where $n$ is the number of reinforcements. With $n = 0$, a memory falls to about 5% of its initial strength in 90 days; with $n = 20$ the window grows to 540 days.

### 3.2 Reinforcement
On recall:

$$S \leftarrow \min\left(1.0, \; S + 0.17 \cdot \ln(1 + r)\right)$$

where $r$ is the number of retrievals in the current window.

### 3.3 Spreading activation
For each associate $j$ of a recalled memory $i$ with reinforcement $\Delta S_i$:

$$S_j \leftarrow \min\left(1.0, \; S_j + 0.10 \cdot w_{ij} \cdot \Delta S_i\right), \qquad w_{ij} = \begin{cases} 1.0 & \text{similarity neighbour} \\ 0.7 & \text{same episode} \end{cases}$$

### 3.4 Scoring
Each candidate $m$ for query $q$ is scored from its relevance $r(m, q)$ (the reranker's score when reranked, otherwise cosine similarity) and its strength, with **cue-dependent reactivation**: a strong cue restores a decayed memory to full strength, as a specific question brings back an old memory.

$$c = \min\left(1, \left(\frac{r}{0.40}\right)^2\right), \qquad \tilde{S} = S \cdot (1 - c) + c$$

$$\text{Score}(m, q) = 0.75 \cdot r + 0.25 \cdot \tilde{S} + 0.10 \cdot [\text{same project}] + 0.15 \cdot [\text{graph-linked}] + 0.15 \cdot \ell(m, q)$$

where $\ell(m, q) \in [0, 1]$ is the strength of the graph link between $m$ and the query's entities. Candidates below a relevance floor of 0.35 are discarded.

### 3.5 Reranker gating
The neural reranker is skipped when the vector candidates are already decisive:

$$\text{rerank} = \neg\left( |\mathbf{s}| \le 1 \; \lor \; s_{(1)} \ge 0.88 \; \lor \; s_{(1)} - s_{(2)} \ge 0.15 \right)$$

With high-context candidate pools, the reranker runs on about 98% of recalls; it is worth about 14 accuracy points on LoCoMo.

### 3.6 Recall size
For a recall of $k$ memories (default 50; compact recall passes $k = 10$): $k$ situational slots, a candidate pool of $\max(50, 2k)$, a prompt budget of $\max(14{,}000, 900k)$ characters, and $\max(3, k/10)$ episode seeds.

### 3.7 Surprisal
For memory $m$ with family $F(m) = \{ m' : t(m') < t(m), \cos(m, m') \ge 0.45 \}$, the 12 nearest, $|F(m)| \ge 4$: a judge model returns $\sigma(m) \in [0, 1]$, a note on what differs and the usual pattern. $m$ is flagged at recall when $\sigma(m) \ge 0.7$ and $(\{m\} \cup F(m)) \cap R \neq \emptyset$ for the recalled set $R$; flags are ordered by $\min_{x \in \{m\} \cup F(m)} \text{rank}_R(x)$.

### 3.8 Absence
For a top recalled memory $m$, its routine is $\{m\} \cup F(m)$ plus every memory whose family shares at least two members with it. With entry times $t_1 < \dots < t_N$ ($N \ge 5$), intervals $g_i = t_{i+1} - t_i$, median $\tilde{g}$ and coefficient of variation $\text{cv}(g) < 1$, the routine is flagged when

$$4 \text{ days} \le t_{\text{now}} - t_N \le 365 \text{ days} \quad \text{and} \quad t_{\text{now}} - t_N \ge 3 \cdot \tilde{g}.$$

---

## 4. Implementation

### 4.1 Ingestion
- **Exact duplicates** reinforce the existing memory without an embedding call.
- **Near duplicates** ($\cos \ge 0.94$) are merged only when they state the same figures; "ran 5k in 27:10" and "ran 5k in 41:30" are different events.
- **Embeddings**: OpenAI `text-embedding-3-small` at 768 dimensions, stored as `halfvec(768)` in PostgreSQL with an HNSW index; inputs are capped at 20,000 characters (the full text is stored).
- **Graph extraction** runs asynchronously after the write; failed extractions are retried by a repair job rather than stored as guesses.

### 4.2 Recall pipeline
Query embedding and graph lookup run in parallel, followed by HNSW search, reranking (DashScope `qwen3-rerank`), scoring, episode expansion and anomaly flags. Median recall latency is about 0.7-1.0 s, dominated by the embedding (~300-400 ms) and reranker (~400-500 ms) calls; scoring, graph and flag queries add tens of milliseconds.

### 4.3 Pinned guardrails cache
Pinned memories are mirrored in an in-process cache, invalidated on every change, so `[ASKER CONTEXT]` costs no database round trip.

### 4.4 Detached side effects
Reinforcement and spreading activation run after the response is returned, off the request path.

### 4.5 Background jobs
| Job | Cadence | Purpose |
| :--- | :--- | :--- |
| Decay sweep | periodic | Applies the strength function incrementally |
| Consolidation and demotion | periodic | Consolidates clusters, demotes faded memories |
| Vector repair | every 10 min | Embeds memories stored without a vector; re-embeds after a model change |
| Graph repair | every 10 min | Re-extracts memories missing from the graph; rebuilds a lost tenant graph |
| Surprisal sweep | every 2 min | Judges new memories against their families |

### 4.6 Multi-tenancy
Postgres rows are scoped per user and organization. Each tenant's graph is a physically separate Kùzu database directory on a persistent volume.

### 4.7 Space
About 4.5 KB per memory: text 0.48 KB, `halfvec(768)` 2.0 KB, HNSW index and keys 2.06 KB.

---

## 5. Evaluation

### 5.1 Method
- **Datasets**: LoCoMo (10 conversations; dev split of 4 conversations, held-out test split of 6, 885 questions in categories 1-4), LongMemEval-S, and a synthetic personal-surprisal set (Section 5.6).
- **Systems** run the production code path: real embeddings, reranker, Postgres and graph.
- **Answer model**: DeepSeek V4.1 Flash; **judge**: Qwen3.8 Flash. Prompts are copied verbatim from Mem0's published harness (LoCoMo) and the LongMemEval authors' code. Accuracy is the binary "classic" J-score; we also report Mem0's more lenient judge.
- **Statistics**: paired exact McNemar tests on the same questions. Configurations are chosen on dev and confirmed once on test.

### 5.2 LoCoMo
| System (test split, 885 questions) | Accuracy (95% CI) | Context tokens |
| :--- | :---: | :---: |
| Embedding search, top 50 | 77.1% | 1,918 |
| MemVault, compact recall (k = 10) | 87.3% | ~1,090 |
| **MemVault, default (k = 50 + episodes)** | **91.9%** (89.9-93.5) | ~2,830 |
| Full conversation in context (ceiling) | 93.7% (91.9-95.1) | 28,201 |

By category (default): multi-hop 87.2%, temporal 92.2%, open-domain 67.9%, single-hop 96.0%. With Mem0's judge the same answers score 94.2%.

Contributions measured as paired steps on test: switching to OpenAI embeddings +6.3 points (p = 5e-9), episode context +2.0 (p = 0.03), high-context recall +4.6 (p = 3e-7). The knowledge graph's linked-memory mode added +6.6 points on dev.

### 5.3 Head-to-head with Mem0 (same harness)
Mem0's cloud product answered 776 of the 885 test questions; on those:

| System | Accuracy | Context tokens |
| :--- | :---: | :---: |
| **MemVault default** | **91.8%** | ~2,830 |
| Mem0 cloud, top 200 | 89.9% | 6,771 |
| Mem0 cloud, top 50 | 85.6% | 1,616 |
| Mem0 cloud, top 10 | 75.3% | — |

MemVault vs Mem0 top 200: p = 0.11 (not significant). Vs Mem0 top 50: p = 6e-8.

Published LoCoMo results from other vendors use different answer models, judges and prompts and are not comparable to these numbers: Zep 94.7% (an independent re-run measured 75.1%), Mem0 92.5%, ByteRover 92.2%, Letta 74.0%.

### 5.4 LongMemEval (knowledge updates only)
On the 72 knowledge-update questions of LongMemEval-S (does memory return the newest version of a changed fact?), split into fixed halves: 97.2% on the dev half (the newest evidence was retrieved for 36 of 36 questions) and 91.7% on the held-out half. The full benchmark has not been run.

### 5.5 Lifecycle
A 180-day simulation on the LoCoMo dev split: half of the questions are asked on a schedule (production recall, with reinforcement and spreading activation), the clock advances daily, production decay and pruning run each day, and the other half is evaluated at the end.

| Forgetting policy after 180 days | Accuracy on never-asked questions |
| :--- | :---: |
| Fresh store (no time elapsed) | 85.7% |
| Delete faded memories | 60.6% (p = 1e-16) |
| Delete, with stronger spreading activation | 62.5% |
| **Demote faded memories (production)** | **86.0%** (p = 1.0 vs fresh) |

Spreading activation alone kept more of the not-yet-asked evidence alive (58.3% vs 55.8% without it) but cannot compensate for deletion. Consolidation, measured separately at production defaults on dev, raised accuracy from 90.7% to 92.7% (p = 0.024).

### 5.6 Surprisal and absence (synthetic)
No public benchmark measures whether a memory system helps an agent notice what is out of the ordinary, so we built one. 24 synthetic personas (12 dev, 12 test), each with 8 personal routines of 8-12 entries (runs, sleep, bills, calls home, groceries) and about 30 unrelated memories. Half the routines contain one planted outlier among their last four entries (a value, category or content change, stated matter-of-factly); in a second version, half of the remaining routines stop well before the question. Each routine has an everyday request, asked the day after, that never mentions the outlier. The dataset is generated once and frozen.

| Held-out test split | Without flags | With flags |
| :--- | :---: | :---: |
| Answers accounting for a planted outlier | 35.4% | **60.4%** (p = 0.012) |
| Same, second independent ingest | 41.7% | 50.0% (p = 0.42) |
| Answers noticing a stopped routine | 4.5% | **54.5%** (p = 0.001) |
| Answers inventing an anomaly (control routines) | 0-2.1% | 3.8-4.2% (n.s.) |

Without flags, the outlier was already in the agent's context 90-95% of the time; agents simply did not notice it. The detector flags 77-100% of value outliers and 84-93% of content outliers but only 53-58% of category changes (a different store or person), and flags a normal entry in about 1-2% of judged memories. Every variant that caught more category changes (three prompt designs and a structural attribute check) also flagged more normal entries, so the precise detector is the one in production. No continuing routine was flagged as having stopped.

The same test exposed a data-integrity problem in near-duplicate merging (routine entries with different figures were merged, and the older entry was re-dated), which is why near duplicates must now state the same figures.

### 5.7 What did not help
Measured on dev against the production configuration, all not significant: multi-hop graph walks, canonical entity merging, sibling ("swarm") scouting, adaptive result breadth, candidate pools of 100 or 150, LLM-written summaries in place of source memories (-7 points), a faster third-party reranker (no single relevance floor matched both accuracy and off-topic precision), and stored associative edges (about half of storage, no accuracy effect).

### 5.8 Limitations
- LoCoMo results are in the same band as the best published systems but do not beat their self-reported numbers, which use different answer models and judges.
- Only the knowledge-update category of LongMemEval has been measured.
- The surprisal results come from a synthetic dataset we built; they demonstrate the mechanism, not real-world prevalence.
- Category changes are the weakest kind of surprisal.
- Recall latency (about 0.7-1.0 s) is dominated by hosted embedding and reranking calls.

---

### Citation
```bibtex
@article{skillvault2026pcm,
  title={Peripheral Cognitive Mesh (PCM): A Biologically-Inspired Memory Architecture for Autonomous AI Agents},
  author={SkillVault Engineering},
  year={2026},
  version={2.0},
  url={https://skillvault.dev/pcm-spec}
}
```
