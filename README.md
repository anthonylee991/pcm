# Peripheral Cognitive Mesh (PCM)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**PCM is the lifecycle and attention core of SkillVault's agent memory: Ebbinghaus decay with a savings effect, pinned guardrails that never fade, spreading activation, and Peripheral Attention Engineering (memory delivered as structured prompt slots).**

Development continues in **[ACM (Arboreal Cognitive Mesh)](https://github.com/anthonylee991/acm)**, the architecture that runs in production in [MemVault](https://skillvault.dev). ACM includes everything in PCM plus forgetting by demotion, recall-time spreading activation, episode context, surprisal and absence detection, and the measured benchmark results. This repository is kept for reference and is read-only.

## What is here

| Module | Purpose |
| :--- | :--- |
| `src/core/decay.ts` | Strength model: initial strength by importance, exponential decay, reinforcement on access |
| `src/core/pinned-cache.ts` | In-memory cache for pinned guardrails |
| `src/core/associative.ts` | Association weights and spreading activation |
| `src/core/margin-heuristic.ts` | Decides when a neural reranker is needed |
| `src/core/prompt-builder.ts` | Peripheral Attention Engineering slots (`[ASKER CONTEXT]`, `[SITUATIONAL CONTEXT]`, `[ANOMALY FLAGS]`) |
| `src/graph/` | Kùzu knowledge-graph client and code-graph parser |

```bash
git clone https://github.com/anthonylee991/pcm.git
cd pcm
bun install
bun test
```

## Specification and results

- **[PCM Specification, version 2.0](PCM-SPEC.md)**: architecture, equations and evaluation (also at [skillvault.dev/pcm-spec](https://skillvault.dev/pcm-spec)).
- **[ACM repository](https://github.com/anthonylee991/acm)**: the maintained library, benchmark results and the synthetic surprisal datasets.

## License

MIT © 2026 Anthony Lee & SkillVault Engineering.
