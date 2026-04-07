3/4/2026 update:
This project was worked on from mid/late July 2025, to early October, and was my very first AI project. During this time, I had to learn a completely new tech stack (docker, kubernetes, local model hosting, vector databases, lots of AI/ML theory, relearned python etc). This was before coding models were as powerful as they are today - For reference, this was completed before Opus 4.5 was released. Coding models back then required substantially more human oversight and correction to maintain code quality, which made heavy AI-assisted development less practical for this project. I would consider this one of the last projects that I have written primarily hand, where over 90% was done manually. 

I used opus 4.6 to cleanup the entire project. I have not self verified all the code, and cannot guarantee that everything matches 1:1 with the original project code. If anything, the structure should remain the same, so the high level idea does not change. When I was doing this project I also intentionally tried to avoid using libraries such as langchain. This is because I felt like libraries over-abstracted the internals which was hindering my learning oppurtunity.

Agents have evolved overtime and the methodology today (as of March 2026) would probably allow for more tasks to be compressed within a single call. This project makes a separate LLM call with a different prompt anytime the agent needs to pick a search method, analyze responses, and guides agent logic through code. While today's agents can do this fully autonomously with no problem. In hindsight, this may be due to the fact that reasoning was still fairly "new" and companies were still in the process of scaling up RL. Long horizon tasks were not as efficient, and context rot was a real issue. At the time, Gemini 2.5 was considered SOTA.

Link to training script: https://github.com/dankit/rag-reranker-finetuning

The chroma embeddings have been published here: https://huggingface.co/datasets/dhlak/legal_chroma_embeddings


# Legal RAG

An AI-powered retrieval-augmented generation system for legal document search and conversational Q&A. Combines dense vector search, sparse keyword search, cross-encoder reranking, and LLM-based synthesis with multi-level self-correction.

---

## System Overview

A legal document search system with a conversational AI interface. The system combines vector search, keyword search, and LLM reasoning with built-in self-triage — it can iteratively refine its own search strategy, fetch more context, try different document collections, and fall back to web search when internal documents are insufficient.

**Three Response Modes:**

1. **Direct Response** — Greetings, clarifications, general knowledge (no search needed)
2. **Hybrid Search** — Query internal document collections (ChromaDB + Elasticsearch)
3. **Web Search** — External web search via Tavily for current events or out-of-corpus queries

---

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────┐
│  ChatAgent                                       │
│  Decides: Direct / SimpleSearch / SequentialSearch│
│  Memory: Rolling window (10 turns) + compression │
└──────────────────┬───────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────┐
│  SearchAgent (orchestrator)                      │
│  1. QueryProcessor  → expand into subqueries     │
│  2. Execute subqueries in parallel               │
│  3. ResultSynthesizer → combine into final answer│
└──────────────────┬───────────────────────────────┘
                   │  (per subquery)
                   ▼
┌──────────────────────────────────────────────────┐
│  SearchMethodSelector                            │
│  LLM routes to Hybrid or Web search              │
│  Excludes previously tried collections           │
│  Falls back to WebSearch after 3 failed attempts │
└─────────┬────────────────────────┬───────────────┘
          │                        │
          ▼                        ▼
┌──────────────────┐    ┌──────────────────┐
│  Hybrid Search   │    │  Web Search      │
│  ChromaDB vector │    │  (Tavily API)    │
│  + Elasticsearch │    └────────┬─────────┘
│  → Weighted RRF  │             │
└────────┬─────────┘             │
         │                       │
         ▼                       │
┌──────────────────┐             │
│  BGE Reranker    │             │
│  Cross-encoder   │             │
│  Top 10 results  │             │
└────────┬─────────┘             │
         │                       │
         ▼                       ▼
┌──────────────────────────────────────────────────┐
│  DocumentSummarizer                              │
│  Summarizes results, can self-correct by         │
│  fetching chunk neighbors/by ID (up to 5x)      │
└──────────────────┬───────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────┐
│  ResultSynthesizer                               │
│  Combines all subquery summaries                 │
│  Can trigger re-searches (up to 3x)              │
│  Falls back to web search at limit               │
└──────────────────────────────────────────────────┘
```

---

## Search Pipeline (detailed)

### 1. Query Expansion

The user's query is expanded into up to 3 subqueries by the `QueryProcessor`. Abbreviations are expanded to canonical forms (e.g. "DUI" → "drinking under the influence"), and the query is rephrased to maximize retrieval relevance. All subqueries execute in parallel via `ThreadPoolExecutor`.

### 2. Search Method Selection (with collection triage)

For each subquery, the `SearchMethodSelector` uses the LLM to choose the best search method:

- **Hybrid Search** — If the query can be answered from an internal document collection
- **Web Search** — If the query is about current events, news, or information not in the corpus

The selector is aware of all available collections and picks the most relevant one. Critically, it **excludes collections that were already tried** for the same query (tracked in `QueryHistory.prev_collection_names`). If a hybrid search on collection A didn't yield good results and the system re-searches, it will try collection B instead — cycling through available collections before giving up.

**Fallback behavior:** If `max_requery_attempts` (default: 3) is reached, the system automatically falls back to web search regardless of what the LLM would choose. If the LLM selects a collection that doesn't exist, it also falls back to web search.

### 3. Hybrid Search

Combines two retrieval methods and fuses their results:

- **Dense search** — BGE-m3 embeddings queried against ChromaDB (inner product similarity)
- **Sparse search** — BM25 keyword matching via Elasticsearch

Results are combined using **Weighted Reciprocal Rank Fusion (RRF)** with near-equal weights (0.495 sparse / 0.505 dense). An alternative linear combination method with min-max normalization is also implemented.

### 4. Reranking

The top 25 hybrid results are reranked by a cross-encoder (`BAAI/bge-reranker-large` in fp16) down to the top 10 most relevant chunks. The reranker uses a mutex lock for thread safety since multiple subqueries may rerank concurrently and the system is not optimized against complications that arise (such as going OOM, fragmented memory allocations).

### 5. Document Summarization (with context refinement)

The `DocumentSummarizer` receives the reranked chunks and their metadata, then asks the LLM to summarize the most relevant information. If the LLM decides it needs more context, it can use tool calls to:

- **GetChunkNeighborsTool** — Fetch chunks adjacent to a relevant chunk (expanding context window)
- **GetChunkByIDTool** — Fetch a specific chunk by its ID

This is a recursive process: after fetching more context, the summarizer re-invokes itself with the expanded document set. This repeats up to `max_tool_calls_attempts` (default: **5 iterations**), with previous tool calls tracked to prevent duplicate work.

### 6. Result Synthesis (with re-search)

The `ResultSynthesizer` combines all subquery summaries into a final cited answer. If the synthesizer determines the summaries are insufficient, it can trigger a **new search** via `SearchTool` — which re-enters the full pipeline (method selection → search → rerank → summarize).

Re-searches are limited to `max_requery_attempts` (default: **3**). Once the limit is reached, the system produces a best-effort answer and clearly articulates uncertainty. If a re-search previously used web search, the query is re-expanded by the `QueryProcessor` to try different phrasing.

---

## Self-Correction Summary

| Level | Component | Actions | Limit | Scope |
|-------|-----------|---------|-------|-------|
| **Context refinement** | DocumentSummarizer | GetChunkNeighbors, GetChunkByID | 5 iterations | Hybrid search only |
| **Re-searching** | ResultSynthesizer | New search (full pipeline re-entry) | 3 attempts | All search types |
| **Collection cycling** | SearchMethodSelector | Excludes previously tried collections | All available | Hybrid search only |
| **Web fallback** | SearchMethodSelector | Falls back to web search | After 3 attempts | Automatic |

---

## ChatAgent: Conversation & Memory

The `ChatAgent` is the user-facing interface. It decides how to handle each message:

| Mode | When | Example |
|------|------|---------|
| **Direct** | Greetings, clarifications, general knowledge | *"Hello"*, *"Thanks"*, *"What is contract law?"* |
| **SimpleSearch** | Single straightforward question | *"What is the speed limit in Iowa school zones?"* |
| **SequentialSearch** | Multi-step questions where answers build on each other | *"Who can file for emancipation and what rights do they gain?"* |

**Sequential search** breaks a complex question into up to 3 logical steps, executes them in order (each step's result becomes context for the next), then synthesizes all step results into a single answer.

**Memory management:** Conversation history uses a rolling window of 10 turns. When the limit is reached, the oldest 5 turns are compressed into a summary by the LLM, preserving context while saving tokens.

---

## Data Ingestion

Two preprocessing pipelines are available:

### LLM Preprocessor (`llm_preprocessor.py`)

Uses Gemini to semantically chunk PDF pages. The LLM decides where to split based on meaning, extracts metadata (topic, section), and handles text that spans page boundaries via a carry-over mechanism. Supports resuming from the last processed page.

### Naive Preprocessor (`naive_preprocessor.py`)

Rule-based pipeline that:
1. Analyzes font sizes/weights to detect headings dynamically
2. Converts PDF pages to markdown with heading hierarchy
3. Splits on markdown headers, then by size with `RecursiveCharacterTextSplitter`
4. Validates chunks (minimum size, word count, alpha ratio, citation filtering)
5. Deduplicates via content hashing
6. Streams page-by-page for memory efficiency

Both pipelines store chunks in ChromaDB (for vector search) and Elasticsearch (for keyword search).

---

## Example Flows

**Simple search (happy path):**
```
"What is the speed limit in Iowa school zones?"
→ ChatAgent selects SimpleSearchTool
→ SearchAgent expands query into subqueries
→ Hybrid search finds relevant chunks from Iowa code collection
→ Reranker picks top 10
→ DocumentSummarizer creates cited answer
→ ResultSynthesizer returns final response
```

**Simple search (with self-correction):**
```
"What are penalties for repeat DUI offenses in Iowa?"
→ Initial hybrid search finds general DUI info
→ DocumentSummarizer needs more context
  → GetChunkNeighbors (fetches adjacent chunks, iteration 1)
  → GetChunkNeighbors (fetches more, iteration 2)
→ Still incomplete → ResultSynthesizer triggers re-search with refined query
→ Second search finds specific repeat-offense penalties
→ Complete answer synthesized
```

**Sequential search:**
```
"If someone is convicted of theft in Iowa, can they later get a professional license?"
→ ChatAgent selects SequentialSearchTool
→ Step 1: "theft conviction penalties Iowa" → finds Class D felony classification
→ Step 2: "professional licensing with felony conviction Iowa" (using Step 1 context)
→ Step 3: "license restoration process Iowa" (using both previous answers)
→ Synthesize: Complete answer about conviction impact + restoration path
```

**Worst case (max attempts, web fallback):**
```
Query about an obscure recent event not in any collection
→ Hybrid search on collection A → irrelevant results
→ Re-search: tries collection B (A excluded) → still insufficient
→ Re-search: tries collection C → still insufficient
→ Max requery attempts (3) reached → automatic fallback to web search
→ Tavily returns current web results
→ Best available answer returned with web citations
```

---

## Project Structure

```
app/
├── agents/          # ChatAgent — conversational interface with tool-calling
├── config/          # SearchConfig, QueryHistory — all tunable parameters
├── core/            # Embeddings (BGE-m3), reranker (BGE-large), ChromaDB, preprocessors
├── evals/           # Retrieval/reranker benchmarking with parameter sweeping
├── prompts/         # All LLM prompts (chat, search, preprocessing)
├── search/          # SearchAgent, hybrid search, query expansion, summarization, synthesis
├── tools/           # Pydantic tool definitions for LLM function calling
├── utils/           # File utilities and Elasticsearch client
└── webscrapers/     # Tavily web search and Iowa Administrative Code scraper
tests/               # Database tests, evaluation scripts, data quality validation
```

---

## Component Reference

| Component | Type | Purpose |
|-----------|------|---------|
| **ChatAgent** | Interface | Conversation management, decides response mode, delegates to SearchAgent |
| **SearchAgent** | Orchestrator | Coordinates full search pipeline, parallel subquery execution |
| **QueryProcessor** | LLM | Expands queries into subqueries, expands abbreviations |
| **SearchMethodSelector** | LLM | Chooses hybrid vs web, cycles collections, web fallback at limit |
| **DocumentSummarizer** | LLM | Summarizes results, fetches more context via tool calls (5x max) |
| **ResultSynthesizer** | LLM | Combines subquery results, triggers re-searches (3x max) |
| **HybridSearch** | Search | ChromaDB (dense) + Elasticsearch (BM25) → Weighted RRF |
| **WebSearch** | Search | Tavily API for external web search |
| **VectorDatabaseClient** | Singleton | ChromaDB connection manager with HNSW configuration |
| **ElasticSearchClient** | Data | BM25 keyword search and bulk indexing |
| **BGEEmbeddings** | Singleton | BAAI/bge-m3 embeddings, GPU-accelerated with semaphore |
| **BGEReranker** | Singleton | BAAI/bge-reranker-large cross-encoder, fp16, mutex-locked |
| **PDFIndexer** | Processor | JSON chunks → embeddings → ChromaDB |
| **LLM_PDF_Preprocessor** | Processor | Gemini-based semantic chunking with metadata |
| **NaivePDFPreprocessor** | Processor | Rule-based markdown chunking with heading detection |

---

## Configuration

All search parameters are centralized in `SearchConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_tool_calls_attempts` | 5 | Max context-refinement iterations in DocumentSummarizer |
| `max_requery_attempts` | 3 | Max re-searches before web fallback |
| `hybrid_search_top_k` | 25 | Number of results from hybrid search before reranking |
| `rerank_top_k` | 10 | Number of results after reranking |
| `excluded_collections` | `["testing"]` | Collections to never search |

---

## Design Patterns

- **Singleton** — `VectorDatabaseClient`, `BGEEmbeddings`, `BGEReranker` all use thread-safe singletons to prevent OOM while allowing parallel search agents
- **Tool-based self-correction** — LLMs dynamically decide search strategy, context refinement, and re-searching, all with iteration limits to prevent infinite loops
- **Separation of concerns** — Query processing → method selection → search execution → summarization → synthesis, each as an independent component
- **Parallel execution** — `ThreadPoolExecutor` for subqueries and tool calls

---

## Technology Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| Vector Database | ChromaDB (HNSW, inner product) | Singleton connection, thread-safe |
| Keyword Search | Elasticsearch (BM25) | Parallel with vector search |
| Embeddings | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | Singleton, GPU-accelerated, semaphore (10 concurrent) |
| Reranker | [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large) | Singleton, fp16, mutex-locked |
| LLM | Gemini 2.5 Flash (Vertex AI) | Query expansion, routing, summarization, synthesis |
| Web Search | [Tavily](https://tavily.com/) | Advanced search depth for current events |
| PDF Processing | PyMuPDF, LangChain | Streaming page-by-page for memory efficiency |
| Orchestration | ThreadPoolExecutor | Parallel subqueries and tool calls |

---

## Setup

### Prerequisites

- Python 3.10+
- Google Cloud project with Vertex AI API enabled
- Elasticsearch instance running (for hybrid search)
- CUDA-capable GPU recommended (for embeddings and reranking)

### Installation

```bash
pip install -r requirements.txt
```

### Environment

1. Copy the environment template and fill in your keys:
   ```bash
   cp .env.example .env
   ```

2. `TAVILY_API_KEY` — Get one at [tavily.com](https://tavily.com/)

3. Authenticate with Google Cloud for Vertex AI:
   ```bash
   gcloud auth application-default login
   ```

### Usage

**Interactive chat:**
```bash
python -m app.agents.chat_agent
```

**Run preprocessing:**
```bash
python -m app.core.naive_preprocessor
```

**Run benchmarks:**
```bash
python -m app.evals.benchmarker
```

**Run tests:**
```bash
pytest tests/
```
