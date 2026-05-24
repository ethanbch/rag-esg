# ESG RAG Studio

![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![RAG](https://img.shields.io/badge/RAG-Retrieval--Augmented%20Generation-6A5ACD)
![Streamlit](https://img.shields.io/badge/Streamlit-1.49%2B-FF4B4B?logo=streamlit&logoColor=white)
![Vector DB](https://img.shields.io/badge/Vector%20DB-ChromaDB-6E56CF)
![API](https://img.shields.io/badge/API-Albert-0055A4)
![Embedding](https://img.shields.io/badge/Embedding-BAAI%2Fbge--m3-0A66C2)
![Reranker](https://img.shields.io/badge/Reranker-BAAI%2Fbge--reranker--v2--m3-1F7A8C)
![Evaluation](https://img.shields.io/badge/Evaluation-ROUGE-2E8B57)

Production-ready local RAG workspace for ESG reports:

- PDF ingestion pipeline (parse -> chunk -> index)
- Streamlit chat UI with grounded evidence
- Multi-collection retrieval with reranking controls
- Score-threshold filtering to remove weak chunks
- Highlighted source evidence rendered directly in PDFs

## What This Project Does

This project lets you index ESG documents into ChromaDB and ask grounded questions.
The app shows both generated answers and the exact retrieved evidence chunks.

Main use cases:

- ESG report exploration and Q&A
- Traceable answer generation with source chunks
- Prompt and retrieval parameter experimentation
- Lightweight automatic evaluation with ROUGE metrics

## Architecture & Pipeline

```mermaid
flowchart TB
    %% Colors & Styles
    classDef db fill:#e1bee7,stroke:#8e24aa,stroke-width:2px,color:#000000;
    classDef process fill:#bbdefb,stroke:#1976d2,stroke-width:1px,color:#000000;
    classDef llm fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,color:#000000;
    classDef user fill:#c8e6c9,stroke:#388e3c,stroke-width:2px,color:#000000;
    %% PART 1: INGESTION (Offline)
    subgraph Ingestion["Document Ingestion Pipeline (ingestion_pipeline/)"]
        direction LR
        S0[Scraping / Collection<br/>step00] --> S1[Text Parsing<br/>step01]
        S1 --> S2[Strategic Chunking<br/>step02]
        S2 --> S3[Bi-Encoder Embedding<br/>step03]
        S3 --> DB[(ChromaDB<br/>Vector Store)]
    end

    class S0,S1,S2,S3 process;
    class DB db;

    %% PART 2: QUERYING (Online / RAG App)
    subgraph RAG["Retrieval-Augmented Generation RAG (rag_app/service.py)"]
        direction TB

        UI((User<br/>Streamlit UI)) -->|Asks a question| Q_Orig[Original Question]

        %% Phase 1: Expansion
        subgraph MQR["Phase 1: Multi-Query Expansion"]
            Q_Orig --> LLM_MQR{LLM Albert}
            LLM_MQR -->|Generates| Q_Alt1[Alt. Question 1]
            LLM_MQR -->|Generates| Q_Alt2[Alt. Question 2]
            LLM_MQR -->|Generates| Q_AltN[Alt. Question N]
        end

        %% Phase 2: Bi-Encoder
        subgraph Retrieval["Phase 2: Fast Retrieval (Bi-Encoder BGE-M3)"]
            Q_Orig -.-> VSearch[Vector Search<br/>Reranker Disabled]
            Q_Alt1 -.-> VSearch
            Q_Alt2 -.-> VSearch
            Q_AltN -.-> VSearch
            VSearch <==> DB
            VSearch --> Pool[Large Pool of<br/>Raw Chunks]
        end

        %% Phase 3: Cross-Encoder
        subgraph Reranking["Phase 3: Deduplication & Fine Reranking (Cross-Encoder)"]
            Pool --> Dedup[Strict Deduplication<br/>by Chunk ID]
            Dedup --> Chunks_U[Unique Chunks]
            Chunks_U --> CrossEnc{Reranker API}
            Q_Orig -.->|Exclusive Reference| CrossEnc
            CrossEnc --> Filter[Minimum Score Filtering<br/>& Top-K Max]
            Filter --> Context[Final High-Quality<br/>Context]
        end

        %% Phase 4: Generation
        subgraph Generation["Phase 4: Synthesis"]
            Context --> BuildPrompt[Prompt Assembly<br/>System + Context]
            Q_Orig --> BuildPrompt
            BuildPrompt --> LLM_Gen{LLM Albert 120b}
            LLM_Gen --> Answer[Final Answer]
        end
    end

    Answer --> UI

    class BuildPrompt,VSearch,Dedup,Filter,Pool,Chunks_U,Context process;
    class LLM_MQR,LLM_Gen,CrossEnc llm;
    class UI,Answer user;

    %% Cross-graph link for readability
    Ingestion ~~~ RAG
```

## Core RAG Flow

At query time, the app executes the following robust pipeline:

1. **Multi-Query Retrieval (MQR)**: Generate diverse semantic variations of the original question.
2. **Fast Bi-Encoder Retrieval**: Retrieve a large pool of Top K chunks for _all_ query variations (Reranker is disabled here for speed and recall).
3. **Deduplication**: Keep only unique chunks based on their ID.
4. **Cross-Encoder Reranking**: Re-evaluate the entire deduped pool strictly against the _original question_.
5. **Score Filtering**: Drop chunks below the minimum rerank score.
6. **Final Generation**: Generate the answer from the filtered, high-quality context chunks.
7. **Traceability UI**: Render the answer alongside evidence chunks and optional highlighted PDF pages.

## Quick Start

### 1) Prerequisites

- Python >= 3.12
- `uv` installed
- Albert API key

### 2) Install dependencies

```bash
uv sync
```

### 3) Configure environment

Create a `.env` file at project root:

```bash
ALBERT_API_KEY=your_api_key
```

Optional variables are listed in the Configuration section.

### 4) Run ingestion (local PDF folder)

```bash
uv run python -m ingestion_pipeline.main
```

This indexes PDFs from `downloads/` into local ChromaDB.

### 5) Launch Streamlit

```bash
uv run streamlit run streamlit_app.py
```

## Streamlit UI Overview

### Core Config

- API key (runtime override)
- LLM model selection from API-discovered models
- Chroma collection multi-select

### Upload PDF

From the sidebar:

1. Select a PDF
2. Click Ingest & index uploaded PDF
3. Ask questions immediately against the new collection

The upload flow stores the PDF in a dedicated collection and tracks page metadata for evidence highlighting.

### Advanced Controls

The following controls are available under Advanced controls:

- Embedding model (read-only)
- Enable reranker (On/Off)
- Reranker model (read-only)
- Top K (per collection retrieval)
- Final Top K (final chunk count after filtering)
- Reranker candidate pool (max candidates sent to reranker)
- Min rerank score (chunks below this score are discarded)
- System prompt

Default values used by Reset settings:

- `Top K = 3`
- `Final Top K = 8`
- `Reranker candidate pool = 24`
- `Min rerank score = 0.25`
- `System prompt = DEFAULT_SYSTEM_PROMPT`
- `Reranker backend/model = config defaults`
- `Embedding model = config default`

Two utility actions are available:

- Reset settings: resets all Advanced controls to defaults
- Reset prompt: resets only the system prompt

## Evidence & Traceability

For each assistant response, you can inspect:

- Evidence chunks (text, score, distance, metadata)
- PDF evidence (highlighted pages when source PDF is available)

Notes:

- Highlight quality depends on extractable text quality
- Scanned/OCR-poor PDFs may reduce highlight precision
- For historical indexed collections, source PDFs should exist locally (for example in `downloads/`)

## Configuration

Configured in `config.py` and environment variables.

```bash
ALBERT_API_KEY=...
EMBEDDING_MODEL=BAAI/bge-m3
RERANK_BACKEND=api            # api | none
RERANK_MODEL=BAAI/bge-reranker-v2-m3

CHUNKING_STRATEGY=token_overlap   # token_overlap | sentence_overlap
MIN_CHUNK_TOKENS=300
MAX_CHUNK_TOKENS=500
CHUNK_OVERLAP_TOKENS=50
SENTENCE_CHUNK_SIZE=8
SENTENCE_CHUNK_OVERLAP=2
```

## Evaluation

### ROUGE Metrics

Run automatic basic evaluation using ROUGE (token overlap):

```bash
uv run evaluate_rag.py --dataset evaluation/qa_dataset.example.json --output evaluation/last_eval_report.json
```

### LLM-as-a-judge

For a deeper semantic evaluation assessing correctness, relevance, and groundedness of the generation:

```bash
uv run evaluate_rag_llmaaj.py --dataset evaluation/qa_dataset.example.json --output evaluation/last_eval_report_llm.json
```

Evaluation reports include:

- Per-sample scores (ROUGE / LLM judgments)
- Aggregated averages
- Retrieved chunk metadata

Tip: prefer the one-line command above to avoid shell line-break issues.

## Project Structure

```text
.
├── config.py
├── evaluate_rag.py
├── evaluate_rag_llmaaj.py
├── pyproject.toml
├── streamlit_app.py
├── ingestion_pipeline/
│   ├── main.py
│   ├── step00_scraping.py
│   ├── step01_parsing.py
│   ├── step02_chunking.py
│   └── step03_indexing.py
├── rag_app/
│   ├── albert_client.py
│   ├── prompts.py
│   ├── retriever.py
│   ├── service.py
│   └── types.py
├── streamlit_ui/
│   └── ui.py
├── evaluation/
│   ├── qa_dataset.example.json
│   ├── last_eval_report.json
│   └── last_eval_report_llm.json
└── chroma_db/
```

## Troubleshooting

### `ALBERT_API_KEY is required`

Set `ALBERT_API_KEY` in `.env` or in the Streamlit sidebar.

### `zsh: command not found: --dataset`

Use the evaluation command on one line:

```bash
uv run evaluate_rag.py --dataset evaluation/qa_dataset.example.json --output evaluation/last_eval_report.json
```

### No collections found in UI

Run ingestion first:

```bash
uv run python -m ingestion_pipeline.main
```

### Too many irrelevant chunks

Increase `Min rerank score` in Advanced controls and/or reduce `Final Top K`.
