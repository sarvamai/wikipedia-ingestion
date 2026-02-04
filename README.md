# Wikipedia Ingestion (Nested Index)

Ingest Wikipedia CirrusSearch dumps into AWS OpenSearch with a **nested** index: one document per article, each with a `chunks[]` array (content + vector per chunk).

For EC2 or any server: configure `.env` and optionally `config/pipeline.json`, then run the scripts.

## Setup

1. **Clone/copy this repo** to the machine (e.g. EC2).

2. **Create `.env`** from the example:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and set:
   - **OpenSearch:** `OPENSEARCH_HOST` (hostname only), `OPENSEARCH_INDEX`, `OPENSEARCH_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
   - **Input:** `INPUT_FILE` — path to a single dump file or a directory of `.json.gz` / `.json.bz2` files (e.g. `data/streams` after downloading — see below)
   - **Azure embeddings:** `AZURE_EMBEDDING_ENDPOINT`, `AZURE_EMBEDDING_API_KEY`
   - Optional: `MAX_PAGES`, `PROGRESS_FILE`

3. **Optional: edit `config/pipeline.json`** for chunk size, batch size, skip sections, bulk/embedding workers, etc. Values in `.env` override the config file where applicable.

4. **Download dump streams** (optional, if not using your own dumps):
   ```bash
   ./data/download_streams.sh
   ```
   Downloads enwiki CirrusSearch content dumps (20260201, files 00–62) into `data/streams/`. Uses `wget -c` so you can resume. Override with `DATE=20250101` or `COUNT=5` for a small test set.

5. **Install dependencies** (using [uv](https://docs.astral.sh/uv/)):
   ```bash
   uv sync
   ```
   Or with pip: `pip install -r requirements.txt`

## Usage

1. **Create the OpenSearch index** (nested k-NN schema from `config/index_schema.json`):
   ```bash
   uv run python scripts/01_create_index.py
   ```

2. **Run the pipeline** (stream → chunk → embed → push):
   ```bash
   uv run python scripts/02_run_pipeline.py
   ```
   Without uv, use `python scripts/01_create_index.py` and `python scripts/02_run_pipeline.py` after installing deps.
   Progress is saved to `output/pipeline_progress_nested.json` (or `PROGRESS_FILE`). Re-run to resume from the last batch.

## Config files

| File | Purpose |
|------|--------|
| `.env` | Secrets and paths: OpenSearch host (`OPENSEARCH_HOST`), AWS keys, Azure keys, `INPUT_FILE`, `PROGRESS_FILE`, optional overrides |
| `config/index_schema.json` | Index settings and mappings (shards, replicas, nested `chunks` with k-NN vector) |
| `config/pipeline.json` | Chunking and pipeline tuning: `chunk_size`, `batch_size`, `skip_sections`, embedding/bulk workers, etc. |

You only need to update `.env` and, if you want, `config/pipeline.json`; then run the two scripts.

## Structure

```
wikipedia-ingestion/
├── .env                 # You create from .env.example
├── pyproject.toml       # Project and dependencies (uv)
├── uv.lock              # Locked dependencies (uv)
├── config/
│   ├── index_schema.json
│   └── pipeline.json
├── src/
│   ├── dump_reader.py    # Stream pages from CirrusSearch dumps
│   ├── chunking.py      # Sections (source_text == Heading ==) + chunks
│   ├── embedding.py     # Azure embeddings (parallel, truncation)
│   ├── opensearch_client.py
│   └── pipeline_nested.py
├── scripts/
│   ├── 01_create_index.py
│   └── 02_run_pipeline.py
├── data/
│   ├── download_streams.sh   # Download dump into data/streams/
│   └── streams/              # .json.bz2 / .json.gz (set INPUT_FILE to this dir)
├── output/              # Progress file for resume
├── requirements.txt     # For pip (optional)
└── README.md
```

## Notes

- **Resume:** Delete `output/pipeline_progress_nested.json` to start from scratch; otherwise the pipeline skips already-processed batches.
- **Full dump:** Leave `MAX_PAGES` unset to process the entire dump. Expect long runtimes (days) depending on size and API limits.
- **Embedding limit:** Chunk text is truncated to stay under the Azure token limit (8192); long sections are split by `chunk_size`/overlap in `config/pipeline.json`.
