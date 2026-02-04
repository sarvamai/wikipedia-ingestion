# Run ingestion

## 1. Download data

```bash
DATE=20260201 ./data/download_streams.sh
```

Files are saved to `data/streams/`.

## 2. Update configs

Edit `.env` (and optionally `config/pipeline.json`). Set at least: `OPENSEARCH_HOST`, OpenSearch index/region, AWS keys, Azure embedding endpoint/key, and `INPUT_FILE=data/streams` (or the full path to `data/streams`).

## 3. Run ingestion

```bash
uv run python scripts/01_create_index.py
uv run python scripts/02_run_pipeline.py
```
