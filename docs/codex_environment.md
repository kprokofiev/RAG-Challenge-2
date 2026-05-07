# Codex Cloud Environment

## Repository

- Repository: `kprokofiev/RAG-Challenge-2`
- Branch: `main`

## Runtime

- Python: `3.11`

The dependency set includes pinned binary packages such as
`faiss-cpu==1.9.0.post1`, which are not available for Python 3.14.

## Setup Script

```bash
bash scripts/codex_setup.sh
```

## Maintenance Script

```bash
bash scripts/codex_setup.sh
```

## Environment Variables

Only add secrets when a task needs to run the worker or call LLM/storage APIs.
Basic code review and test tasks do not need these values.

- `OPENAI_API_KEY`
- `REDIS_URL`
- `STORAGE_ENDPOINT_URL`
- `STORAGE_ACCESS_KEY`
- `STORAGE_SECRET_KEY`
- `STORAGE_BUCKET_NAME`
- `QUEUE_DOC_PARSE_INDEX`
- `QUEUE_REPORT_GENERATE`
- `LOG_LEVEL`
- `JOB_CALLBACK_URL`

## Internet Access

Setup requires internet access for PyPI. Agent internet access can stay disabled
for normal code tasks. Enable it only for tasks that must fetch external docs,
models, or live API data.
