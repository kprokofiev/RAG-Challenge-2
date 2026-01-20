# RAG Challenge Winner Solution

**Read more about this project:**
- Russian: https://habr.com/ru/articles/893356/
- English: https://abdullin.com/ilya/how-to-build-best-rag/

This repository contains the winning solution for both prize nominations in the RAG Challenge competition. The system achieved state-of-the-art results in answering questions about company annual reports using a combination of:

- Custom PDF parsing with Docling
- Vector search with parent document retrieval
- LLM reranking for improved context relevance
- Structured output prompting with chain-of-thought reasoning
- Query routing for multi-company comparisons

## Disclaimer

This is competition code - it's scrappy but it works. Some notes before you dive in:

- IBM Watson integration won't work (it was competition-specific)
- The code might have rough edges and weird workarounds
- No tests, minimal error handling - you've been warned
- You'll need your own API keys for OpenAI/Gemini
- GPU helps a lot with PDF parsing (I used 4090)

If you're looking for production-ready code, this isn't it. But if you want to explore different RAG techniques and their implementations - check it out!

## Quick Start

Clone and setup:
```bash
git clone https://github.com/IlyaRice/RAG-Challenge-2.git
cd RAG-Challenge-2
python -m venv venv
venv\Scripts\Activate.ps1  # Windows (PowerShell)
pip install -e . -r requirements.txt
```

Rename `env` to `.env` and add your API keys.

## Test Dataset

The repository includes two datasets:

1. A small test set (in `data/test_set/`) with 5 annual reports and questions
2. The full ERC2 competition dataset (in `data/erc2_set/`) with all competition questions and reports

Each dataset directory contains its own README with specific setup instructions and available files. You can use either dataset to:

- Study example questions, reports, and system outputs
- Run the pipeline from scratch using provided PDFs

## Worker Mode (Asynchronous Processing)

The system now supports asynchronous job processing via Redis queues for production deployment.

### Supported Job Types

1. **Document Parse & Index** (`doc_parse_index`)
   - Downloads PDF from S3/MinIO storage
   - Parses document and creates vector embeddings
   - Stores processed chunks and vectors back to storage

2. **Report Generation** (`report_generate`)
   - Generates DD reports from sections plan
   - Processes evidence and creates structured output
   - Stores final report JSON to storage

### Environment Variables

Required:
- `OPENAI_API_KEY` - OpenAI API key
- `REDIS_URL` - Redis connection URL (e.g., `redis://localhost:6379`)
- `STORAGE_ENDPOINT_URL` - S3/MinIO endpoint
- `STORAGE_ACCESS_KEY` - Storage access key
- `STORAGE_SECRET_KEY` - Storage secret key

Optional:
- `STORAGE_BUCKET_NAME` - Bucket name (default: `ddkit-documents`)
- `QUEUE_DOC_PARSE_INDEX` - Queue name for doc processing (default: `ddkit:doc_parse_index`)
- `QUEUE_REPORT_GENERATE` - Queue name for report generation (default: `ddkit:report_generate`)
- `LOG_LEVEL` - Logging level (default: `INFO`)
- `JOB_CALLBACK_URL` - HTTP endpoint for job completion callbacks

### Running the Worker

```bash
# Start worker (processes jobs from Redis queues)
python main.py worker

# Health check
python main.py health-check

# Health check with JSON output
python main.py health-check --format json
```

### Docker Deployment

```bash
# Build and run with docker-compose
docker-compose up --build

# Or run manually
docker build -t ddkit-worker .
docker run -e OPENAI_API_KEY=your_key -e REDIS_URL=redis://host:6379 ddkit-worker
```

### Job Payload Examples

**Document Parse Job:**
```json
{
  "job_type": "doc_parse_index",
  "tenant_id": "demo",
  "case_id": "c1",
  "doc_id": "d1",
  "doc_kind": "epar",
  "s3_rendered_pdf_key": "tenants/demo/cases/c1/documents/d1/rendered/document.pdf"
}
```

**Report Generate Job:**
```json
{
  "job_type": "report_generate",
  "tenant_id": "demo",
  "case_id": "c1",
  "sections_plan_key": "tenants/demo/cases/c1/sections/plan.json"
}
```
- Use pre-processed data to skip directly to specific pipeline stages

See the respective README files for detailed dataset contents and setup instructions:
- `data/test_set/README.md` - For the small test dataset
- `data/erc2_set/README.md` - For the full competition dataset

## Usage

You can run any part of pipeline by uncommenting the method you want to run in `src/pipeline.py` and executing:
```bash
python .\src\pipeline.py
```

You can also run any pipeline stage using `main.py`, but you need to run it from the directory containing your data:
```bash
cd .\data\test_set\
python ..\..\main.py process-questions --config max_nst_o3m
```

### CLI Commands

Get help on available commands:
```bash
python main.py --help
```

Available commands:
- `download-models` - Download required docling models
- `parse-pdfs` - Parse PDF reports with parallel processing options
- `serialize-tables` - Process tables in parsed reports
- `process-reports` - Run the full pipeline on parsed reports
- `process-questions` - Process questions using specified config

Each command has its own options. For example:
```bash
python main.py parse-pdfs --help
# Shows options like --parallel/--sequential, --chunk-size, --max-workers

python main.py process-reports --config ser_tab
# Process reports with serialized tables config
```

## Some configs

- `max_nst_o3m` - Best performing config using OpenAI's o3-mini model
- `ibm_llama70b` - Alternative using IBM's Llama 70B model
- `gemini_thinking` - Full context answering with using enormous context window of Gemini. It is not RAG, actually

Check `pipeline.py` for more configs and detils on them.

## DD Kit Production Features (v2)

This repository has been extended with production-ready DD (Due Diligence) Kit features for pharmaceutical document analysis:

### New Components

- **StorageClient**: S3/MinIO integration for document storage
- **DocumentLoader**: Batch document ingestion with metadata
- **EvidenceCandidatesBuilder**: Deterministic evidence extraction
- **ValidationGates**: Evidence linkage validation
- **DDSectionAnswerSchema**: Unified structured output schema
- **Hierarchical Chunking**: Parent/child relationships with typed chunks

### DD-Specific Features

- **Evidence-First Architecture**: Every claim links to source evidence
- **Multi-Jurisdictional Support**: RU/EU/US regulatory document handling
- **Conflict Resolution**: Handles contradictory information across sources
- **Tenant/Case Isolation**: Strict data separation for multi-tenant usage
- **Report Orchestration**: Automated multi-section report generation

### CLI Commands

#### Document Ingestion
```bash
# Ingest from manifest
python main.py ingest-documents --manifest documents_manifest_example.json

# Ingest single document
python main.py ingest-document --tenant-id demo_tenant --case-id rivaroxaban_dd_2024 --doc-id epar_eu_2024 --s3-rendered-pdf-key "tenants/demo_tenant/cases/rivaroxaban_dd_2024/documents/epar_eu_2024/rendered/document.pdf" --doc-kind epar --title "EPAR Rivaroxaban"
```

#### Report Generation
```bash
# Generate DD report
python main.py generate-dd-report --case-id rivaroxaban_dd_2024 --sections-plan sections_plan_example.json --output dd_report.json
```

### Configuration

Set these environment variables in `.env`:

```bash
# Storage (S3/MinIO)
STORAGE_ENDPOINT_URL=https://minio.example.com
STORAGE_ACCESS_KEY=your_access_key
STORAGE_SECRET_KEY=your_secret_key
STORAGE_BUCKET_NAME=ddkit-documents
STORAGE_REGION=us-east-1
STORAGE_USE_SSL=true

# OpenAI (existing)
OPENAI_API_KEY=sk-...
```

### Example Files

- `documents_manifest_example.json` - Sample document manifest
- `sections_plan_example.json` - Sample report sections plan
- `dd_report.json` - Generated report output format

### Architecture Overview

```
Documents (S3/MinIO) → DocumentLoader → PDFParser → TextSplitter (hierarchical)
                                                            ↓
Vector Index (per tenant/case) ← Chunking → BM25 Index
                                                            ↓
Retriever → EvidenceCandidatesBuilder → LLM (DDSectionAnswerPrompt)
                                                            ↓
ValidationGates → ReportAssembler → report.json (for UI)
```

## License

MIT