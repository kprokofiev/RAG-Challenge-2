import click
from pathlib import Path
import logging
import json
import sys
from src.pipeline import Pipeline, configs, preprocess_configs
from src.worker import DDKitWorker
from src.settings import settings

@click.group()
def cli():
    """Pipeline command line interface for processing PDF reports and questions."""
    pass

@cli.command()
def download_models():
    """Download required docling models."""
    click.echo("Downloading docling models...")
    Pipeline.download_docling_models()

@cli.command()
@click.option('--parallel/--sequential', default=True, help='Run parsing in parallel or sequential mode')
@click.option('--chunk-size', default=2, help='Number of PDFs to process in each worker')
@click.option('--max-workers', default=10, help='Number of parallel worker processes')
def parse_pdfs(parallel, chunk_size, max_workers):
    """Parse PDF reports with optional parallel processing."""
    root_path = Path.cwd()
    pipeline = Pipeline(root_path)
    
    click.echo(f"Parsing PDFs (parallel={parallel}, chunk_size={chunk_size}, max_workers={max_workers})")
    pipeline.parse_pdf_reports(parallel=parallel, chunk_size=chunk_size, max_workers=max_workers)

@cli.command()
@click.option('--max-workers', default=10, help='Number of workers for table serialization')
def serialize_tables(max_workers):
    """Serialize tables in parsed reports using parallel threading."""
    root_path = Path.cwd()
    pipeline = Pipeline(root_path)
    
    click.echo(f"Serializing tables (max_workers={max_workers})...")
    pipeline.serialize_tables(max_workers=max_workers)

@cli.command()
@click.option('--config', type=click.Choice(['ser_tab', 'no_ser_tab']), default='no_ser_tab', help='Configuration preset to use')
def process_reports(config):
    """Process parsed reports through the pipeline stages."""
    root_path = Path.cwd()
    run_config = preprocess_configs[config]
    pipeline = Pipeline(root_path, run_config=run_config)
    
    click.echo(f"Processing parsed reports (config={config})...")
    pipeline.process_parsed_reports()

@cli.command()
@click.option('--config', type=click.Choice(['base', 'pdr', 'max', 'max_no_ser_tab', 'max_nst_o3m', 'max_st_o3m', 'ibm_llama70b', 'ibm_llama8b', 'gemini_thinking']), default='base', help='Configuration preset to use')
def process_questions(config):
    """Process questions using the pipeline."""
    root_path = Path.cwd()
    run_config = configs[config]
    pipeline = Pipeline(root_path, run_config=run_config)

    click.echo(f"Processing questions (config={config})...")
    pipeline.process_questions()

@cli.command()
@click.option('--manifest', required=True, help='Path to documents manifest JSON file')
def ingest_documents(manifest):
    """Ingest documents from manifest into the system."""
    import json
    from src.job_processors import DocParseIndexProcessor

    click.echo(f"Ingesting documents from manifest: {manifest}")

    # Load manifest
    with open(manifest, 'r', encoding='utf-8') as f:
        manifest_data = json.load(f)

    tenant_id = manifest_data.get('tenant_id')
    case_id = manifest_data.get('case_id')

    click.echo(f"Processing tenant: {tenant_id}, case: {case_id}")

    processor = DocParseIndexProcessor()
    documents = manifest_data.get('documents', [])
    click.echo(f"Found {len(documents)} documents to process")

    for doc in documents:
        click.echo(f"  - {doc['doc_id']}: {doc.get('title', 'Untitled')}")
        job_data = {
            "job_type": "doc_parse_index",
            "tenant_id": tenant_id,
            "case_id": case_id,
            "doc_id": doc["doc_id"],
            "doc_kind": doc.get("doc_kind", "unknown"),
            "title": doc.get("title", ""),
            "source_url": doc.get("source_url", ""),
            "s3_rendered_pdf_key": doc["s3_rendered_pdf_key"],
            "s3_parsed_json_key": doc.get("s3_parsed_json_key")
        }
        if not processor.process_job(job_data):
            click.echo(f"Failed to ingest document {doc['doc_id']}")

    click.echo("Ingestion completed")

@cli.command()
@click.option('--tenant-id', required=True, help='Tenant identifier')
@click.option('--case-id', required=True, help='Case identifier')
@click.option('--doc-id', required=True, help='Document identifier')
@click.option('--s3-rendered-pdf-key', required=True, help='S3 key for rendered PDF')
@click.option('--doc-kind', default='unknown', help='Document type/kind')
@click.option('--title', help='Document title')
def ingest_document(tenant_id, case_id, doc_id, s3_rendered_pdf_key, doc_kind, title):
    """Ingest a single document."""
    from src.job_processors import DocParseIndexProcessor
    click.echo(f"Ingesting document {doc_id} for tenant {tenant_id}, case {case_id}")
    click.echo(f"S3 key: {s3_rendered_pdf_key}")
    click.echo(f"Type: {doc_kind}, Title: {title or 'Untitled'}")

    processor = DocParseIndexProcessor()
    job_data = {
        "job_type": "doc_parse_index",
        "tenant_id": tenant_id,
        "case_id": case_id,
        "doc_id": doc_id,
        "doc_kind": doc_kind,
        "title": title or "",
        "s3_rendered_pdf_key": s3_rendered_pdf_key
    }
    if not processor.process_job(job_data):
        click.echo("Single document ingestion failed")
        sys.exit(1)
    click.echo("Single document ingestion completed")

@cli.command()
@click.option('--case-id', required=True, help='Case identifier')
@click.option('--sections-plan', required=True, help='Path to sections plan JSON file')
@click.option('--output', default='dd_report.json', help='Output report file path')
def generate_dd_report(case_id, sections_plan, output):
    """Generate DD report for a case."""
    import json

    click.echo(f"Generating DD report for case {case_id}")
    click.echo(f"Sections plan: {sections_plan}")
    click.echo(f"Output: {output}")

    # Load sections plan
    with open(sections_plan, 'r', encoding='utf-8') as f:
        sections_data = json.load(f)

    click.echo(f"Loaded plan with {len(sections_data)} sections")

    pipeline = Pipeline(Path.cwd())
    pipeline.generate_dd_report(case_id, sections_data, output)
    click.echo(f"Report saved to {output}")

@cli.command()
def worker():
    """Start the DD Kit RAG worker to process jobs from Redis queues."""
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    click.echo("Starting DD Kit RAG worker...")
    click.echo(f"Redis URL: {settings.redis_url}")
    click.echo(f"Doc parse queue: {settings.queue_doc_parse_index}")
    click.echo(f"Report generate queue: {settings.queue_report_generate}")

    try:
        worker_instance = DDKitWorker()
        worker_instance.start()
    except KeyboardInterrupt:
        click.echo("Worker stopped by user")
    except Exception as e:
        click.echo(f"Worker failed: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--format', type=click.Choice(['json', 'text']), default='text', help='Output format')
def health_check(format):
    """Perform health check and exit with appropriate status code."""
    # Configure logging to ERROR only for health checks
    logging.basicConfig(level=logging.ERROR)

    try:
        worker_instance = DDKitWorker()
        health_status = worker_instance.check_health()
        overall_healthy = bool(health_status.get("healthy", False))

        if format == 'json':
            click.echo(json.dumps(health_status, indent=2))
        else:
            click.echo("DD Kit RAG Worker Health Check")
            click.echo("=" * 40)

            click.echo(f"Overall Status: {'✓ HEALTHY' if overall_healthy else '✗ UNHEALTHY'}")

            for check_name, check_data in health_status.get("checks", {}).items():
                status = "✓" if check_data.get("healthy", False) else "✗"
                error = check_data.get("error", "")
                click.echo(f"{check_name.upper()}: {status} {error}")

        # Exit with appropriate code
        sys.exit(0 if overall_healthy else 1)

    except Exception as e:
        if format == 'json':
            click.echo(json.dumps({
                "healthy": False,
                "error": str(e)
            }, indent=2))
        else:
            click.echo(f"Health check failed: {e}")

        sys.exit(1)

if __name__ == '__main__':
    cli()
