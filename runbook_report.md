# DDKit RAG Worker - Runbook

## 1. Архитектура системы

```
┌─────────────────────┐     ┌─────────────┐     ┌─────────────────────────┐
│   API Gateway (Go)  │────▶│    Redis    │◀────│    RAG Worker (Python)  │
│   ddkit_handler.go  │     │   Queues    │     │    main.py worker       │
└─────────────────────┘     └─────────────┘     └─────────────────────────┘
                                   │                       │
                                   │                       ▼
                                   │            ┌─────────────────────────┐
                                   │            │      MinIO / S3         │
                                   └───────────▶│   Document Storage      │
                                                └─────────────────────────┘
```

### Компоненты:
1. **API Gateway** (Go) - принимает HTTP-запросы, создает jobs в Redis
2. **Redis** - очереди задач: `ddkit:doc_parse_index`, `ddkit:report_generate`, `ddkit:case_view_generate`
3. **RAG Worker** (Python) - обрабатывает задачи из очередей
4. **MinIO/S3** - хранилище PDF и JSON документов

## 2. Переменные окружения

### Обязательные:
```bash
# OpenAI API
OPENAI_API_KEY=sk-proj-...

# Redis
REDIS_URL=redis://localhost:6379

# База данных (для метаданных)
DDKIT_DB_DSN=postgresql://user:password@localhost:5432/pharm_search

# Хранилище документов (S3/MinIO)
STORAGE_ENDPOINT_URL=http://localhost:9000
STORAGE_ACCESS_KEY=minioadmin
STORAGE_SECRET_KEY=minioadmin
STORAGE_BUCKET_NAME=ddkit
STORAGE_USE_SSL=false
```

### Опциональные:
```bash
# Worker mode: all | doc_parse_index | report_generate | case_view_generate
WORKER_MODE=all
WORKER_CONCURRENCY=2

# Таймауты
DDKIT_LLM_TIMEOUT_SECONDS=120
JOB_TIMEOUT_SECONDS=900

# LLM провайдер и модель
DDKIT_LLM_PROVIDER=openai          # openai | anthropic | gemini
DDKIT_ANSWER_MODEL=gpt-4o          # конкретная модель (опционально)

# Embeddings
EMBEDDINGS_BATCH_SIZE=128
EMBEDDINGS_MAX_CONCURRENCY=4

# Chunking
CHUNK_SIZE_TOKENS=800
CHUNK_OVERLAP_TOKENS=100
CHUNK_DEDUP=true

# Docling
DOCLING_DO_OCR=false
DOCLING_DO_TABLES=false
DOCLING_DO_PICTURES=false

# Логирование
LOG_LEVEL=INFO

# Очереди (обычно не меняются)
QUEUE_DOC_PARSE_INDEX=ddkit:doc_parse_index
QUEUE_REPORT_GENERATE=ddkit:report_generate
QUEUE_CASE_VIEW_GENERATE=ddkit:case_view_generate

# Callback URL для уведомлений о завершении job'ов
JOB_CALLBACK_URL=http://api-gateway:8085/ddkit/jobs/callback
DDKIT_CALLBACK_TOKEN=<secret>
```

## 3. Запуск системы

### 3.1 Docker Compose (рекомендуемый)

```bash
cd C:\GItHub\Vector_db\RAG-Challenge-2

# Запуск всех сервисов
docker-compose up -d

# Просмотр логов
docker-compose logs -f rag-worker
```

Сервисы:
- `redis` - Redis на порту 6385
- `minio` - MinIO на портах 9002 (API) и 9003 (Console)
- `rag-worker` - Универсальный воркер (все очереди)
- `rag-worker-index` - Воркер для парсинга документов (WORKER_MODE=doc_parse_index)
- `rag-worker-report` - Воркер для генерации отчетов (WORKER_MODE=report_generate)

### 3.2 Локальный запуск (разработка)

```bash
cd C:\GItHub\Vector_db\RAG-Challenge-2

# Создание .env файла
cp .env.example .env
# Заполнить OPENAI_API_KEY

# Установка зависимостей
pip install -r requirements.txt

# Загрузка моделей Docling (однократно)
python main.py download-models

# Запуск воркера
python main.py worker
```

### 3.3 CLI команды

```bash
# Health check
python main.py health-check

# Парсинг PDF (без Redis, локально)
python main.py parse-pdfs --parallel --max-workers 10

# Генерация DD отчета (без Redis, локально)
python main.py generate-dd-report \
    --case-id "test_case_001" \
    --sections-plan sections_plan.json \
    --output dd_report.json

# Инджест одного документа
python main.py ingest-document \
    --tenant-id tenant1 \
    --case-id case123 \
    --doc-id doc001 \
    --s3-rendered-pdf-key "documents/doc001.pdf" \
    --doc-kind "EPAR" \
    --title "Assessment Report"

# Инджест документов из манифеста
python main.py ingest-documents --manifest documents_manifest.json
```

## 4. Pipeline обработки

### 4.1 Полный цикл создания кейса

```
1. POST /ddkit/cases              → Создает кейс в БД
2. POST /ddkit/cases/{id}/dossier:build → Запускает сбор документов
3. Job: doc_parse_index           → Парсит PDF → JSON, индексирует в Vector DB
4. POST /ddkit/cases/{id}/case_view:generate → Генерирует case_view (структурированный)
5. POST /ddkit/cases/{id}/report:generate    → Генерирует DD report (Q&A)
```

### 4.2 Структура Job'а в Redis

```json
{
  "job_id": "uuid",
  "job_type": "doc_parse_index | report_generate | case_view_generate",
  "tenant_id": "tenant_001",
  "case_id": "case_001",
  "status": "pending | processing | completed | failed",
  "created_at": "2024-01-15T10:00:00Z",

  // Для doc_parse_index:
  "doc_id": "doc_001",
  "doc_kind": "EPAR | FDA_LABEL | GRLS_PDF | PATENT | ...",
  "s3_rendered_pdf_key": "documents/tenant/case/doc.pdf",

  // Для report_generate:
  "sections_plan": { ... },
  "s3_output_key": "reports/tenant/case/dd_report.json"
}
```

## 5. Мониторинг и отладка

### 5.1 Проверка здоровья системы

```bash
python main.py health-check --format json
```

Вывод:
```json
{
  "healthy": true,
  "checks": {
    "redis": {"healthy": true},
    "storage": {"healthy": true},
    "embeddings": {"healthy": true}
  }
}
```

### 5.2 Просмотр очередей Redis

```bash
redis-cli -p 6379

# Количество задач в очереди
LLEN ddkit:doc_parse_index
LLEN ddkit:report_generate
LLEN ddkit:case_view_generate

# Просмотр задачи
LRANGE ddkit:report_generate 0 0
```

### 5.3 Логи

Worker логирует:
- Начало/конец обработки job'а
- Время retrieval и LLM вызовов
- Количество найденных passages и evidence candidates
- Ошибки валидации evidence_ids

```
2024-01-15 10:30:00 - INFO - Starting job doc_parse_index: doc_001
2024-01-15 10:30:05 - INFO - Parsed 45 pages, 312 chunks
2024-01-15 10:30:10 - INFO - Indexed 312 chunks to vector DB
2024-01-15 10:30:10 - INFO - Job completed in 10.2s
```

## 6. Типичные проблемы

### 6.1 Пустые секции в отчете

**Симптом:** claims=0 для секции patents или clinical_trials

**Причины:**
1. `answer_type: "table"` в sections_plan (LLM не может сконвертировать в claims)
2. Нет документов с нужным `doc_kind` в кейсе
3. Retrieval не находит релевантные passages

**Решение:**
- Проверить `answer_type` в ddkit_handler.go (должно быть `"facts"`)
- Проверить что документы проиндексированы с правильным doc_kind
- Проверить логи retrieval ("Retrieved 0 passages")

### 6.2 Evidence validation errors

**Симптом:** `Invalid evidence_ids: [ev_xxx]`

**Причина:** LLM сгенерировал evidence_id, которого нет в candidates

**Решение:** Система автоматически перемещает такие claims в unknowns через ValidationGates

### 6.3 Timeout при генерации

**Симптом:** `TimeoutError: report_timeout at after_llm:patents`

**Решение:**
- Увеличить `DDKIT_LLM_TIMEOUT_SECONDS`
- Увеличить `JOB_TIMEOUT_SECONDS`
- Уменьшить количество вопросов в sections_plan

## 7. Структура файлов

```
RAG-Challenge-2/
├── main.py                      # CLI точка входа
├── docker-compose.yml           # Docker конфигурация
├── requirements.txt             # Python зависимости
├── .env                         # Переменные окружения
├── src/
│   ├── worker.py                # DDKitWorker - обработка очередей
│   ├── job_processors.py        # Процессоры для каждого типа job
│   ├── dd_report_generator.py   # Генератор DD отчета (Q&A)
│   ├── case_view_v2_generator.py# Генератор case_view (структурированный)
│   ├── retrieval.py             # HybridRetriever (vector + reranking)
│   ├── evidence_builder.py      # EvidenceCandidatesBuilder
│   ├── validation_gates.py      # Валидация evidence references
│   ├── prompts.py               # LLM промпты и схемы
│   ├── api_requests.py          # APIProcessor (OpenAI/Anthropic/Gemini)
│   └── storage_client.py        # S3/MinIO клиент
└── data/
    ├── documents/               # Parsed JSON документы
    └── vector_db/               # ChromaDB индекс
```

## 8. API Gateway интеграция

API Gateway (Go) создает задачи так:

```go
// ddkit_handler.go

// Генерация report
jobData := map[string]interface{}{
    "job_type":      "report_generate",
    "tenant_id":     tenantID,
    "case_id":       caseID,
    "sections_plan": sectionsPlan,
    "s3_output_key": fmt.Sprintf("reports/%s/%s/dd_report.json", tenantID, caseID),
}
redis.LPush("ddkit:report_generate", json.Marshal(jobData))

// Генерация case_view
jobData := map[string]interface{}{
    "job_type":   "case_view_generate",
    "tenant_id":  tenantID,
    "case_id":    caseID,
    "inn":        inn,
    "query":      query,
    "use_web":    true,
    "use_snapshot": true,
    "snapshot":   dossierSnapshot,  // данные из FrontendSnapshotStore
}
redis.LPush("ddkit:case_view_generate", json.Marshal(jobData))
```

## 9. Выходные артефакты

Система генерирует два типа артефактов:

1. **DD Report** (`dd_report.json`) - Q&A формат для аналитиков
   - Секции с вопросами и ответами
   - Claims с evidence references
   - См. `output_artefact_dd_report.md`

2. **Case View** (`case_view.json`) - Структурированный формат для UI
   - Паспорт препарата
   - Регистрация по регионам
   - Клинические исследования
   - Патенты
   - См. `output_artefact_case_view.md`
