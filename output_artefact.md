# DDKit Output Artifacts - Точная структура JSON

## 1. DD Report (dd_report.json)

Генерируется через `DDReportGenerator.generate_report()`.
Формат: Q&A - ответы на вопросы из sections_plan.

### 1.1 Корневая структура

```json
{
  "report_id": "dd_report_case123_1705312800",
  "case_id": "case123",
  "generated_at": "2024-01-15T10:00:00Z",
  "sections": [ /* Section[] */ ],
  "evidence_index": [ /* Evidence[] */ ],
  "documents": [ /* DocumentMeta[] */ ]
}
```

### 1.2 Section

```json
{
  "section_id": "patents",
  "title": "Патенты",
  "claims": [
    {
      "id": "pat_blocking_1",
      "text": "Основной патент на субстанцию US8456789 действует до 2028 года",
      "evidence_ids": ["ev_patent_doc1_p5_abc123"],
      "confidence": 0.92
    }
  ],
  "numbers": [
    {
      "id": "num_patent_count",
      "label": "Количество патентных семейств",
      "value": 7,
      "as_reported": "7 patent families",
      "unit": null,
      "currency": null,
      "scale": null,
      "period": null,
      "as_of_date": "2024-01",
      "evidence_ids": ["ev_patent_summary_p1_def456"]
    }
  ],
  "risks": [
    {
      "id": "risk_patent_expiry",
      "title": "Истечение патентной защиты в 2025",
      "severity": "high",
      "description": "Патент на способ лечения US7654321 истекает в марте 2025, что открывает возможность для дженериков",
      "evidence_ids": ["ev_patent_doc2_p12_ghi789"]
    }
  ],
  "unknowns": [
    {
      "id": "unk_patent_litigation",
      "question": "Есть ли текущие патентные споры?",
      "reason": "No supporting evidence in provided candidates"
    }
  ],
  "evidence": [
    {
      "id": "ev_patent_doc1_p5_abc123",
      "doc_id": "patent_analysis_2024",
      "doc_title": "Patent Landscape Analysis",
      "page": 5,
      "snippet": "The primary compound patent US8456789 expires in 2028..."
    }
  ]
}
```

### 1.3 Полная типизация

```typescript
// DD Report Root
interface DDReport {
  report_id: string;           // "dd_report_{case_id}_{timestamp}"
  case_id: string;
  generated_at: string;        // ISO 8601: "2024-01-15T10:00:00Z"
  sections: Section[];
  evidence_index: Evidence[];  // Глобальный индекс всех evidence
  documents: DocumentMeta[];   // Метаданные документов кейса
}

// Section
interface Section {
  section_id: string;          // "passport" | "regulatory" | "clinical_trials" | "patents" | ...
  title: string;               // "Паспорт препарата"
  claims: Claim[];
  numbers: NumberWithEvidence[];
  risks: Risk[];
  unknowns: Unknown[];
  evidence: Evidence[];        // Evidence, использованный в этой секции
}

// Claim - основной блок с утверждением
interface Claim {
  id: string;                  // Уникальный ID в рамках секции
  text: string;                // Текст утверждения для отчета
  evidence_ids: string[];      // Ссылки на Evidence
  confidence: number | null;   // 0.0 - 1.0, опционально
}

// NumberWithEvidence - числовое значение с контекстом
interface NumberWithEvidence {
  id: string;
  label: string;               // "Количество пациентов в исследовании"
  value: number;               // Нормализованное значение
  as_reported: string;         // "1,234 patients" - как в документе
  unit: string | null;         // "mg", "ml", "%"
  currency: string | null;     // "USD", "EUR"
  scale: string | null;        // "units" | "thousands" | "millions"
  period: string | null;       // "Q1 2024", "FY2023"
  as_of_date: string | null;   // "2024-01"
  evidence_ids: string[];
}

// Risk - риск/красный флаг
interface Risk {
  id: string;
  title: string;               // Короткое название
  severity: "high" | "medium" | "low" | "unknown";
  description: string;         // Детальное описание
  evidence_ids: string[];
}

// Unknown - нерешенный вопрос
interface Unknown {
  id: string;
  question: string;            // Исходный вопрос
  reason: string;              // Почему не удалось ответить
}

// Evidence - источник/доказательство
interface Evidence {
  id: string;                  // "ev_{doc_id}_{page}_{hash}"
  doc_id: string;              // ID документа
  doc_title: string | null;    // Название документа
  page: number;                // Номер страницы
  snippet: string;             // Текст сниппета (200-400 chars)
}

// DocumentMeta - метаданные документа
interface DocumentMeta {
  doc_id: string;
  title: string;
  kind: string | null;         // "EPAR" | "FDA_LABEL" | "PATENT" | ...
  source_url: string | null;
}
```

---

## 2. Case View (case_view.json)

Генерируется через `CaseViewV2Generator.generate_case_view()`.
Формат: структурированный для UI, с фактами и evidence.

### 2.1 Корневая структура

```json
{
  "schema_version": "2.0",
  "case_id": "case123",
  "query": "Ривароксабан",
  "inn_normalized": "ривароксабан",
  "generated_at": "2024-01-15T10:00:00Z",
  "passport": { /* Passport */ },
  "sections": {
    "brief": { /* Brief */ },
    "regulatory": { /* Regulatory */ },
    "clinical": { /* Clinical */ },
    "patents": { /* Patents */ },
    "synthesis": { /* Synthesis */ },
    "sources": { /* Sources */ },
    "unknowns": { /* Unknowns */ }
  },
  "source_stats": { /* SourceStats */ }
}
```

### 2.2 Passport (Паспорт препарата)

```json
{
  "inn": {
    "label": "МНН",
    "value": "Ривароксабан",
    "citations": [{"source": "snapshot", "path": "$.inn"}]
  },
  "trade_names": {
    "label": "Торговые названия",
    "value": {"RU": ["Ксарелто"], "EU": ["Xarelto"], "US": ["Xarelto"]},
    "citations": [
      {"source": "doc", "doc_id": "grls_card_001", "page": 1, "evidence_id": "ev_grls_1_abc"}
    ]
  },
  "fda_approval": {
    "label": "Одобрение FDA",
    "value": "Approved 2011-07-01 for DVT prophylaxis",
    "citations": [
      {"source": "doc", "doc_id": "fda_label_001", "page": 1, "evidence_id": "ev_fda_1_def"}
    ]
  },
  "registered_in": {
    "label": "Где зарегистрирован",
    "value": {"RU": "active", "EU": "active", "US": "active"},
    "citations": [{"source": "snapshot", "path": "$.regulatory"}]
  },
  "chemical_formula": {
    "label": "Химическая формула",
    "value": "C19H18ClN3O5S",
    "citations": [
      {"source": "doc", "doc_id": "epar_001", "page": 12, "evidence_id": "ev_epar_12_ghi"}
    ]
  },
  "drug_class": {
    "label": "Класс препарата",
    "value": "Factor Xa inhibitor",
    "citations": [
      {"source": "doc", "doc_id": "epar_001", "page": 5, "evidence_id": "ev_epar_5_jkl"}
    ]
  },
  "registration_holders": {
    "label": "Держатели регистрации",
    "value": {"RU": "Bayer AG", "EU": "Bayer AG", "US": "Janssen Pharmaceuticals"},
    "citations": [
      {"source": "doc", "doc_id": "grls_card_001", "page": 1, "evidence_id": "ev_grls_1_mno"}
    ]
  },
  "dosage_forms": {
    "label": "Формы выпуска",
    "value": ["tablets 10mg", "tablets 15mg", "tablets 20mg"],
    "citations": []
  },
  "data_quality": "ok"
}
```

### 2.3 Brief (Раздел "Кратко")

```json
{
  "summary": {
    "text": "Ривароксабан — пероральный ингибитор фактора Xa, одобрен для профилактики инсульта при ФП и лечения ТГВ/ТЭЛА. Основная патентная защита истекает в 2024 году.",
    "citations": []
  },
  "key_facts": [
    {
      "label": "Статус регистрации",
      "value": "Зарегистрирован в RU, EU, US",
      "citations": [{"source": "snapshot", "path": "$.regulatory"}]
    },
    {
      "label": "Основной результат эффективности",
      "value": "ROCKET-AF: non-inferior to warfarin for stroke prevention (HR 0.88)",
      "citations": [
        {"source": "doc", "doc_id": "clinical_study_001", "page": 8, "evidence_id": "ev_clinical_8_xyz"}
      ]
    },
    {
      "label": "Патентная стена",
      "value": "До 2024 года (основной патент на субстанцию)",
      "citations": [
        {"source": "doc", "doc_id": "patent_analysis", "page": 3, "evidence_id": "ev_patent_3_uvw"}
      ]
    }
  ],
  "data_quality": "ok"
}
```

### 2.4 Regulatory (Регистрация)

```json
{
  "ru": {
    "entries": [
      {
        "trade_name": "Ксарелто",
        "holder": "Bayer AG",
        "dosage_forms": ["таблетки 10мг", "таблетки 15мг", "таблетки 20мг"],
        "reg_number": "ЛП-001234",
        "reg_date": "2012-05-15",
        "citations": [
          {"source": "doc", "doc_id": "grls_card_001", "page": 1, "evidence_id": "ev_grls_001"}
        ]
      }
    ]
  },
  "eu": {
    "trade_names": {
      "value": ["Xarelto"],
      "evidence_ids": ["ev_epar_1_abc"]
    },
    "holders": {
      "value": ["Bayer AG"],
      "evidence_ids": ["ev_epar_1_def"]
    },
    "dosage_forms_and_strengths": {
      "value": ["Film-coated tablets: 10mg, 15mg, 20mg"],
      "evidence_ids": ["ev_epar_2_ghi"]
    },
    "status": {
      "value": "Authorised",
      "evidence_ids": ["ev_epar_1_jkl"]
    },
    "countries_covered": {
      "value": ["All EU/EEA member states"],
      "evidence_ids": ["ev_epar_1_mno"]
    }
  },
  "us": {
    "trade_names": {
      "value": ["Xarelto"],
      "evidence_ids": ["ev_fda_1_pqr"]
    },
    "holders": {
      "value": ["Janssen Pharmaceuticals, Inc."],
      "evidence_ids": ["ev_fda_1_stu"]
    },
    "dosage_forms_and_strengths": {
      "value": ["Tablets: 10mg, 15mg, 20mg, 2.5mg"],
      "evidence_ids": ["ev_fda_3_vwx"]
    },
    "status": {
      "value": "Approved",
      "evidence_ids": ["ev_fda_1_yza"]
    }
  },
  "instructions_highlights": {
    "indications": [
      {
        "value": "Профилактика инсульта и системной эмболии у пациентов с фибрилляцией предсердий",
        "evidence_ids": ["ev_smpc_4_abc"]
      },
      {
        "value": "Лечение тромбоза глубоких вен (ТГВ) и тромбоэмболии легочной артерии (ТЭЛА)",
        "evidence_ids": ["ev_smpc_4_def"]
      }
    ],
    "dosing": [
      {
        "value": "20 мг один раз в сутки во время еды",
        "evidence_ids": ["ev_smpc_8_ghi"]
      }
    ],
    "restrictions": [
      {
        "value": "Противопоказан при тяжелой почечной недостаточности (КлКр <15 мл/мин)",
        "evidence_ids": ["ev_smpc_12_jkl"]
      }
    ]
  },
  "data_quality": "ok"
}
```

### 2.5 Clinical (Клинические исследования)

```json
{
  "global": {
    "phase_3": [
      {
        "trial_id": "NCT00403767",
        "title": "ROCKET AF - Rivaroxaban vs Warfarin in AF",
        "phase": "Phase 3",
        "study_type": "Randomized, double-blind, active-controlled",
        "countries": ["US", "EU", "Asia"],
        "enrollment": "14264",
        "comparator": "Warfarin",
        "regimen": "Rivaroxaban 20mg once daily",
        "status": "Completed",
        "efficacy_key_points": [
          "Primary endpoint: stroke/systemic embolism HR 0.88 (95% CI 0.74-1.03)",
          "Non-inferior to warfarin (p<0.001 for non-inferiority)"
        ],
        "conclusion": "Rivaroxaban was non-inferior to warfarin for prevention of stroke",
        "where_conducted": "Multi-center, international",
        "evidence_ids": ["ev_clinical_nct00403767_p1", "ev_clinical_nct00403767_p8"]
      }
    ],
    "phase_2": [],
    "phase_1": []
  },
  "ru": {
    "phase_3": [],
    "phase_4": []
  },
  "ongoing": {
    "recruiting": [],
    "active_not_recruiting": []
  },
  "pubmed": {
    "comparative": [
      {
        "title": "Meta-analysis of NOACs vs Warfarin",
        "summary": "Rivaroxaban showed similar efficacy to other NOACs...",
        "evidence_ids": ["ev_pubmed_001"]
      }
    ],
    "abstracts": [],
    "real_world": [],
    "combination": []
  },
  "data_quality": "partial"
}
```

### 2.6 Patents (Патенты) - УЛЬТРА-ФОКУС

```json
{
  "blocking_families": [
    {
      "family_id": "fam_US8456789",
      "representative": "US8456789",
      "priority_date": "2004-05-15",
      "coverage_type": {
        "value": ["composition"],
        "evidence_ids": ["ev_patent_1_abc"]
      },
      "summary": {
        "value": "Covers the rivaroxaban compound and crystalline forms",
        "evidence_ids": ["ev_patent_1_def"]
      },
      "countries_covered": ["US", "EP", "JP", "CN", "RU"],
      "expiry_by_country": {
        "US": "2024-07-15",
        "EP": "2024-05-15",
        "JP": "2024-05-15",
        "RU": "2024-05-15"
      },
      "holders": ["Bayer AG"],
      "is_blocking": true,
      "citations": [
        {"source": "doc", "doc_id": "patent_doc_001", "page": 1, "evidence_id": "ev_patent_1_ghi"}
      ]
    }
  ],
  "families": [
    {
      "family_id": "fam_US7851456",
      "representative": "US7851456",
      "priority_date": "2006-08-20",
      "coverage_type": {
        "value": ["treatment"],
        "evidence_ids": ["ev_patent_2_abc"]
      },
      "summary": {
        "value": "Method of treating thromboembolic disorders",
        "evidence_ids": ["ev_patent_2_def"]
      },
      "key_claims": [
        {
          "value": "Claim 1: A method of preventing stroke comprising administering rivaroxaban...",
          "evidence_ids": ["ev_patent_2_ghi"]
        }
      ],
      "countries_covered": ["US", "EP"],
      "expiry_by_country": {
        "US": "2026-08-20",
        "EP": "2026-08-20"
      },
      "holders": ["Bayer AG"],
      "is_blocking": false,
      "jurisdiction_statuses": [
        {
          "jurisdiction": "US",
          "status": "Granted",
          "event_date": "2010-12-14",
          "publication_number": "US7851456B2",
          "evidence_ids": ["ev_patent_2_jkl"]
        }
      ],
      "citations": []
    }
  ],
  "views": {
    "blocking": [/* Патенты с is_blocking=true */],
    "composition": [/* Патенты с coverage_type содержащим "composition" */],
    "treatment": [/* Патенты с coverage_type содержащим "treatment" */],
    "synthesis": [/* Патенты с coverage_type содержащим "synthesis" */]
  },
  "patent_wall_until": "2024-07-15",
  "data_quality": "partial"
}
```

### 2.7 Synthesis (Синтез)

```json
{
  "synthesis_route": {
    "steps": [
      {
        "text": "Step 1: Condensation of 4-chloroaniline with morpholine...",
        "evidence_ids": ["ev_synth_patent_p5_abc"]
      },
      {
        "text": "Step 2: Ring formation via cyclization...",
        "evidence_ids": ["ev_synth_patent_p6_def"]
      }
    ],
    "source_patents": ["US8456789", "WO2005123456"]
  },
  "treatment_method_from_patents": {
    "value": "Administration of 20mg once daily with food for stroke prevention",
    "evidence_ids": ["ev_patent_treatment_p12_ghi"]
  },
  "data_quality": "partial"
}
```

### 2.8 Sources (Источники)

```json
{
  "documents": [
    {
      "doc_id": "epar_rivaroxaban_2024",
      "title": "Xarelto EPAR - Product Information",
      "kind": "EPAR",
      "region": "EU",
      "year": 2024,
      "source_url": "https://www.ema.europa.eu/...",
      "linked_trials": [],
      "linked_patent_families": []
    },
    {
      "doc_id": "patent_US8456789",
      "title": "Substituted oxazolidinones and their use...",
      "kind": "PATENT",
      "region": null,
      "year": 2004,
      "source_url": "https://patents.google.com/...",
      "linked_trials": [],
      "linked_patent_families": ["fam_US8456789"]
    }
  ],
  "by_kind": {
    "EPAR": ["epar_rivaroxaban_2024"],
    "FDA_LABEL": ["fda_label_xarelto"],
    "GRLS_PDF": ["grls_xarelto_001"],
    "PATENT": ["patent_US8456789", "patent_US7851456"],
    "CLINICAL": ["clinical_rocket_af"]
  },
  "by_region": {
    "EU": ["epar_rivaroxaban_2024"],
    "US": ["fda_label_xarelto"],
    "RU": ["grls_xarelto_001"]
  },
  "data_quality": "ok"
}
```

### 2.9 Unknowns (Неизвестное)

```json
{
  "gaps": [
    {
      "field": "patents.synthesis",
      "reason": "No synthesis patents found in indexed documents",
      "suggested_sources": ["patent_database", "synthesis_literature"]
    },
    {
      "field": "clinical.ru.phase_4",
      "reason": "No Russian post-marketing studies found",
      "suggested_sources": ["grls_clinical", "pubmed_ru"]
    }
  ],
  "data_quality": "partial"
}
```

### 2.10 Source Stats

```json
{
  "facts_total": 47,
  "facts_with_evidence": 42,
  "gaps_total": 5,
  "documents_used": 12,
  "ready_for_ui": true,
  "quality_gates": {
    "passport_complete": true,
    "has_regulatory_data": true,
    "has_clinical_data": true,
    "has_patent_data": true,
    "evidence_coverage": 0.89
  }
}
```

---

## 3. Полная типизация Case View (TypeScript)

```typescript
interface CaseView {
  schema_version: "2.0";
  case_id: string;
  query: string;
  inn_normalized: string;
  generated_at: string;  // ISO 8601
  passport: Passport;
  sections: {
    brief: Brief;
    regulatory: Regulatory;
    clinical: Clinical;
    patents: Patents;
    synthesis: Synthesis;
    sources: Sources;
    unknowns: Unknowns;
  };
  source_stats: SourceStats;
}

// Базовые типы
interface Citation {
  source: "snapshot" | "doc";
  path?: string;           // для snapshot: "$.inn"
  doc_id?: string;         // для doc
  page?: number;           // для doc
  evidence_id?: string;    // для doc
}

interface Fact {
  label: string;
  value: string | string[] | Record<string, string | string[]>;
  citations: Citation[];
}

interface EvidenceLockedValue {
  value: string | string[] | null;
  evidence_ids: string[];
  note?: string;
}

// Passport
interface Passport {
  inn?: Fact;
  trade_names?: Fact;
  fda_approval?: Fact;
  registered_in?: Fact;
  chemical_formula?: Fact;
  drug_class?: Fact;
  registration_holders?: Fact;
  dosage_forms?: Fact;
  data_quality: "ok" | "partial" | "empty";
}

// Brief
interface Brief {
  summary: {
    text: string;
    citations: Citation[];
  };
  key_facts: Array<{
    label: string;
    value: string;
    citations: Citation[];
  }>;
  data_quality: "ok" | "partial" | "empty";
}

// Regulatory
interface Regulatory {
  ru?: {
    entries: Array<{
      trade_name: string;
      holder: string;
      dosage_forms: string[];
      reg_number?: string;
      reg_date?: string;
      citations: Citation[];
    }>;
  };
  eu?: RegulatoryMarket;
  us?: RegulatoryMarket;
  instructions_highlights?: {
    indications: EvidenceLockedValue[];
    dosing: EvidenceLockedValue[];
    restrictions: EvidenceLockedValue[];
  };
  data_quality: "ok" | "partial" | "empty";
}

interface RegulatoryMarket {
  trade_names?: EvidenceLockedValue;
  holders?: EvidenceLockedValue;
  dosage_forms_and_strengths?: EvidenceLockedValue;
  status?: EvidenceLockedValue;
  countries_covered?: EvidenceLockedValue;
}

// Clinical
interface Clinical {
  global: TrialsBucket;
  ru: TrialsBucket;
  ongoing: {
    recruiting: TrialCard[];
    active_not_recruiting: TrialCard[];
  };
  pubmed: {
    comparative: PublicationItem[];
    abstracts: PublicationItem[];
    real_world: PublicationItem[];
    combination: PublicationItem[];
  };
  data_quality: "ok" | "partial" | "empty";
}

interface TrialsBucket {
  phase_3?: TrialCard[];
  phase_2?: TrialCard[];
  phase_1?: TrialCard[];
  phase_4?: TrialCard[];
}

interface TrialCard {
  trial_id: string;
  title: string;
  phase: string;
  study_type: string;
  countries: string[];
  enrollment?: string;
  comparator?: string;
  regimen?: string;
  status?: string;
  efficacy_key_points: string[];
  conclusion?: string;
  where_conducted?: string;
  evidence_ids: string[];
}

interface PublicationItem {
  title: string;
  summary: string;
  evidence_ids: string[];
}

// Patents
interface Patents {
  blocking_families: PatentFamily[];
  families: PatentFamily[];
  views: {
    blocking: PatentFamily[];
    composition: PatentFamily[];
    treatment: PatentFamily[];
    synthesis: PatentFamily[];
  };
  patent_wall_until?: string;
  data_quality: "ok" | "partial" | "empty";
}

interface PatentFamily {
  family_id: string;
  representative: string;
  priority_date?: string;
  coverage_type?: EvidenceLockedValue;
  summary?: EvidenceLockedValue;
  key_claims?: EvidenceLockedValue[];
  countries_covered: string[];
  expiry_by_country: Record<string, string>;
  holders: string[];
  is_blocking: boolean;
  jurisdiction_statuses?: JurisdictionStatus[];
  citations: Citation[];
}

interface JurisdictionStatus {
  jurisdiction: string;
  status: string;
  event_date?: string;
  publication_number?: string;
  evidence_ids: string[];
}

// Synthesis
interface Synthesis {
  synthesis_route?: {
    steps: Array<{
      text: string;
      evidence_ids: string[];
    }>;
    source_patents?: string[];
  };
  treatment_method_from_patents?: EvidenceLockedValue;
  data_quality: "ok" | "partial" | "empty";
}

// Sources
interface Sources {
  documents: DocumentEntry[];
  by_kind: Record<string, string[]>;
  by_region: Record<string, string[]>;
  data_quality: "ok" | "partial" | "empty";
}

interface DocumentEntry {
  doc_id: string;
  title: string;
  kind: string;
  region?: string;
  year?: number;
  source_url?: string;
  linked_trials: string[];
  linked_patent_families: string[];
}

// Unknowns
interface Unknowns {
  gaps: Array<{
    field: string;
    reason: string;
    suggested_sources: string[];
  }>;
  data_quality: "ok" | "partial" | "empty";
}

// Source Stats
interface SourceStats {
  facts_total: number;
  facts_with_evidence: number;
  gaps_total: number;
  documents_used: number;
  ready_for_ui: boolean;
  quality_gates: {
    passport_complete: boolean;
    has_regulatory_data: boolean;
    has_clinical_data: boolean;
    has_patent_data: boolean;
    evidence_coverage: number;  // 0.0 - 1.0
  };
}
```

---

## 4. Связь структур с требованиями (structure_for_report.md)

| Требование | DD Report | Case View |
|------------|-----------|-----------|
| Паспорт препарата | sections[0] (passport) | passport |
| Кратко (сводка) | sections[1] (brief) | sections.brief |
| Регистрация | sections[2] (regulatory) | sections.regulatory |
| Клинические исследования | sections[3] (clinical_trials) | sections.clinical |
| Патенты (50% фокус) | sections[4] (patents) | sections.patents |
| Синтез | sections[5] (synthesis) | sections.synthesis |
| Источники | documents + evidence_index | sections.sources |
| Неизвестное | unknowns в каждой секции | sections.unknowns |
| Панель доказательств | evidence_index | citations везде |

---

## 5. Пример минимального валидного ответа

### DD Report (минимум):
```json
{
  "report_id": "dd_report_case1_1705312800",
  "case_id": "case1",
  "generated_at": "2024-01-15T10:00:00Z",
  "sections": [],
  "evidence_index": [],
  "documents": []
}
```

### Case View (минимум):
```json
{
  "schema_version": "2.0",
  "case_id": "case1",
  "query": "",
  "inn_normalized": "",
  "generated_at": "2024-01-15T10:00:00Z",
  "passport": {"data_quality": "empty"},
  "sections": {
    "brief": {"summary": {"text": "", "citations": []}, "key_facts": [], "data_quality": "empty"},
    "regulatory": {"data_quality": "empty"},
    "clinical": {"global": {}, "ru": {}, "ongoing": {}, "pubmed": {}, "data_quality": "empty"},
    "patents": {"blocking_families": [], "families": [], "views": {}, "data_quality": "empty"},
    "synthesis": {"data_quality": "empty"},
    "sources": {"documents": [], "by_kind": {}, "by_region": {}, "data_quality": "empty"},
    "unknowns": {"gaps": [], "data_quality": "ok"}
  },
  "source_stats": {
    "facts_total": 0,
    "facts_with_evidence": 0,
    "gaps_total": 0,
    "documents_used": 0,
    "ready_for_ui": false,
    "quality_gates": {}
  }
}
```
