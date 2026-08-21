# Intelligent Support System — LangGraph Flow

```mermaid
flowchart TD
    A[Ticket In] --> B[Sentiment + Policy Check]

    B --> C[RAG Answer Draft]

    C --> D[Route Decision]

    D -->|Auto-Resolve| E[Prepare Draft]
    D -->|Escalate| E
    D -->|Refuse| E
    D -->|Ask More Info| E

    E --> F[Confidence / Groundedness Check]

    F -->|Pass| G[Human Approval]

    F -->|Fail| H[Refine Retrieval]
    H --> C

    G -->|Approve| I[Audit Log]
    G -->|Edit / Reject| J[Revise Draft]
    J --> F

    I --> K[END]
```

## End-to-End Flow

```text
Ticket In
   ↓
Ticket Intake + Thread Memory
   ↓
VADER Sentiment Tool
   ↓
Ticket Classification
   ↓
Policy Check
   ↓
RAG Retrieval + Draft
   ↓
LangGraph Route Decision
   ├── Auto-Resolve
   ├── Escalate
   ├── Refuse
   └── Ask More Information
   ↓
Groundedness / Confidence Check
   ├── PASS → HITL
   └── FAIL
         ↓
      Retrieval Refinement
         ↓
      RAG Again
         ↓
      Confidence Re-check
         ↓
      Force Escalation if retry limit reached
   ↓
Human Approval
   ├── Approve → Audit
   ├── Edit → Grounding Re-check → HITL
   └── Reject → Revision → Grounding Re-check → HITL
   ↓
SQLite Audit Log
   ↓
END
```

## Safety Mapping

| Condition | Action |
|---|---|
| Relevant policy/FAQ is not found | Escalate rather than fabricate |
| Refund-abuse request is supported by refund policy | Refuse with scripted response |
| Abusive-content request is supported by abusive-content policy | Refuse with scripted response |
| Ticket needs non-sensitive information | Ask for more information |
| KB fully supports resolution | Auto-resolve candidate |
| Groundedness/confidence is too low | Refine retrieval |
| Retrieval refinement limit is reached | Force escalation |
| Any customer-facing reply | Draft only; never auto-sent |
| Human edits a draft | Re-run groundedness evaluation |
| Human rejects a draft | Revise using feedback, then re-evaluate |
| Final human decision | Write an idempotent audit event |

## RAG Flow

```mermaid
flowchart LR
    A[knowledge_base/*.md] --> B[Document Loader]
    B --> C[Recursive Text Splitter]
    C --> D[Local Hashing Embeddings]
    D --> E[(Chroma DB)]
    F[Ticket / Refined Query] --> G[Similarity Search]
    E --> G
    G --> H[Top-K Chunks]
    H --> I[RAG Prompt]
    I --> J[Grounded Draft]
```

## Memory Flow

```mermaid
flowchart LR
    A[Customer / Ticket] --> B[thread_id = customer_id]
    B --> C[(LangGraph Checkpointer)]
    C --> D[SupportState.messages]
    D --> E[RAG Agent Recent Conversation Context]
    F[HITL Interrupt] --> C
    C --> G[Command resume]
    G --> H[Continue Same Graph Execution]
```

## Main LangGraph Nodes

```text
START
  ↓
ticket_in
  ↓
sentiment_and_classification
  ↓
policy_check
  ↓
department_router
  ↓
rag_answer_draft
  ↓
route_decision
  ├─ auto_resolve
  ├─ escalate
  ├─ refuse
  └─ ask_more_info
       ↓
groundedness_evaluator
  ├─ hitl_approval
  ├─ refine_retrieval ─────→ rag_answer_draft
  └─ force_escalation ─────→ hitl_approval

hitl_approval
  ├─ approve ───────────────→ audit_log
  ├─ edit ──────────────────→ groundedness_evaluator
  └─ reject ────────────────→ revise_response
                                  ↓
                           groundedness_evaluator

audit_log
  ↓
END
```
## Setup
Example:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
copy .env.template .env
```

Edit `.env` in the project root:

```text
OPENAI_API_KEY=...
OPENAI_MODEL=...
```

## Run the interactive queue

```bat
python main.py
```

## Test in notebook

Open:

`notebooks/langgrpah_flow_demo.ipynb`

Run the cells in order.
```