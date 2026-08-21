# Intelligent Support System — LangGraph Flow

```mermaid
flowchart TD
    A[Ticket In<br/>Synthetic Ticket Queue] --> B[Ticket Intake Node<br/>Initialize State + Thread Memory]

    B --> C[Sentiment & Classification Agent]
    C --> C1[VADER Sentiment Tool<br/>Local / No LLM Tokens]
    C1 --> C2[Ticket Category Classification<br/>technical / billing / account / general / feature_request]

    C2 --> D[Policy Agent]
    D --> D1[Retrieve Policy / FAQ Chunks from Chroma]
    D1 --> D2{Relevant Policy Found?}

    D2 -- No --> E1[Escalate<br/>Do Not Fabricate Policy]
    D2 -- Yes --> D3{Safety Policy Match?}

    D3 -- Refund Abuse --> E2[Refuse<br/>Scripted Polite Response]
    D3 -- Abusive Content Request --> E2
    D3 -- No Prohibited Request --> F[Department Router]

    F --> G[RAG Agent]
    G --> G1[Build Retrieval Query]
    G1 --> G2[Chroma Similarity Search]
    G2 --> G3[Top-K Policy / FAQ Chunks]
    G3 --> G4[RAG Answer Draft<br/>Grounded in KB Only]

    G4 --> H[LangGraph Triage Agent]
    H --> I{Route Decision}

    I -->|Auto-Resolve| I1[Auto-Resolve Draft]
    I -->|Escalate| I2[Escalation Draft]
    I -->|Refuse| I3[Scripted Refusal Draft]
    I -->|Ask More Info| I4[Ask-for-Information Draft]

    E1 --> I2
    E2 --> I3

    I1 --> J[Groundedness & Confidence Evaluator]
    I2 --> J
    I3 --> J
    I4 --> J

    J --> K{Grounded + Confident?}
    K -- Yes --> L[HITL Approval Gate]
    K -- No --> M{More Retrieval Could Help?}

    M -- Yes --> N{Refinement Attempts<br/>Below Limit?}
    N -- Yes --> O[Retrieval Query Refiner]
    O --> G
    N -- No --> P[Force Escalation]
    M -- No --> P
    P --> L

    L --> Q{Human Decision}
    Q -->|Approve| R[Approved Draft<br/>Never Auto-Sent]
    Q -->|Edit| S[Human-Edited Draft]
    S --> J
    Q -->|Reject| T{Revision Cycles<br/>Below Limit?}
    T -- Yes --> U[Revision Agent<br/>Use Human Feedback + KB]
    U --> J
    T -- No --> V[Escalate for Manual Handling]
    V --> W[Audit Log]
    R --> W

    W --> X[(SQLite Audit DB)]
    W --> Y[END]

    B -. thread_id .-> Z[(LangGraph Checkpointer<br/>Conversation Memory)]
    Z -. restores thread state .-> B

    AA[(Markdown Policy / FAQ KB)] --> AB[Document Loader + Chunker]
    AB --> AC[Local Hashing Embeddings]
    AC --> AD[(Chroma Vector Store)]
    AD --> D1
    AD --> G2
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