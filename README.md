End-to-End Flow

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
