# Module 3: GenAI Policy Support Assistant (`/support_assistant`)

# Overview & Architecture

The Zepto GenAI Policy Support Assistant is a Retrieval-Augmented Generation (RAG) service designed to answer customer support policy queries using grounded context from official documents.

```text
[Customer Query]
       │
       ▼
┌──────────────────────┐
│  classify_intent     │ ──(Keyword Heuristic Router)
└──────────────────────┘
       │
       ├─────────────────────────────────┐
       │ (policy_question)               │ (general_question)
       ▼                                 ▼
┌──────────────────────┐      ┌──────────────────────┐
│ retrieve_and_answer  │      │    direct_answer     │
│ (ChromaDB Vector DB) │      │ (Canned Text Output) │
└──────────────────────┘      └──────────────────────┘
       │                                 │
       └────────────────┬────────────────┘
                        ▼
            ┌──────────────────────┐
            │   SupportResponse    │
            │  (Pydantic Schema)   │
            └──────────────────────┘