# 🎙️ Meeting AI — Backend API & Processing Pipeline

The central processing engine powering the **AI Meeting Intelligence Platform**. This service handles secure audio/video ingestion, high-speed transcription, speaker identification, LLM-based structured analytics, and vector-based semantic search (RAG).

---

## ⚡ Tech Stack

- **API Framework:** FastAPI (Python 3.10+)
- **Speech-to-Text:** Faster-Whisper (CUDA / CPU optimized)
- **Speaker Diarization:** Pyannote.audio
- **LLM & RAG Framework:** LangChain / LlamaIndex + OpenAI GPT-4o
- **Database & Vector Store:** PostgreSQL + `pgvector` extension
- **Task Management:** FastAPI BackgroundTasks / Celery

---

## 🗺️ Processing Pipeline Architecture

```text
Audio File Upload
        │
        ▼
FFmpeg Cleanup
        │
        ▼
Faster-Whisper (STT + Timestamps)
        │
        ▼
Pyannote Diarization (Speaker ID)
        │
        ▼
Pgvector Embedding
        │
        ▼
LLM Extraction
        │
        ├──► Summaries
        ├──► Action Items
        ├──► Decisions
        └──► Sentiment Analysis
        │
        ▼
JSON Response
