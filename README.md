## 🎥 YouTube Video Summarizer & QA Bot 🤖

This project extracts transcripts from YouTube videos and allows users to ask questions or generate summaries using an open-source LLM (TinyLlama-1.1B-Chat-v1.0) and semantic search with vector embeddings.

📌 Features
✅ Extracts video captions/transcripts using YouTubeTranscriptApi

✅ Splits long transcripts into manageable chunks

✅ Uses FAISS to store and retrieve relevant transcript segments

✅ Powered by TinyLlama for fast and lightweight text generation

✅ Enables custom question-answering and summarization

✅ Embedding with sentence-transformers/all-MiniLM-L6-v2


🧠 Tech Stack
YouTubeTranscriptApi – for transcript extraction

LangChain – modular LLM framework

FAISS – fast vector similarity search

TinyLlama – open-source LLM for lightweight inference

HuggingFace Embeddings – for semantic understanding of transcript chunks
