#!/bin/bash

echo "🚀 Starting BM25s Healthcare RAG API Server..."
echo "📍 Running on http://0.0.0.0:8000"
echo "🔗 Prediction endpoint: http://0.0.0.0:8000/predict"
echo ""

# Start the server
cd /Users/nybruker/Documents/nm-ai/norwegian-ai-championship-2025/src/rag
uv run python api.py
