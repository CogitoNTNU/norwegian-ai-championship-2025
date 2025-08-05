#!/bin/bash

echo "🚀 Starting BM25s Healthcare RAG API Server..."
echo "📍 Running on http://0.0.0.0:8000"
echo "🔗 Prediction endpoint: http://0.0.0.0:8000/predict"
echo ""

# Start the server
cd "$(dirname "$0")"
uv run python api.py
