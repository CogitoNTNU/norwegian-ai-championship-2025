# RAG Evaluation System

A comprehensive evaluation framework for comparing different RAG (Retrieval-Augmented Generation) templates using RAGAS metrics and additional performance indicators.

## Features

- **Multiple RAG Templates**: Compare Simple RAG, Hybrid RAG (with BM25), and HyDE implementations
- **RAGAS Metrics**: Context Precision, Context Recall, Faithfulness, Answer Relevancy
- **Additional Metrics**: NDCG, Precision@K, Recall@K, F1 Score, Response Time
- **Visualization**: Radar charts and bar charts for comparative analysis
- **Statistical Testing**: T-tests for significance testing
- **Comprehensive Reporting**: Ranking and recommendations

## Setup

### 1. Install Dependencies

```bash
uv sync
```

### 2. Environment Configuration

Copy the example environment file and configure your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your Google API key:

```
GOOGLE_API_KEY=your_actual_google_api_key_here
```

### 3. Prepare Test Data

Ensure you have a `testset.json` file in the root directory with the following structure:

```json
[
  {
    "user_input": "Your question here",
    "reference": "Expected answer",
    "reference_contexts": ["Context 1", "Context 2", ...]
  }
]
```

## Usage

Run the test-set generation:

```bash
python src/testset-generation/testset_generator.py
```

Run the evaluation:

```bash
uv run python src/evaluation/main.py
```

The script will:

1. Load your test dataset
1. Execute all RAG templates on each question
1. Calculate RAGAS and additional metrics
1. Generate visualizations
1. Provide rankings and recommendations
1. Save results to `rag_evaluation_results.json`

## RAG Templates

### Simple RAG

- Basic vector similarity search
- Uses Chroma vector database
- Single retriever approach

### Hybrid RAG

- Combines vector similarity and BM25 keyword search
- Uses MergerRetriever to combine results
- Better for diverse query types

### HyDE (Hypothetical Document Embeddings)

- Generates hypothetical answers first
- Uses hypothetical answer for retrieval
- Better for complex reasoning tasks

## Metrics Explained

### RAGAS Metrics

- **Context Precision**: How relevant are the retrieved contexts?
- **Context Recall**: How much of the relevant context was retrieved?
- **Faithfulness**: How factually accurate is the generated answer?
- **Answer Relevancy**: How relevant is the answer to the question?

### Additional Metrics

- **NDCG**: Normalized Discounted Cumulative Gain
- **Precision@K**: Precision at top K results
- **Recall@K**: Recall at top K results
- **F1 Score**: Harmonic mean of precision and recall
- **Response Time**: Average response time per query

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Install all packages from `requirements.txt`
1. **API Key Issues**: Ensure `GOOGLE_API_KEY` is set in `.env`
1. **Import Errors**: Make sure you're using the latest versions of langchain packages
1. **Memory Issues**: Reduce batch size or use smaller test datasets

### Environment Variables

The script automatically handles environment variables:

- `USER_AGENT`: Set automatically if not provided
- LangSmith tracing is disabled by default (no API keys needed)

## Output

The evaluation generates:

- Console output with rankings and analysis
- Interactive visualizations (radar charts, bar charts)
- `rag_evaluation_results.json` with detailed results
- Statistical significance tests

## Customization

### Adding New RAG Templates

1. Create a new template class in the `templates/` directory
1. Implement the `run(question, context)` method
1. Add it to the `templates` dictionary in `rag_evaluation.py`

### Modifying Metrics

You can adjust the ranking weights in the `rank_templates()` function:

```python
def rank_templates(
    template_ragas_scores,
    template_additional_scores,
    recall_weight=0.6,
    precision_weight=0.4,
):
    # Your ranking logic here
    pass
```

### Custom Evaluation Criteria

Modify the `provide_recommendations()` function to change the recommendation logic based on your specific use case requirements.

## Files Structure

```
├── rag_evaluation.py          # Main evaluation script
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
├── testset.json              # Test dataset
├── templates/                # RAG template implementations
│   ├── simple_rag.py
│   ├── hybrid_rag.py
│   └── hyde.py
└── README.md                 # This file
```

## Contributing

1. Fork the repository
1. Create a feature branch
1. Add your improvements
1. Test thoroughly
1. Submit a pull request

## License

This project is open source and available under the MIT License.
