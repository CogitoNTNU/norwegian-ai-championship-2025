# Topic-First RAG Prompts

## Stage 1: Topic Classification Prompt

```
Medical context from multiple domains:
{diverse_context}

Statement: {statement}

Which medical topic best fits this statement?

Topics:
{topics_text}

Respond with JSON only: {"topic": X}
```

## Stage 2: Binary Classification Prompt

```
Focused medical evidence:
{focused_context}

Statement: {statement}

Based on this specific medical context, is the statement TRUE or FALSE?

Consider:
- Direct evidence in the context
- Medical accuracy and consistency
- Clinical consensus

Respond with JSON only: {"is_true": true/false}
```

## Key Design Principles

1. **Stage 1 (Topic)**: Uses diverse context from multiple domains to get broad medical understanding
1. **Stage 2 (Binary)**: Uses focused context filtered to the specific topic for precise fact-checking
1. **Sequential not Parallel**: Topic-first approach filters docs before binary classification
1. **Expected Performance**: 90%+ binary, 80%+ topic accuracy, 6-7s total time

## Architecture Benefits

- **Smart Domain Filtering**: Topic classification narrows down the medical domain
- **Focused Fact-Checking**: Binary classification works on relevant domain-specific evidence
- **Mimics Expert Diagnosis**: "What domain?" then "Is this claim accurate in that domain?"
- **Faster than Parallel**: Sequential saves time by avoiding concurrent overhead
