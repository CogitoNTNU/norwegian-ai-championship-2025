# Emergency Healthcare RAG - Exploratory Data Analysis Report

**Generated on:** eda.py
**Dataset:** Norwegian AI Championship 2025 - Emergency Healthcare RAG Track

## Executive Summary

This report provides a comprehensive analysis of the training dataset for the Emergency Healthcare RAG challenge. The dataset contains 630 medical statements with corresponding truth labels and topic classifications across 115 different medical topics.

## Dataset Overview

### Basic Statistics

- **Total Samples:** 630
- **Data Consistency:** Consistent
- **Unique Topics Covered:** 115 out of 115 (100.0% coverage)

### Statement Characteristics

#### Length Analysis (Characters)

- **Minimum:** 96 characters
- **Maximum:** 375 characters
- **Mean:** 172.7 characters
- **Median:** 161.0 characters
- **Standard Deviation:** 46.3 characters

#### Word Count Analysis

- **Minimum:** 13 words
- **Maximum:** 55 words
- **Mean:** 24.5 words
- **Median:** 23.0 words
- **Standard Deviation:** 6.5 words

## Class Balance Analysis

### Truth Value Distribution

- **True Statements:** 336 (53.3%)
- **False Statements:** 294 (46.7%)
- **Balance Ratio (True:False):** 1.14:1

### Reasonably Balanced Classes

The classes are reasonably balanced, which is good for model training and evaluation.

## Topic Distribution Analysis

### Coverage Statistics

- **Topics with Data:** 115 topics
- **Topics without Data:** 0 topics
- **Coverage Percentage:** 100.0%

### Topic Frequency Statistics

- **Most Common Topic:** 9 statements (Hypertensive Emergency)
- **Least Common Topic:** 4 statements (Penetrating Trauma)
- **Average per Topic:** 5.5 statements
- **Median per Topic:** 6.0 statements

### Top 10 Topics by Frequency

| Rank | Topic                              | Count | Percentage |
| ---- | ---------------------------------- | ----- | ---------- |
| 1    | Hypertensive Emergency             | 9     | 1.4%       |
| 2    | Testicular Torsion                 | 6     | 1.0%       |
| 3    | Embolism                           | 6     | 1.0%       |
| 4    | Pulmonary Hypertension             | 6     | 1.0%       |
| 5    | Toxicology Screen                  | 6     | 1.0%       |
| 6    | Overdose_Poisoning                 | 6     | 1.0%       |
| 7    | Pulmonary Embolism                 | 6     | 1.0%       |
| 8    | Urinalysis                         | 6     | 1.0%       |
| 9    | Takotsubo Cardiomyopathy           | 6     | 1.0%       |
| 10   | Pulse Oximetry_Arterial Saturation | 6     | 1.0%       |

### Bottom 10 Topics by Frequency

| Rank | Topic                     | Count | Percentage |
| ---- | ------------------------- | ----- | ---------- |
| 106  | Hyperventilation Syndrome | 4     | 0.6%       |
| 107  | Anaphylaxis               | 4     | 0.6%       |
| 108  | Burns                     | 4     | 0.6%       |
| 109  | Pericarditis              | 4     | 0.6%       |
| 110  | Multi-organ Failure       | 4     | 0.6%       |
| 111  | Bronchoscopy with BAL     | 4     | 0.6%       |
| 112  | Acute Kidney Injury       | 4     | 0.6%       |
| 113  | Syncope                   | 4     | 0.6%       |
| 114  | Cardiac Tamponade         | 4     | 0.6%       |
| 115  | Penetrating Trauma        | 4     | 0.6%       |

## Topic-Truth Correlation Analysis

### Topics with Highest True Statement Ratios

| Topic                                        | Total | True | False | True % |
| -------------------------------------------- | ----- | ---- | ----- | ------ |
| Acute Coronary Syndrome                      | 5     | 3    | 2     | 60.0%  |
| CT Other                                     | 5     | 3    | 2     | 60.0%  |
| Complete Blood Count (CBC) with differential | 5     | 3    | 2     | 60.0%  |
| Hyponatremia_Hypernatremia                   | 5     | 3    | 2     | 60.0%  |
| Creatine Phosphokinase                       | 5     | 3    | 2     | 60.0%  |
| Perforated Viscus                            | 5     | 3    | 2     | 60.0%  |
| Cardiac Arrest                               | 5     | 3    | 2     | 60.0%  |
| Ovarian Torsion                              | 5     | 3    | 2     | 60.0%  |
| Unstable Angina                              | 5     | 3    | 2     | 60.0%  |
| Chest Pain (non-cardiac)                     | 5     | 3    | 2     | 60.0%  |

### Topics with Highest False Statement Ratios

| Topic                              | Total | True | False | False % |
| ---------------------------------- | ----- | ---- | ----- | ------- |
| Meningitis                         | 5     | 2    | 3     | 60.0%   |
| Testicular Torsion                 | 6     | 3    | 3     | 50.0%   |
| Embolism                           | 6     | 3    | 3     | 50.0%   |
| Pulmonary Hypertension             | 6     | 3    | 3     | 50.0%   |
| Toxicology Screen                  | 6     | 3    | 3     | 50.0%   |
| Overdose_Poisoning                 | 6     | 3    | 3     | 50.0%   |
| Pulmonary Embolism                 | 6     | 3    | 3     | 50.0%   |
| Urinalysis                         | 6     | 3    | 3     | 50.0%   |
| Takotsubo Cardiomyopathy           | 6     | 3    | 3     | 50.0%   |
| Pulse Oximetry_Arterial Saturation | 6     | 3    | 3     | 50.0%   |

## Topic Gap Analysis

### Missing Topics (0 topics)

Topics with zero training statements:

| Topic ID | Topic Name | Total | True | False |
| -------- | ---------- | ----- | ---- | ----- |

### Sparse Topics (0 topics)

Topics with fewer than 4 training statements:

| Topic ID | Topic Name | Total | True | False |
| -------- | ---------- | ----- | ---- | ----- |

### One-Sided Topics (0 topics)

Topics where all statements are either true or false:

| Topic ID | Topic Name | Total | True | False |
| -------- | ---------- | ----- | ---- | ----- |

## Data Quality Observations

### Potential Issues

- No significant data quality issues identified

## Generation Needs

Full CSV at data/eda/topic_needs.csv

**Target:** 5 total statements per topic, with at least 2 true and 2 false statements each.

**Summary:**

- Topics needing additional data: 10 out of 115
- Total true statements to generate: 10
- Total false statements to generate: 0
- Total additional statements needed: 10

### Preview: First 15 Topics with Generation Needs

| Topic ID | Topic Name                | Needed True | Needed False |
| -------- | ------------------------- | ----------- | ------------ |
| 5        | Acute Kidney Injury       | 1           | 0            |
| 9        | Anaphylaxis               | 1           | 0            |
| 20       | Burns                     | 1           | 0            |
| 24       | Cardiac Tamponade         | 1           | 0            |
| 41       | Hyperventilation Syndrome | 1           | 0            |
| 50       | Multi-organ Failure       | 1           | 0            |
| 55       | Penetrating Trauma        | 1           | 0            |
| 57       | Pericarditis              | 1           | 0            |
| 76       | Syncope                   | 1           | 0            |
| 88       | Bronchoscopy with BAL     | 1           | 0            |

*This analysis was generated automatically by the EDA pipeline. For questions or issues, refer to the source code in `src/rag/eda.py`.*
