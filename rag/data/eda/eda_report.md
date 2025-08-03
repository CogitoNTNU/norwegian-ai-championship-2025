# Emergency Healthcare RAG - Exploratory Data Analysis Report

**Generated on:** eda.py 
**Dataset:** Norwegian AI Championship 2025 - Emergency Healthcare RAG Track

## Executive Summary

This report provides a comprehensive analysis of the training dataset for the Emergency Healthcare RAG challenge. The dataset contains 200 medical statements with corresponding truth labels and topic classifications across 99 different medical topics.

## Dataset Overview

### Basic Statistics
- **Total Samples:** 200
- **Data Consistency:** Consistent
- **Unique Topics Covered:** 99 out of 115 (86.1% coverage)

### Statement Characteristics

#### Length Analysis (Characters)
- **Minimum:** 133 characters
- **Maximum:** 375 characters  
- **Mean:** 225.2 characters
- **Median:** 221.5 characters
- **Standard Deviation:** 41.6 characters

#### Word Count Analysis
- **Minimum:** 17 words
- **Maximum:** 55 words
- **Mean:** 31.6 words
- **Median:** 31.0 words
- **Standard Deviation:** 6.4 words

## Class Balance Analysis

### Truth Value Distribution
- **True Statements:** 107 (53.5%)
- **False Statements:** 93 (46.5%)
- **Balance Ratio (True:False):** 1.15:1

###  Reasonably Balanced Classes

The classes are reasonably balanced, which is good for model training and evaluation.

## Topic Distribution Analysis

### Coverage Statistics
- **Topics with Data:** 99 topics
- **Topics without Data:** 16 topics
- **Coverage Percentage:** 86.1%

### Topic Frequency Statistics
- **Most Common Topic:** 6 statements (Hypertensive Emergency)
- **Least Common Topic:** 1 statements (Mitral Regurgitation)
- **Average per Topic:** 2.0 statements
- **Median per Topic:** 2.0 statements

### Top 10 Topics by Frequency

| Rank | Topic | Count | Percentage |
|------|--------|--------|-----------|
| 1 | Hypertensive Emergency | 6 | 3.0% |
| 2 | Pulmonary Hypertension | 5 | 2.5% |
| 3 | Hemoglobin A1C | 5 | 2.5% |
| 4 | Pulmonary Embolism | 4 | 2.0% |
| 5 | Pulse Oximetry_Arterial Saturation | 4 | 2.0% |
| 6 | COPD Exacerbation | 4 | 2.0% |
| 7 | Acute Abdomen | 4 | 2.0% |
| 8 | Cervical Spine Injury | 4 | 2.0% |
| 9 | Meningitis | 4 | 2.0% |
| 10 | Acute Coronary Syndrome | 3 | 1.5% |

### Bottom 10 Topics by Frequency

| Rank | Topic | Count | Percentage |
|------|--------|--------|-----------|
| 90 | Seizures_Status Epilepticus | 1 | 0.5% |
| 91 | Interstitial Lung Disease | 1 | 0.5% |
| 92 | Encephalitis | 1 | 0.5% |
| 93 | Sleep Apnea | 1 | 0.5% |
| 94 | Procalcitonin | 1 | 0.5% |
| 95 | Cardiac Catheterization | 1 | 0.5% |
| 96 | Hypothermia_Hyperthermia | 1 | 0.5% |
| 97 | Arterial Blood Gas (ABG) | 1 | 0.5% |
| 98 | Acute Myocardial Infarction (STEMI_NSTEMI) | 1 | 0.5% |
| 99 | Mitral Regurgitation | 1 | 0.5% |

## Topic-Truth Correlation Analysis

### Topics with Highest True Statement Ratios

| Topic | Total | True | False | True % |
|--------|--------|--------|--------|---------|
| Ruptured AAA | 3 | 3 | 0 | 100.0% |
| Compartment Syndrome | 3 | 3 | 0 | 100.0% |
| Diabetic Ketoacidosis | 3 | 3 | 0 | 100.0% |
| Toxicology Screen | 2 | 2 | 0 | 100.0% |
| Pneumomediastinum | 2 | 2 | 0 | 100.0% |
| CT Angiogram | 2 | 2 | 0 | 100.0% |
| Lactate | 2 | 2 | 0 | 100.0% |
| Acute Liver Failure | 2 | 2 | 0 | 100.0% |
| Central Venous Pressure | 2 | 2 | 0 | 100.0% |
| Testicular Torsion | 1 | 1 | 0 | 100.0% |

### Topics with Highest False Statement Ratios

| Topic | Total | True | False | False % |
|--------|--------|--------|--------|---------|
| Hyperventilation Syndrome | 2 | 0 | 2 | 100.0% |
| Creatine Phosphokinase | 2 | 0 | 2 | 100.0% |
| Takotsubo Cardiomyopathy | 2 | 0 | 2 | 100.0% |
| Cardiomyopathy | 2 | 0 | 2 | 100.0% |
| Blood Cultures | 2 | 0 | 2 | 100.0% |
| Lung Cancer | 2 | 0 | 2 | 100.0% |
| Acute Appendicitis | 2 | 0 | 2 | 100.0% |
| Rhabdomyolysis | 2 | 0 | 2 | 100.0% |
| Atrial Fibrillation | 2 | 0 | 2 | 100.0% |
| CT Other | 1 | 0 | 1 | 100.0% |

## Data Quality Observations

### Potential Issues
- **Missing Topics:** 16 topics have no training data
- **Low Sample Topics:** 71 topics have ≤2 statements



*This analysis was generated automatically by the EDA pipeline. For questions or issues, refer to the source code in `src/rag/eda.py`.*
