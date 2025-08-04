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

### Reasonably Balanced Classes

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

| Rank | Topic                              | Count | Percentage |
| ---- | ---------------------------------- | ----- | ---------- |
| 1    | Hypertensive Emergency             | 6     | 3.0%       |
| 2    | Pulmonary Hypertension             | 5     | 2.5%       |
| 3    | Hemoglobin A1C                     | 5     | 2.5%       |
| 4    | Pulmonary Embolism                 | 4     | 2.0%       |
| 5    | Pulse Oximetry_Arterial Saturation | 4     | 2.0%       |
| 6    | COPD Exacerbation                  | 4     | 2.0%       |
| 7    | Acute Abdomen                      | 4     | 2.0%       |
| 8    | Cervical Spine Injury              | 4     | 2.0%       |
| 9    | Meningitis                         | 4     | 2.0%       |
| 10   | Acute Coronary Syndrome            | 3     | 1.5%       |

### Bottom 10 Topics by Frequency

| Rank | Topic                                      | Count | Percentage |
| ---- | ------------------------------------------ | ----- | ---------- |
| 90   | Seizures_Status Epilepticus                | 1     | 0.5%       |
| 91   | Interstitial Lung Disease                  | 1     | 0.5%       |
| 92   | Encephalitis                               | 1     | 0.5%       |
| 93   | Sleep Apnea                                | 1     | 0.5%       |
| 94   | Procalcitonin                              | 1     | 0.5%       |
| 95   | Cardiac Catheterization                    | 1     | 0.5%       |
| 96   | Hypothermia_Hyperthermia                   | 1     | 0.5%       |
| 97   | Arterial Blood Gas (ABG)                   | 1     | 0.5%       |
| 98   | Acute Myocardial Infarction (STEMI_NSTEMI) | 1     | 0.5%       |
| 99   | Mitral Regurgitation                       | 1     | 0.5%       |

## Topic-Truth Correlation Analysis

### Topics with Highest True Statement Ratios

| Topic                   | Total | True | False | True % |
| ----------------------- | ----- | ---- | ----- | ------ |
| Ruptured AAA            | 3     | 3    | 0     | 100.0% |
| Compartment Syndrome    | 3     | 3    | 0     | 100.0% |
| Diabetic Ketoacidosis   | 3     | 3    | 0     | 100.0% |
| Toxicology Screen       | 2     | 2    | 0     | 100.0% |
| Pneumomediastinum       | 2     | 2    | 0     | 100.0% |
| CT Angiogram            | 2     | 2    | 0     | 100.0% |
| Lactate                 | 2     | 2    | 0     | 100.0% |
| Acute Liver Failure     | 2     | 2    | 0     | 100.0% |
| Central Venous Pressure | 2     | 2    | 0     | 100.0% |
| Testicular Torsion      | 1     | 1    | 0     | 100.0% |

### Topics with Highest False Statement Ratios

| Topic                     | Total | True | False | False % |
| ------------------------- | ----- | ---- | ----- | ------- |
| Hyperventilation Syndrome | 2     | 0    | 2     | 100.0%  |
| Creatine Phosphokinase    | 2     | 0    | 2     | 100.0%  |
| Takotsubo Cardiomyopathy  | 2     | 0    | 2     | 100.0%  |
| Cardiomyopathy            | 2     | 0    | 2     | 100.0%  |
| Blood Cultures            | 2     | 0    | 2     | 100.0%  |
| Lung Cancer               | 2     | 0    | 2     | 100.0%  |
| Acute Appendicitis        | 2     | 0    | 2     | 100.0%  |
| Rhabdomyolysis            | 2     | 0    | 2     | 100.0%  |
| Atrial Fibrillation       | 2     | 0    | 2     | 100.0%  |
| CT Other                  | 1     | 0    | 1     | 100.0%  |

## Topic Gap Analysis

### Missing Topics (16 topics)

Topics with zero training statements:

| Topic ID | Topic Name                        | Total | True | False |
| -------- | --------------------------------- | ----- | ---- | ----- |
| 13       | Aspiration Pneumonia              | 0     | 0    | 0     |
| 18       | Brain Death                       | 0     | 0    | 0     |
| 31       | Eclampsia                         | 0     | 0    | 0     |
| 37       | GI Bleeding                       | 0     | 0    | 0     |
| 42       | Hypoglycemia                      | 0     | 0    | 0     |
| 46       | Lung Abscess                      | 0     | 0    | 0     |
| 51       | Myocarditis                       | 0     | 0    | 0     |
| 54       | Pancreatitis                      | 0     | 0    | 0     |
| 66       | Respiratory Acidosis              | 0     | 0    | 0     |
| 72       | Sepsis_Septic Shock               | 0     | 0    | 0     |
| 75       | Stroke (Ischemic_Hemorrhagic)     | 0     | 0    | 0     |
| 89       | C-Reactive Protein (CRP)          | 0     | 0    | 0     |
| 95       | Coagulation Studies (PT_PTT_INR)  | 0     | 0    | 0     |
| 107      | Thyroid Stimulating Hormone (TSH) | 0     | 0    | 0     |
| 110      | Ultrasound Doppler (extremities)  | 0     | 0    | 0     |
| 114      | Urine Pregnancy Test              | 0     | 0    | 0     |

### Sparse Topics (33 topics)

Topics with fewer than 4 training statements:

| Topic ID | Topic Name                                   | Total | True | False |
| -------- | -------------------------------------------- | ----- | ---- | ----- |
| 3        | Acute Cholecystitis                          | 3     | 2    | 1     |
| 4        | Acute Coronary Syndrome                      | 3     | 1    | 2     |
| 5        | Acute Kidney Injury                          | 2     | 1    | 1     |
| 8        | Acute Respiratory Distress Syndrome          | 2     | 1    | 1     |
| 10       | Aortic Dissection                            | 2     | 1    | 1     |
| 11       | Aortic Stenosis                              | 2     | 1    | 1     |
| 12       | Arrhythmias (various)                        | 2     | 1    | 1     |
| 33       | Embolism                                     | 2     | 1    | 1     |
| 34       | Empyema                                      | 3     | 2    | 1     |
| 38       | Heart Failure (Acute_Chronic)                | 3     | 1    | 2     |
| 39       | Hemothorax                                   | 2     | 1    | 1     |
| 50       | Multi-organ Failure                          | 2     | 1    | 1     |
| 53       | Overdose_Poisoning                           | 2     | 1    | 1     |
| 55       | Penetrating Trauma                           | 3     | 1    | 2     |
| 56       | Perforated Viscus                            | 2     | 1    | 1     |
| 61       | Pneumonia (bacterial_viral_atypical)         | 3     | 2    | 1     |
| 62       | Pneumothorax                                 | 2     | 1    | 1     |
| 64       | Pulmonary Fibrosis                           | 2     | 1    | 1     |
| 67       | Respiratory Failure                          | 2     | 1    | 1     |
| 69       | Right Heart Failure                          | 2     | 1    | 1     |
| 73       | Shock (various types)                        | 2     | 1    | 1     |
| 76       | Syncope                                      | 3     | 2    | 1     |
| 79       | Traumatic Brain Injury                       | 3     | 2    | 1     |
| 80       | Unstable Angina                              | 3     | 2    | 1     |
| 81       | Upper Airway Obstruction                     | 2     | 1    | 1     |
| 83       | 12-lead ECG                                  | 3     | 1    | 2     |
| 84       | Angiography (invasive)                       | 3     | 1    | 2     |
| 86       | B-type Natriuretic Peptide (BNP)             | 3     | 2    | 1     |
| 88       | Bronchoscopy with BAL                        | 3     | 2    | 1     |
| 96       | Complete Blood Count (CBC) with differential | 3     | 1    | 2     |
| 102      | Lumbar Puncture_CSF Analysis                 | 2     | 1    | 1     |
| 106      | Stress Test (exercise or pharmacologic)      | 3     | 2    | 1     |
| 113      | Urinalysis                                   | 3     | 2    | 1     |

### One-Sided Topics (57 topics)

Topics where all statements are either true or false:

| Topic ID | Topic Name                                 | Total | True | False |
| -------- | ------------------------------------------ | ----- | ---- | ----- |
| 0        | Abdominal Trauma                           | 1     | 1    | 0     |
| 2        | Acute Appendicitis                         | 2     | 0    | 2     |
| 6        | Acute Liver Failure                        | 2     | 2    | 0     |
| 7        | Acute Myocardial Infarction (STEMI_NSTEMI) | 1     | 1    | 0     |
| 9        | Anaphylaxis                                | 1     | 0    | 1     |
| 14       | Asthma Exacerbation                        | 1     | 1    | 0     |
| 15       | Atrial Fibrillation                        | 2     | 0    | 2     |
| 16       | Blunt Trauma                               | 1     | 1    | 0     |
| 17       | Bowel Obstruction                          | 1     | 0    | 1     |
| 19       | Bronchitis                                 | 1     | 0    | 1     |
| 20       | Burns                                      | 1     | 1    | 0     |
| 22       | Cardiac Arrest                             | 1     | 0    | 1     |
| 23       | Cardiac Contusion                          | 1     | 0    | 1     |
| 24       | Cardiac Tamponade                          | 1     | 0    | 1     |
| 25       | Cardiomyopathy                             | 2     | 0    | 2     |
| 27       | Chest Pain (non-cardiac)                   | 1     | 0    | 1     |
| 28       | Compartment Syndrome                       | 3     | 3    | 0     |
| 29       | Delirium                                   | 1     | 0    | 1     |
| 30       | Diabetic Ketoacidosis                      | 3     | 3    | 0     |
| 32       | Ectopic Pregnancy                          | 1     | 1    | 0     |
| 35       | Encephalitis                               | 1     | 0    | 1     |
| 36       | Endocarditis                               | 1     | 1    | 0     |
| 41       | Hyperventilation Syndrome                  | 2     | 0    | 2     |
| 43       | Hyponatremia_Hypernatremia                 | 1     | 1    | 0     |
| 44       | Hypothermia_Hyperthermia                   | 1     | 0    | 1     |
| 45       | Interstitial Lung Disease                  | 1     | 0    | 1     |
| 47       | Lung Cancer                                | 2     | 0    | 2     |
| 49       | Mitral Regurgitation                       | 1     | 1    | 0     |
| 52       | Ovarian Torsion                            | 1     | 1    | 0     |
| 57       | Pericarditis                               | 1     | 0    | 1     |
| 58       | Placental Abruption                        | 1     | 1    | 0     |
| 59       | Pleural Effusion                           | 1     | 1    | 0     |
| 60       | Pneumomediastinum                          | 2     | 2    | 0     |
| 68       | Rhabdomyolysis                             | 2     | 0    | 2     |
| 70       | Ruptured AAA                               | 3     | 3    | 0     |
| 71       | Seizures_Status Epilepticus                | 1     | 0    | 1     |
| 74       | Sleep Apnea                                | 1     | 1    | 0     |
| 77       | Takotsubo Cardiomyopathy                   | 2     | 0    | 2     |
| 78       | Testicular Torsion                         | 1     | 1    | 0     |
| 82       | Ventricular Tachycardia                    | 1     | 1    | 0     |
| 85       | Arterial Blood Gas (ABG)                   | 1     | 0    | 1     |
| 87       | Blood Cultures                             | 2     | 0    | 2     |
| 90       | CT Angiogram                               | 2     | 2    | 0     |
| 91       | CT Other                                   | 1     | 0    | 1     |
| 92       | Cardiac Catheterization                    | 1     | 0    | 1     |
| 93       | Central Venous Pressure                    | 2     | 2    | 0     |
| 94       | Chest X-ray                                | 1     | 1    | 0     |
| 97       | Creatine Phosphokinase                     | 2     | 0    | 2     |
| 98       | Echocardiogram                             | 1     | 1    | 0     |
| 100      | Lactate                                    | 2     | 2    | 0     |
| 101      | Lipase                                     | 1     | 1    | 0     |
| 103      | MRI                                        | 1     | 0    | 1     |
| 104      | Procalcitonin                              | 1     | 1    | 0     |
| 108      | Toxicology Screen                          | 2     | 2    | 0     |
| 109      | Troponin I_T                               | 1     | 1    | 0     |
| 111      | Ultrasound FAST exam                       | 1     | 1    | 0     |
| 112      | Ultrasound Other                           | 1     | 0    | 1     |

## Data Quality Observations

### Potential Issues

- **Missing Topics:** 16 topics have no training data
- **Low Sample Topics:** 71 topics have ≤2 statements

## Generation Needs

Full CSV at data/eda/topic_needs.csv

**Target:** 5 total statements per topic, with at least 2 true and 2 false statements each.

**Summary:**

- Topics needing additional data: 113 out of 115
- Total true statements to generate: 237
- Total false statements to generate: 140
- Total additional statements needed: 377

### Preview: First 15 Topics with Generation Needs

| Topic ID | Topic Name                                 | Needed True | Needed False |
| -------- | ------------------------------------------ | ----------- | ------------ |
| 0        | Abdominal Trauma                           | 2           | 2            |
| 1        | Acute Abdomen                              | 1           | 0            |
| 2        | Acute Appendicitis                         | 3           | 0            |
| 3        | Acute Cholecystitis                        | 1           | 1            |
| 4        | Acute Coronary Syndrome                    | 2           | 0            |
| 5        | Acute Kidney Injury                        | 2           | 1            |
| 6        | Acute Liver Failure                        | 1           | 2            |
| 7        | Acute Myocardial Infarction (STEMI_NSTEMI) | 2           | 2            |
| 8        | Acute Respiratory Distress Syndrome        | 2           | 1            |
| 9        | Anaphylaxis                                | 3           | 1            |
| 10       | Aortic Dissection                          | 2           | 1            |
| 11       | Aortic Stenosis                            | 2           | 1            |
| 12       | Arrhythmias (various)                      | 2           | 1            |
| 13       | Aspiration Pneumonia                       | 3           | 2            |
| 14       | Asthma Exacerbation                        | 2           | 2            |

*Showing first 15 of 113 topics that need additional data.*

*This analysis was generated automatically by the EDA pipeline. For questions or issues, refer to the source code in `src/rag/eda.py`.*
