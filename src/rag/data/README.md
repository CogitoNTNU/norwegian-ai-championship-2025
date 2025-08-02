# Data directory

This folder contains all data files used in the project, organised so that the
original competition files stay immutable while all derived artefacts are
written elsewhere.

data/
├── raw/ # untouched competition data
│ ├── train/ # 200 medical statements (statements/*.txt) + ground-truth labels (answers/*.json)
│ └── topics/ # StatPearls reference articles for the 115 emergency topics
├── processed/ # cleaned-up versions of the original data that our own scripts create automatically
└── augmented/ # extra training data we create to boost performance. (paraphrases, hard negatives, etc.)
