# Data Directory

This document describes the how to use the `data/` directory used in the Norwegian AI Championship 2025 project.

## Directory Structure

```txt
├── data
│   ├── external       <- Data from third party sources.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
```

### Raw Data (`data/raw`)

- Contains original data files exactly as received with subdirectories for different tasks.
- **Do not modify** or overwrite files in this folder.
- If data needs corrections or cleaning, create copies in data/processed or a new subdirectory under data/processed.

### External Data (`data/external`)

- Store data fetched from third-party sources.
- This can include datasets, APIs, or any external resources that are not part of the original project.
- Document for each source (URL, date fetched). So other developers can understand where the data came from and how to access it.

### Processed Data (`data/processed`)

- It must be possible to reproduce the processed data from the raw data. Each data pipeline must be a DAG (Directed Acyclic Graph) with documentation on how to reproduce the data.
- Organize subfolders by task
- Include a README.md in each subdirectory describing its contents, processing scripts used, and version information.
