import yaml
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Config:
    # Core settings
    chroma_path: str
    chunk_size: int
    chunk_overlap: int
    k: int
    model_names: List[str]
    
    # Hybrid search settings
    bm25_weight: float
    use_hybrid_search: bool
    
    # Processing settings
    plot_results: bool
    save_results: bool
    reset_db: bool
    
    # Fact checker settings
    export_file: str
    
    # Data tester settings
    statements_dir: str
    answers_dir: str
    topics_file: str
    
    # Validation settings
    wait_for_validation: bool
    validation_timeout: int
    
    # Statement range settings
    from_statement: Optional[int]
    to_statement: Optional[int]


def load_config() -> Config:
    """Load configuration from YAML file into dataclass."""
    with open('config.yaml', 'r') as file:
        yaml_data = yaml.safe_load(file)
    
    return Config(**yaml_data)


# Global config instance
config = load_config()
