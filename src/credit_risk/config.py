from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "credit-risk-api"
    app_env: str = "uat"
    log_level: str = "INFO"
    api_key: str = ""

    project_root: Path = Field(default_factory=lambda: Path.cwd())
    raw_data_path: Path = Path("data/raw/data.csv")
    processed_data_path: Path = Path("data/processed/processed_data.csv")
    target_data_path: Path = Path("data/processed/processed_data_with_target.csv")
    model_path: Path = Path("artifacts/model.joblib")

    random_state: int = 42
    test_size: float = 0.2
    decision_threshold: float = 0.5
    n_risk_clusters: int = 3
    quick_train: bool = False

    mlflow_tracking_uri: str = ""
    mlflow_experiment: str = "Credit_Risk_Modeling"

    # Scorecard: score increases as odds of non-default increase.
    score_pdo: float = 50.0
    score_base_odds: float = 50.0
    score_base: float = 600.0
    score_min: int = 300
    score_max: int = 850

    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else self.project_root / path


settings = Settings()
