from pathlib import Path

# 현재 파일 기준 경로
ROOT_DIR = Path(__file__).resolve().parents[2]
TEST_DATA_DIR = ROOT_DIR / "test_data"
MODEL_DIR = ROOT_DIR / "models"
VECTOR_DIR = ROOT_DIR / "vectors"
LOGGING_FILE_PATH = ROOT_DIR / "logs"
