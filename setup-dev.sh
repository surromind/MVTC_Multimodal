#!/bin/bash

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m' # 추가: RED 색상 정의
NC='\033[0m' # No Color

# 로그 함수
log_step() {
    echo -e "${BLUE}[SETUP]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 에러 핸들링 함수
handle_error() {
    log_error "$1"
    exit 1
}

# 파일 생성 함수 (파일이 이미 존재하는 경우 생성하지 않음)
create_file_if_not_exists() {
    local file_name="$1"
    local content="$2"

    if [ ! -f "$file_name" ]; then
        echo "$content" > "$file_name"
        echo "$file_name Created"
    else
        echo "$file_name already exists, skipping creation."
    fi
}

# Pipfile 생성
echo "[1/5] Creating Pipfile..."
create_file_if_not_exists "Pipfile" '
[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"

[packages]
python-dotenv = "==1.0.1"

[dev-packages]
pre-commit = "==3.7.1"
pytest = "==8.3.4"

[requires]
python_version = "3.12"
'
echo "Complete Pipfile"

# .pre-commit-config.yaml 생성
echo "[2/5] Creating .pre-commit-config.yaml..."
create_file_if_not_exists ".pre-commit-config.yaml" '
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files

-   repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
    -   id: isort
        args: [--line-length=120]

-   repo: https://github.com/psf/black
    rev: 24.3.0
    hooks:
    -   id: black
        args: [--line-length=120]

-   repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
    -   id: flake8
        args: [--max-line-length=120, '--ignore=F401,W503']
'
echo ".pre-commit-config.yaml Created"

# 스크립트 시작
echo -e "${BLUE}=== Python Development Environment Setup ===${NC}"

# python 가상환경 세팅 및 package 설치
log_step "Upgrading pip..."
pip install --upgrade pip || handle_error "pip 업그레이드 실패"
log_success "Pip 업그레이드 완료"

# pipenv 설치
log_step "Installing pipenv..."
pip install pipenv || handle_error "pipenv 설치 실패"
log_success "Pipenv 설치 완료"

# 의존성 설치
log_step "Installing dependencies..."
pipenv install --dev || handle_error "의존성 설치 실패"
log_success "의존성 설치 완료"

# pre-commit 설치
log_step "Installing pre-commit hooks..."
pipenv run pre-commit install || handle_error "pre-commit 설치 실패"
log_success "Pre-commit hooks 설치 완료"
