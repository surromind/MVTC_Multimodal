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

log_step "Configuring prepare-commit-msg hook..."
# Repository Root에서 시작
cd .git/hooks || handle_error ".git/hooks 디렉토리로 이동 실패"

# prepare-commit-msg 파일 생성
touch prepare-commit-msg

# 실행 권한 부여
chmod +x prepare-commit-msg || handle_error "prepare-commit-msg에 실행 권한 부여 실패"

# .git/hooks/prepare-commit-msg 내용 작성
cat << EOF > prepare-commit-msg
#!/bin/bash
FIRST_LINE=\$(head -n 1 \$1)

COMMITFORMAT="^(feat|fix|docs|style|refactor|test|chore|build|ci|perf|security|hotfix): .+$"


if ! [[ "\$FIRST_LINE" =~ \$COMMITFORMAT ]]; then
  echo ""
  echo " Commit Message 포맷을 아래 예시와 같이 지켜주세요."
  echo " Prefix : 사용 가능한 commit의 Prefix는 아래와 같습니다."
  echo ""
  echo "======================= 반드시 콜론(:) 을 붙여야 합니다. ========================="
  echo ""
  echo "1. feat: 새로운 기능 추가"
  echo "  - 새로운 기능이나 기능적 변경 사항을 추가할 때 사용합니다."
  echo "    예: feat(user): 로그인 기능 추가"
  echo ""
  echo "2. fix: 버그 수정"
  echo "  - 기존 코드에서 발견된 버그를 수정할 때 사용합니다."
  echo "    예: fix(auth): 로그인 페이지에서 발생하는 오류 수정"
  echo ""
  echo "3. docs: 문서 변경"
  echo "  - 문서(README, 도움말, 코드 주석 등)에 대한 수정이나 추가에 사용합니다."
  echo "    예: docs: API 문서 업데이트"
  echo ""
  echo "4. style: 코드 스타일 수정"
  echo "  - 코드의 동작에 영향을 미치지 않는 스타일 수정을 나타냅니다."
  echo "    예: style: 들여쓰기 문제 수정"
  echo ""
  echo "5. refactor: 리팩토링"
  echo "  - 기능 변경 없이 코드 구조나 효율성을 개선하는 변경에 사용합니다."
  echo "    예: refactor(auth): 로그인 기능 리팩토링"
  echo ""
  echo "6. test: 테스트 코드 추가/수정"
  echo "  - 새로운 테스트 추가나 기존 테스트 수정에 사용합니다."
  echo "    예: test: 로그인 기능 유닛 테스트 추가"
  echo ""
  echo "7. chore: 잡다한 작업"
  echo "  - 코드, 문서 등 주요 기능과 관련이 없는 기타 작업에 사용합니다."
  echo "    예: chore: 의존성 패키지 업데이트"
  echo ""
  echo "8. build: 빌드 관련 변경"
  echo "  - 빌드 시스템(webpack, Gulp, Gradle 등)이나 의존성 관련 변경에 사용합니다."
  echo "    예: build: 프로젝트 빌드 설정 업데이트"
  echo ""
  echo "9. ci: CI/CD 설정 변경"
  echo "  - CI/CD 파이프라인 설정 변경에 사용됩니다."
  echo "    예: ci: GitHub Actions 설정 추가"
  echo ""
  echo "10. perf: 성능 개선"
  echo "  - 성능을 개선하기 위한 변경 사항에 사용됩니다."
  echo "    예: perf: 이미지 로딩 속도 개선"
  echo ""
  echo "11. security: 보안 관련 변경"
  echo "  - 보안을 강화하기 위한 변경 사항에 사용됩니다."
  echo "    예: security: XSS 취약점 수정"
  echo ""
  echo "12. hotfix: 긴급 수정"
  echo "  - 배포 후 긴급하게 수정해야 할 버그를 수정하는 경우에 사용됩니다."
  echo "    예: hotfix: 프로덕션 환경에서 발생한 로그인 버그 수정"
  echo ""
  echo "=================================================================================="
  echo ""
  exit 1
fi
EOF

log_success "Prepare-commit-msg hook 설정 완료"

# 최종 완료 메시지
echo -e "  * ${BLUE}Pipfile 명시 패키지 설치 완료${NC}"
echo -e "  * ${BLUE}pre-commit-hook 설정 완료${NC}"
echo -e "${GREEN}=== Development Environment Setup Complete ===${NC}"
