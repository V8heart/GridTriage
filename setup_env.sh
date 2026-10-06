#!/usr/bin/env bash
# GridPulse 가상환경 세팅 (Ubuntu 22.04 / Python 3.10 / CUDA 13.0 / RTX 4090 x2)
# 사용: bash setup_env.sh  그다음  source .venv/bin/activate
set -e

echo "=== [1/4] venv 생성 ==="
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

echo "=== [2/4] 의존성 설치 (requirements.txt, 버전 고정) ==="
# torch는 CUDA 버전별 빌드가 다르다. 기준 서버는 2.14.0+cu130.
# 다른 CUDA라면 https://pytorch.org/get-started/locally/ 의 명령으로 torch를 먼저 설치한 뒤 진행한다.
pip install -r requirements.txt

echo "=== [3/4] 위협 지식베이스 원본 받기 + 구축 (ATT&CK 19.2, ATLAS 5.6.0) ==="
bash scripts/fetch_kb_sources.sh
python -m pipeline.kb.build_kb

echo "=== [4/4] 로컬 LLM ==="
# Ollama 설치: https://ollama.com (이 스크립트 밖에서) 후
#   ollama pull gemma3:12b
# 검색 모델(bge-m3, bge-reranker-v2-m3)은 첫 실행 때 Hugging Face에서 자동으로 받는다.

echo ""
echo "=== 완료 ==="
echo "GPU 확인:"
python -c "import pynvml; pynvml.nvmlInit(); print('GPU 수:', pynvml.nvmlDeviceGetCount()); pynvml.nvmlShutdown()" || echo "(pynvml 확인 실패 - GPU 없는 환경일 수 있음)"
echo ""
echo "다음: source .venv/bin/activate"
echo "Physics Tier B는 별도 Python 3.11 환경: ./setup_physics_env.sh"
