# Windows 10 + RTX 4070 SUPER 딥러닝 환경설정 가이드

## Python 버전 및 Conda 환경 설정 가이드

RTX 4070 SUPER + CUDA 12.2 + cuDNN 8.9 + TensorRT 10.13.2.6 환경에서 최적의 호환성을 위한 Python 버전 및 설치 가이드입니다.

### 권장 Python 버전
- **Python 3.9-3.10**: 모든 라이브러리와 호환성이 가장 좋음
  - PyTorch: Python 3.8-3.11 지원
  - TensorFlow: Python 3.8-3.10 지원
  - Hugging Face: Python 3.8-3.10 권장

### Conda 환경 설정 명령어

```bash
# 딥러닝 가상환경 생성 (Python 3.10 권장)
conda create -n deeplearning python=3.10

# 환경 활성화
conda activate deeplearning

# PyTorch 설치 (CUDA 12.2 지원)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# 또는
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu122

# TensorFlow 설치
pip install tensorflow

# Hugging Face Transformers 설치 (기본 의존성 포함)
pip install transformers

# 추가 Hugging Face 라이브러리 설치
pip install datasets accelerate evaluate optimum
```

### 설치 확인 명령어

```bash
# CUDA와 GPU 사용 가능 여부 확인
python -c "import torch; print('PyTorch 버전:', torch.__version__); print('CUDA 사용 가능:', torch.cuda.is_available()); print('사용 가능한 GPU:', torch.cuda.get_device_name(0))"

python -c "import tensorflow as tf; print('TensorFlow 버전:', tf.__version__); print('GPU 사용 가능:', tf.config.list_physical_devices('GPU'))"

python -c "from transformers import AutoModel; print('Transformers 설치 확인 성공')"
```

### 주요 라이브러리 버전 호환성

| 라이브러리 | 권장 버전 | CUDA 12.2 호환성 |
|:----------|:---------|:--------------|
| PyTorch   | 2.2.x 이상 | 네이티브 지원 (cu122) |
| TensorFlow | 2.15.x 이상 | 자동 호환 지원 |
| Transformers | 4.36.x 이상 | PyTorch/TensorFlow 의존 |
| Python | 3.10 | 최적의 호환성 |


## 설치 순서 및 공식 다운로드 경로

| 순서 | 항목        | 공식사이트/경로                                  | 설치/설정 방법 요약                                         | 환경변수/경로 설정 주의사항                          |
|:---:|:-----------|:-------------------------------------------------|:-----------------------------------------------------------|:----------------------------------------------------|
| 1   | 그래픽 드라이버 | https://www.nvidia.com/Download/index.aspx       | Studio/Game Ready 535.xx 이상, 설치 프로그램 실행           | 자동 설정 (별도 설정 필요 없음)                      |
| 2   | CUDA Toolkit   | https://developer.nvidia.com/cuda-downloads      | 12.2 (Windows 10), 설치 파일 실행                          | 설치 시 자동으로 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin` 등이 PATH에 등록됨 |
| 3   | Visual Studio  | https://visualstudio.microsoft.com/ko/vs/        | Community 2022, C++ 워크로드 반드시 선택                   | 자동 설정 (CUDA, TensorRT, cuDNN 빌드시 MSVC 필요)    |
| 4   | cuDNN          | https://developer.nvidia.com/cudnn               | CUDA 12.x 대응 8.9.x zip 다운로드, 압축해제 후 파일 복사    | 압축해제한 `bin`, `include`, `lib\x64`의 파일을 CUDA 설치 폴더(`v12.2`) 내 같은 폴더에 “덮어쓰기”  |
| 5   | TensorRT       | https://developer.nvidia.com/tensorrt            | CUDA 12.x, Windows용 zip 다운로드, 압축해제 후 복사         | ① lib, include 폴더를 CUDA/프로젝트 경로에 추가<br>② lib 폴더(예: `C:\TensorRT-10.x.x\lib`)를 환경변수 PATH에 추가<br>③ 일부 DLL은 CUDA bin에도 복사하면 오류 예방됨|
| 6   | Anaconda       | https://www.anaconda.com/products/distribution   | 설치 후 터미널에서 가상환경 생성 권장                      | 자동 설정, 가상환경 활성화 필요                      |
| 7   | PyTorch        | https://pytorch.org/get-started/locally/         | pip/conda에서 CUDA 12.2 (cu122)용 명령 사용                | 가상환경 내에서 설치, CUDA DLL 인식 위해 PATH 환경변수(위단계) 세팅 필수 |
| 8   | TensorFlow     | https://www.tensorflow.org/install               | pip install tensorflow<br>(최신버전 자동 CUDA 12.2 지원)    | cuDNN/CUDA DLL 인식 못하면<br>환경변수 활성 확인 필수  |

---

## 환경변수/경로 세팅 요약

- CUDA, cuDNN, TensorRT 관련 DLL 및 lib 경로는 반드시 시스템 환경변수 `PATH`에 포함되어야 함  
  - 예시:  
    - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin`
    - `C:\TensorRT-10.x.x\lib`
- cuDNN, TensorRT의 파일은 압축 해제 후 CUDA 폴더에 직접 “덮어쓰기” 복사
- 각 개발 가상환경(Anaconda) 내에서 pip/conda로 PyTorch, TensorFlow 설치
- Visual Studio는 C++ 워크로드 필수 포함

---

## 설치 및 테스트 체크리스트

- 각 단계마다 설치·복사 후 시스템 재부팅 권장
- PyTorch에서  

```cmd

python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

```

- TensorFlow에서  


```cmd

python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

```

- 위 코드에서 GPU가 정상적으로 인식되는지 반드시 확인할 것

---

**이 문서는 2025년 9월 기준 RTX 4070 SUPER, CUDA 12.2, cuDNN 8.9.x, TensorRT 10.x 환경 최적화를 목표로 작성되었습니다.**
