Python 3.12.11
tensorflow                            2.19.0
torch                                 2.6.0
cuda      _                           12.4
cuDNN                                 8.9.7

1. Conda 환경 생성
```bash
conda create -n llm-book python=3.12.11 -y
```

2. 환경 활성화
```bash
conda activate llm-book
```

3. TensorFlow 설치 (pip 권장)
```bash
pip install tensorflow==2.19.0
```

4. PyTorch 설치 (pip 권장: conda 채널 패키지 부재 대응)
```bash
# PyTorch 2.8.0은 아직 공식 배포되지 않았습니다. 사용 가능한 최신 버전인 2.6.0을 설치합니다.
# pip를 사용하면 더 안정적으로 설치할 수 있습니다.
# (참고: PyTorch 공식 홈페이지에서 CUDA 버전에 맞는 최신 명령을 확인하는 것이 가장 좋습니다.)
# cu124 인덱스에서 사용 가능한 최신 버전인 2.6.0을 설치합니다.
pip3 install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

5. CUDA 및 cuDNN 별도 설치 생략 (충돌 방지)
- 위 PyTorch `pip` 설치 시 필요한 CUDA/cuDNN 런타임이 함께 포함됩니다.
- 별도로 `cuda-toolkit`/`cudnn`을 conda로 추가 설치하면 충돌할 수 있으므로 설치하지 않습니다.

6. 설치 확인 (GPU 지원 포함)
```bash
# TensorFlow 버전 및 GPU 지원 확인
python -c "import tensorflow as tf; print('TF', tf.__version__); print('Built with CUDA?', tf.test.is_built_with_cuda()); print('GPUs:', tf.config.list_physical_devices('GPU'))"

# PyTorch 버전 및 GPU 지원 확인
python -c "import torch; print('Torch', torch.__version__, 'CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"

# 간단한 GPU 연산 테스트 (선택사항)
python -c "import tensorflow as tf; import torch; print('=== TF GPU Test ==='); print('TF GPU devices:', len(tf.config.list_physical_devices('GPU'))); print('=== PyTorch GPU Test ==='); print('PyTorch GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

7. (선택) 불필요한 패키지/캐시 정리
```bash
conda clean --all -y
# pip 캐시 정리 (pip 20.1 이상)
pip cache purge
```



