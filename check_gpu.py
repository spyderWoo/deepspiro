# check_gpu.py

import sys

def check_pytorch():
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        num_devices = torch.cuda.device_count()
        device_names = [torch.cuda.get_device_name(i) for i in range(num_devices)] if cuda_available else []
        print("=== PyTorch GPU 확인 ===")
        print(f"CUDA 사용 가능 여부: {cuda_available}")
        print(f"사용 가능한 GPU 개수: {num_devices}")
        if cuda_available:
            for i, name in enumerate(device_names):
                print(f"  - GPU {i}: {name}")
    except ImportError:
        print("PyTorch가 설치되어 있지 않습니다.")
    print()

def check_tensorflow():
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        print("=== TensorFlow GPU 확인 ===")
        print(f"발견된 GPU 리스트: {gpus}")
        print(f"GPU 개수: {len(gpus)}")
        if gpus:
            for gpu in gpus:
                print(f"  - {gpu.name} (type: {gpu.device_type})")
    except ImportError:
        print("TensorFlow가 설치되어 있지 않습니다.")

if __name__ == "__main__":
    print(f"Python 버전: {sys.version.split()[0]}")
    check_pytorch()
    check_tensorflow()
