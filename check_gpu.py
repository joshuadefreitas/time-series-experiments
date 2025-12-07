#!/usr/bin/env python3
"""
GPU Availability and ML Framework Checker
Checks if GPU is available and if common ML frameworks can use it.
"""

import sys
import subprocess
import platform

def check_nvidia_smi():
    """Check if nvidia-smi is available (indicates NVIDIA GPU driver is installed)"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ NVIDIA GPU driver detected")
            print("\nGPU Information:")
            print(result.stdout.split('\n')[0:15])  # Show first few lines
            return True
        return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def check_pytorch():
    """Check PyTorch GPU support"""
    try:
        import torch
        print(f"\n{'='*60}")
        print("PyTorch GPU Check")
        print(f"{'='*60}")
        print(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA is available!")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"\n  GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
            
            # Test a simple computation
            print("\n  Testing GPU computation...")
            try:
                x = torch.randn(1000, 1000).cuda()
                y = torch.randn(1000, 1000).cuda()
                z = torch.matmul(x, y)
                print("  ✓ GPU computation test passed!")
                return True
            except Exception as e:
                print(f"  ✗ GPU computation test failed: {e}")
                return False
        else:
            print("✗ CUDA is not available in PyTorch")
            print("  (PyTorch is installed but GPU support is not enabled)")
            return False
    except ImportError:
        print("\nPyTorch is not installed")
        return None

def check_tensorflow():
    """Check TensorFlow GPU support"""
    try:
        import tensorflow as tf
        print(f"\n{'='*60}")
        print("TensorFlow GPU Check")
        print(f"{'='*60}")
        print(f"TensorFlow version: {tf.__version__}")
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ {len(gpus)} GPU(s) detected!")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
                try:
                    details = tf.config.experimental.get_device_details(gpu)
                    print(f"    Details: {details}")
                except:
                    pass
            
            # Test a simple computation
            print("\n  Testing GPU computation...")
            try:
                with tf.device('/GPU:0'):
                    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                    c = tf.matmul(a, b)
                print("  ✓ GPU computation test passed!")
                return True
            except Exception as e:
                print(f"  ✗ GPU computation test failed: {e}")
                return False
        else:
            print("✗ No GPUs detected by TensorFlow")
            return False
    except ImportError:
        print("\nTensorFlow is not installed")
        return None

def check_cupy():
    """Check CuPy (NumPy-like GPU arrays)"""
    try:
        import cupy as cp
        print(f"\n{'='*60}")
        print("CuPy GPU Check")
        print(f"{'='*60}")
        print(f"CuPy version: {cp.__version__}")
        
        # Get device info
        mempool = cp.get_default_memory_pool()
        print(f"✓ CuPy is available!")
        print(f"  Current GPU memory usage: {mempool.used_bytes() / 1e9:.2f} GB")
        print(f"  Total GPU memory: {mempool.total_bytes() / 1e9:.2f} GB")
        
        # Test a simple computation
        print("\n  Testing GPU computation...")
        try:
            x = cp.random.randn(1000, 1000)
            y = cp.random.randn(1000, 1000)
            z = cp.dot(x, y)
            print("  ✓ GPU computation test passed!")
            return True
        except Exception as e:
            print(f"  ✗ GPU computation test failed: {e}")
            return False
    except ImportError:
        print("\nCuPy is not installed")
        return None

def main():
    print("="*60)
    print("GPU Availability and ML Framework Checker")
    print("="*60)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    
    # Check for NVIDIA GPU driver
    print(f"\n{'='*60}")
    print("NVIDIA GPU Driver Check")
    print(f"{'='*60}")
    has_nvidia = check_nvidia_smi()
    
    if not has_nvidia:
        print("\n⚠ No NVIDIA GPU driver detected.")
        print("  This could mean:")
        print("  - You don't have an NVIDIA GPU")
        print("  - NVIDIA drivers are not installed")
        print("  - You're on macOS (which doesn't support NVIDIA GPUs)")
        print("  - You have an AMD/Intel GPU (different setup required)")
    
    # Check ML frameworks
    pytorch_result = check_pytorch()
    tensorflow_result = check_tensorflow()
    cupy_result = check_cupy()
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    
    results = []
    if pytorch_result is not None:
        results.append(("PyTorch", pytorch_result))
    if tensorflow_result is not None:
        results.append(("TensorFlow", tensorflow_result))
    if cupy_result is not None:
        results.append(("CuPy", cupy_result))
    
    if results:
        for name, result in results:
            status = "✓ Available" if result else "✗ Not available"
            print(f"{name}: {status}")
    else:
        print("No ML frameworks with GPU support are installed.")
        print("\nTo install GPU-enabled frameworks:")
        print("  PyTorch: https://pytorch.org/get-started/locally/")
        print("  TensorFlow: pip install tensorflow[and-cuda]")
        print("  CuPy: pip install cupy-cuda11x (adjust CUDA version)")

if __name__ == "__main__":
    main()