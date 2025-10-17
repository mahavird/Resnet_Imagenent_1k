#!/usr/bin/env python3
"""
Test script to verify the project setup is working correctly.
Run this before training to catch any import or configuration issues.
"""

import sys
import torch

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        from src.models import Bottleneck, ResNet, resnet50, resnet101, resnet152
        print("✅ Model imports successful")
    except ImportError as e:
        print(f"❌ Model import failed: {e}")
        return False
    
    try:
        from src.data.datasets import DataConfig, make_dataloaders
        print("✅ Data imports successful")
    except ImportError as e:
        print(f"❌ Data import failed: {e}")
        return False
    
    try:
        from src.engine.engine import train_one_epoch, evaluate
        print("✅ Engine imports successful")
    except ImportError as e:
        print(f"❌ Engine import failed: {e}")
        return False
    
    try:
        from src.utils.utils import set_seed, get_device, save_checkpoint
        print("✅ Utils imports successful")
    except ImportError as e:
        print(f"❌ Utils import failed: {e}")
        return False
    
    return True

def test_model_creation():
    """Test that models can be created successfully."""
    print("\nTesting model creation...")
    
    try:
        from src.models import resnet50, resnet101, resnet152
        
        # Test ResNet-50
        model50 = resnet50(num_classes=10)
        print(f"✅ ResNet-50 created: {sum(p.numel() for p in model50.parameters())} parameters")
        
        # Test ResNet-101
        model101 = resnet101(num_classes=10)
        print(f"✅ ResNet-101 created: {sum(p.numel() for p in model101.parameters())} parameters")
        
        # Test ResNet-152
        model152 = resnet152(num_classes=10)
        print(f"✅ ResNet-152 created: {sum(p.numel() for p in model152.parameters())} parameters")
        
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_forward_pass():
    """Test that models can perform forward pass."""
    print("\nTesting forward pass...")
    
    try:
        from src.models import resnet50
        
        model = resnet50(num_classes=10)
        model.eval()
        
        # Create dummy input
        x = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            output = model(x)
        
        expected_shape = (2, 10)  # batch_size, num_classes
        if output.shape == expected_shape:
            print(f"✅ Forward pass successful: {output.shape}")
            return True
        else:
            print(f"❌ Forward pass failed: expected {expected_shape}, got {output.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

def test_device():
    """Test device detection."""
    print("\nTesting device detection...")
    
    try:
        from src.utils.utils import get_device
        
        device = get_device()
        print(f"✅ Device detected: {device}")
        
        # Test that we can move a tensor to device
        x = torch.randn(2, 3, 224, 224).to(device)
        print(f"✅ Tensor moved to device: {x.device}")
        
        return True
    except Exception as e:
        print(f"❌ Device test failed: {e}")
        return False

def test_config():
    """Test configuration loading."""
    print("\nTesting configuration loading...")
    
    try:
        import os
        try:
            import tomllib  # Python 3.11+
        except ImportError:
            import tomli as tomllib  # Python < 3.11
        
        config_path = "configs/experiment.toml"
        if not os.path.exists(config_path):
            print(f"❌ Config file not found: {config_path}")
            return False
        
        with open(config_path, "rb") as f:
            cfg = tomllib.load(f)
        
        required_sections = ["data", "model", "optim", "train"]
        for section in required_sections:
            if section not in cfg:
                print(f"❌ Missing config section: {section}")
                return False
        
        print("✅ Configuration loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running project setup tests...\n")
    
    tests = [
        test_imports,
        test_model_creation,
        test_forward_pass,
        test_device,
        test_config,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "="*50)
    if all(results):
        print("🎉 All tests passed! Project setup looks good.")
        print("\nNext steps:")
        print("1. Update data paths in configs/experiment.toml")
        print("2. Run: python src/train.py")
        return 0
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
