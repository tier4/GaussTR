#!/usr/bin/env python
"""Test script to verify GaussTR Lightning model instantiation.

This script tests that all components can be instantiated without MMEngine.

Usage:
    python -m gausstr_lightning.scripts.test_instantiation
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        import torch
        print(f"  [OK] torch {torch.__version__}")
    except ImportError as e:
        print(f"  [FAIL] torch: {e}")
        return False

    try:
        import pytorch_lightning as pl
        print(f"  [OK] pytorch_lightning {pl.__version__}")
    except ImportError as e:
        print(f"  [FAIL] pytorch_lightning: {e}")
        return False

    try:
        import torchmetrics
        print(f"  [OK] torchmetrics {torchmetrics.__version__}")
    except ImportError as e:
        print(f"  [FAIL] torchmetrics: {e}")
        return False

    try:
        from models.utils import cam2world, rotmat_to_quat
        print("  [OK] models.utils")
    except ImportError as e:
        print(f"  [FAIL] models.utils: {e}")
        return False

    try:
        from models.pytorch_voxelizer import PyTorchVoxelizer
        print("  [OK] models.pytorch_voxelizer")
    except ImportError as e:
        print(f"  [FAIL] models.pytorch_voxelizer: {e}")
        return False

    try:
        from models.vitdet_fpn import ViTDetFPN
        print("  [OK] models.vitdet_fpn")
    except ImportError as e:
        print(f"  [FAIL] models.vitdet_fpn: {e}")
        return False

    try:
        from models.gausstr_decoder import GaussTRDecoder
        print("  [OK] models.gausstr_decoder")
    except ImportError as e:
        print(f"  [FAIL] models.gausstr_decoder: {e}")
        return False

    try:
        from models.gausstr_head import GaussTRHead
        print("  [OK] models.gausstr_head")
    except ImportError as e:
        print(f"  [FAIL] models.gausstr_head: {e}")
        return False

    try:
        from models.gausstr import GaussTRLightning
        print("  [OK] models.gausstr")
    except ImportError as e:
        print(f"  [FAIL] models.gausstr: {e}")
        return False

    try:
        from dataset import GaussTRDataModule
        print("  [OK] dataset")
    except ImportError as e:
        print(f"  [FAIL] dataset: {e}")
        return False

    try:
        from evaluation import OccupancyIoU
        print("  [OK] evaluation")
    except ImportError as e:
        print(f"  [FAIL] evaluation: {e}")
        return False

    print("All imports successful!\n")
    return True


def test_model_instantiation():
    """Test model instantiation."""
    print("Testing model instantiation...")

    import torch
    from models.gausstr import GaussTRLightning

    try:
        model = GaussTRLightning(
            num_queries=300,
            embed_dims=256,
            feat_dims=512,
            neck_in_channels=512,
            neck_out_channels=256,
            decoder_num_layers=3,
            head_text_protos=None,  # Skip loading text protos for test
        )
        print(f"  [OK] Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    except Exception as e:
        print(f"  [FAIL] Model instantiation: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test forward pass with dummy data
    print("\nTesting forward pass with dummy data...")
    try:
        batch_size = 1
        num_views = 6
        h, w = 432, 768
        feat_h, feat_w = h // 16, w // 16

        dummy_batch = {
            'images': torch.randn(batch_size, num_views, 3, h, w),
            'feats': torch.randn(batch_size, num_views, 512, feat_h, feat_w),
            'depth': torch.rand(batch_size, num_views, h, w) * 50,
            'cam2img': torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch_size, num_views, -1, -1).clone(),
            'cam2ego': torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch_size, num_views, -1, -1).clone(),
        }

        # Set focal length
        dummy_batch['cam2img'][:, :, 0, 0] = 1000
        dummy_batch['cam2img'][:, :, 1, 1] = 1000

        model.eval()
        with torch.no_grad():
            # Test inference forward
            output = model(
                images=dummy_batch['images'],
                feats=dummy_batch['feats'],
                depth=dummy_batch['depth'],
                cam2img=dummy_batch['cam2img'],
                cam2ego=dummy_batch['cam2ego'],
            )
            print(f"  [OK] Forward pass successful, output shape: {output.shape}")

    except Exception as e:
        print(f"  [FAIL] Forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_data_components():
    """Test data pipeline components."""
    print("\nTesting data components...")

    try:
        from dataset.transforms import (
            Compose, LoadMultiViewImages, ImageAug3D, LoadFeatMaps, PackInputs
        )
        print("  [OK] Transform classes imported")
    except ImportError as e:
        print(f"  [FAIL] Transform imports: {e}")
        return False

    try:
        from dataset.dataset import NuScenesOccDataset, OCC_CLASSES
        print(f"  [OK] Dataset class imported, {len(OCC_CLASSES)} classes defined")
    except ImportError as e:
        print(f"  [FAIL] Dataset imports: {e}")
        return False

    try:
        from dataset.collate import collate_gausstr
        print("  [OK] Collate function imported")
    except ImportError as e:
        print(f"  [FAIL] Collate imports: {e}")
        return False

    try:
        from dataset.datamodule import GaussTRDataModule
        # Don't actually instantiate as it requires data files
        print("  [OK] DataModule class imported")
    except ImportError as e:
        print(f"  [FAIL] DataModule imports: {e}")
        return False

    return True


def test_evaluation_components():
    """Test evaluation metric."""
    print("\nTesting evaluation components...")

    import torch
    from evaluation import OccupancyIoU, fast_hist, per_class_iou

    try:
        metric = OccupancyIoU(num_classes=18)
        print("  [OK] OccupancyIoU metric created")
    except Exception as e:
        print(f"  [FAIL] OccupancyIoU creation: {e}")
        return False

    try:
        # Test metric update
        preds = torch.randint(0, 18, (2, 200, 200, 16))
        targets = torch.randint(0, 18, (2, 200, 200, 16))
        metric.update(preds, targets)
        results = metric.compute()
        print(f"  [OK] Metric computed: mIoU = {results['miou'].item():.4f}")
    except Exception as e:
        print(f"  [FAIL] Metric computation: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_config_system():
    """Test configuration system."""
    print("\nTesting configuration system...")

    try:
        from config.base import GaussTRConfig
        config = GaussTRConfig.featup()
        print(f"  [OK] FeatUp config created: embed_dims={config.embed_dims}")
    except Exception as e:
        print(f"  [FAIL] Config creation: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        config_talk2dino = GaussTRConfig.talk2dino()
        print(f"  [OK] Talk2DINO config created: feat_dims={config_talk2dino.feat_dims}")
    except Exception as e:
        print(f"  [FAIL] Talk2DINO config: {e}")
        return False

    return True


def check_mmengine_dependency():
    """Check if MMEngine is being imported (should not be for core functionality)."""
    print("\nChecking MMEngine dependency...")

    import sys
    mmengine_modules = [m for m in sys.modules if 'mmengine' in m.lower()]
    mmdet_modules = [m for m in sys.modules if 'mmdet' in m.lower()]
    mmcv_modules = [m for m in sys.modules if 'mmcv' in m.lower()]

    if mmengine_modules:
        print(f"  [WARN] MMEngine modules loaded: {mmengine_modules[:5]}...")
    else:
        print("  [OK] No MMEngine modules loaded")

    if mmdet_modules:
        print(f"  [WARN] MMDet modules loaded: {mmdet_modules[:5]}...")
    else:
        print("  [OK] No MMDet modules loaded")

    # MMCV ops is acceptable for deformable attention
    if mmcv_modules:
        if all('ops' in m for m in mmcv_modules):
            print(f"  [OK] Only mmcv.ops loaded (expected for deformable attention)")
        else:
            print(f"  [WARN] MMCV modules loaded: {mmcv_modules[:5]}...")
    else:
        print("  [OK] No MMCV modules loaded (using fallback attention)")

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("GaussTR Lightning - Model Instantiation Test")
    print("=" * 60)
    print()

    all_passed = True

    all_passed &= test_imports()
    all_passed &= test_config_system()
    all_passed &= test_data_components()
    all_passed &= test_evaluation_components()
    all_passed &= test_model_instantiation()
    all_passed &= check_mmengine_dependency()

    print()
    print("=" * 60)
    if all_passed:
        print("All tests PASSED!")
        print("GaussTR Lightning is ready to use without MMEngine.")
    else:
        print("Some tests FAILED. Check the output above.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
