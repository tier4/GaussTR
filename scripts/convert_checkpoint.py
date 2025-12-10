#!/usr/bin/env python
"""Convert MMEngine checkpoint to pure PyTorch state dict.

This script must be run in an environment with MMEngine installed.

Usage:
    python -m gausstr_lightning.scripts.convert_checkpoint \
        ckpts/gausstr_featup_e24_miou11.70.pth \
        ckpts/gausstr_featup_lightning.pth
"""

import argparse
import sys
from pathlib import Path


def convert_checkpoint(input_path: str, output_path: str):
    """Convert MMEngine checkpoint to pure state dict.

    Args:
        input_path: Path to MMEngine checkpoint.
        output_path: Path to save converted checkpoint.
    """
    import torch

    print(f"Loading checkpoint: {input_path}")
    ckpt = torch.load(input_path, map_location='cpu', weights_only=False)

    # Extract state dict
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    # Clean up keys - remove 'model.' prefix
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        if new_key.startswith('model.'):
            new_key = new_key[6:]  # Remove 'model.'
        cleaned_state_dict[new_key] = v

    # Save as pure state dict
    output = {
        'state_dict': cleaned_state_dict,
        'meta': {
            'converted_from': input_path,
            'format': 'lightning',
        }
    }

    print(f"Saving converted checkpoint: {output_path}")
    torch.save(output, output_path)
    print(f"Done! Keys: {len(cleaned_state_dict)}")


def main():
    parser = argparse.ArgumentParser(description='Convert MMEngine checkpoint')
    parser.add_argument('input', help='Input MMEngine checkpoint path')
    parser.add_argument('output', help='Output Lightning checkpoint path')
    args = parser.parse_args()

    convert_checkpoint(args.input, args.output)


if __name__ == '__main__':
    main()
