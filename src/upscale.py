import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from model import PixelShuffle


def upscale(
    input_path,
    checkpoint_path=None,
    output_path=None,
    comparison_path=None,
    scale=2,
    arch='espcn',
    checkpoint_dir='outputs/checkpoints',
):
    """
    Upscale a single image using a trained ESPCN/SRCNN model.

    Parameters
    ----------
    input_path      : path to the low-resolution input image
    checkpoint_path : path to the .keras model file
                      if None, auto-loads from checkpoint_dir using arch + scale
    output_path     : where to save the SR image
    comparison_path : where to save a side-by-side comparison image
    scale           : upscale factor (2, 3, 4, or 8)
    arch            : model architecture — 'espcn' or 'srcnn'
    checkpoint_dir  : folder containing trained checkpoints
    """
    input_path = Path(input_path)

    if checkpoint_path is None:
        checkpoint_path = Path(checkpoint_dir) / f"{arch}_x{scale}_best.keras"
    else:
        checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = tf.keras.models.load_model(
        checkpoint_path,
        custom_objects={'PixelShuffle': PixelShuffle},
        compile=False,
    )
    print(f"Loaded  : {checkpoint_path} ({arch.upper()}, x{scale})")

    lr_img = Image.open(input_path).convert('RGB')
    orig_w, orig_h = lr_img.size
    print(f"Input   : {input_path} ({orig_w}×{orig_h})")

    lr_arr = np.array(lr_img, dtype=np.float32) / 255.0
    lr_tensor = lr_arr[np.newaxis, ...]  # (1, H, W, 3)

    if arch == 'srcnn':
        lr_tensor = tf.image.resize(
            lr_tensor,
            [orig_h * scale, orig_w * scale],
            method='bicubic'
        ).numpy()
        lr_tensor = np.clip(lr_tensor, 0.0, 1.0)

    sr_arr = model.predict(lr_tensor, verbose=0)[0]
    sr_arr = np.clip(sr_arr, 0.0, 1.0)
    sr_img = Image.fromarray((sr_arr * 255).astype(np.uint8))
    sr_w, sr_h = sr_img.size

    input_stem = input_path.stem
    input_ext = input_path.suffix if input_path.suffix else '.png'

    if output_path is None:
        output_path = input_path.parent / f"{input_stem}_{arch}_x{scale}_sr{input_ext}"
    else:
        output_path = Path(output_path)

    sr_img.save(output_path)
    print(f"SR image: {output_path} ({sr_w}×{sr_h})")

    # Bicubic baseline for comparison
    bicubic_img = lr_img.resize((orig_w * scale, orig_h * scale), Image.BICUBIC)

    # Resize LR preview so all three panels match in displayed size
    lr_preview = lr_img.resize((orig_w * scale, orig_h * scale), Image.NEAREST)

    padding = 12
    label_space = 28
    panel_w = lr_preview.width
    panel_h = lr_preview.height

    comp_w = panel_w * 3 + padding * 4
    comp_h = panel_h + padding * 2 + label_space
    comparison = Image.new('RGB', (comp_w, comp_h), (30, 30, 30))

    x = padding
    y = padding
    comparison.paste(lr_preview, (x, y))
    x += panel_w + padding
    comparison.paste(bicubic_img, (x, y))
    x += panel_w + padding
    comparison.paste(sr_img, (x, y))

    try:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(comparison)

        labels = [
            (padding, panel_h + padding + 6, 'LR Input'),
            (panel_w + padding * 2, panel_h + padding + 6, 'Bicubic'),
            (panel_w * 2 + padding * 3, panel_h + padding + 6, f'{arch.upper()} x{scale}'),
        ]

        for lx, ly, text in labels:
            draw.text((lx, ly), text, fill=(220, 220, 220))
    except Exception as e:
        print(f"Warning: could not draw labels on comparison image: {e}")

    if comparison_path is None:
        comparison_path = input_path.parent / f"{input_stem}_{arch}_x{scale}_compare{input_ext}"
    else:
        comparison_path = Path(comparison_path)

    comparison.save(comparison_path)
    print(f"Compare : {comparison_path} (LR | Bicubic | {arch.upper()} x{scale})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Upscale a single image with a trained ESPCN/SRCNN model'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to the low-resolution input image'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Optional path to .keras model file; if omitted, uses outputs/checkpoints/{arch}_x{scale}_best.keras'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='outputs/checkpoints',
        help='Directory containing trained checkpoints'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for the SR image'
    )
    parser.add_argument(
        '--comparison',
        type=str,
        default=None,
        help='Output path for the side-by-side comparison image'
    )
    parser.add_argument(
        '--scale',
        type=int,
        default=2,
        choices=[2, 3, 4, 8],
        help='Upscale factor'
    )
    parser.add_argument(
        '--arch',
        type=str,
        default='espcn',
        choices=['espcn', 'srcnn'],
        help='Model architecture'
    )

    args = parser.parse_args()

    upscale(
        input_path=args.input,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        comparison_path=args.comparison,
        scale=args.scale,
        arch=args.arch,
        checkpoint_dir=args.checkpoint_dir,
    )
