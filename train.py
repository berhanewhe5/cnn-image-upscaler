import os
import sys
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from model import build_model, PixelShuffle


class SRDataset:
    """
    Super-resolution dataset.

    Supports two data sources (in priority order):
      1. Pre-extracted NumPy patch arrays
      2. Image directories
    """

    def __init__(
        self,
        hr_patches_path=None,
        lr_patches_path=None,
        hr_dir=None,
        lr_dir=None,
        scale_factor=2,
        architecture='espcn',
    ):
        self.architecture = architecture
        self.scale_factor = scale_factor

        if hr_patches_path and lr_patches_path:
            self.hr_data = np.load(hr_patches_path).astype(np.float32)
            self.lr_data = np.load(lr_patches_path).astype(np.float32)
        elif hr_dir and lr_dir:
            self.hr_data, self.lr_data = self._load_from_dirs(hr_dir, lr_dir)
        else:
            raise ValueError("Provide either (hr_patches_path, lr_patches_path) or (hr_dir, lr_dir).")

    def _load_from_dirs(self, hr_dir, lr_dir):
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

        hr_files = sorted([f for f in Path(hr_dir).iterdir() if f.suffix.lower() in exts])
        lr_files = sorted([f for f in Path(lr_dir).iterdir() if f.suffix.lower() in exts])

        assert len(hr_files) == len(lr_files), (
            f"HR and LR directories must have the same number of images "
            f"({len(hr_files)} vs {len(lr_files)})"
        )

        hr_data, lr_data = [], []
        for hf, lf in zip(hr_files, lr_files):
            hr_data.append(np.array(Image.open(hf).convert('RGB'), dtype=np.float32) / 255.0)
            lr_data.append(np.array(Image.open(lf).convert('RGB'), dtype=np.float32) / 255.0)

        return np.array(hr_data), np.array(lr_data)

    def get_tf_datasets(self, batch_size=16, val_split=0.1):
        """
        Split into train/val and return tf.data.Dataset objects.

        For SRCNN, the LR images are bicubic-upsampled to match HR size.
        """
        n = len(self.hr_data)
        val_size = max(1, int(n * val_split))
        train_size = n - val_size

        lr_input = self.lr_data
        if self.architecture == 'srcnn':
            hr_h, hr_w = self.hr_data.shape[1], self.hr_data.shape[2]
            lr_input = tf.image.resize(
                self.lr_data, [hr_h, hr_w], method='bicubic'
            ).numpy().astype(np.float32)
            lr_input = np.clip(lr_input, 0.0, 1.0)

        print(f"Samples — train: {train_size}  val: {val_size}")

        train_ds = (
            tf.data.Dataset.from_tensor_slices((lr_input[:train_size], self.hr_data[:train_size]))
            .shuffle(buffer_size=1000, seed=42)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        val_ds = (
            tf.data.Dataset.from_tensor_slices((lr_input[train_size:], self.hr_data[train_size:]))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        return train_ds, val_ds


def train(args):
    use_patches = args.use_patches

    if use_patches:
        patch_hr = Path(args.hr_patches_path)
        patch_lr = Path(args.lr_patches_path)

        if not patch_hr.exists() or not patch_lr.exists():
            raise FileNotFoundError(
                f"Patch files not found:\nHR: {patch_hr}\nLR: {patch_lr}"
            )

        print("Loading pre-extracted patch arrays...")
        dataset = SRDataset(
            hr_patches_path=str(patch_hr),
            lr_patches_path=str(patch_lr),
            scale_factor=args.scale,
            architecture=args.arch,
        )
    else:
        print("Loading images from directories...")
        print(f"HR dir: {args.hr_dir}")
        print(f"LR dir: {args.lr_dir}")

        dataset = SRDataset(
            hr_dir=args.hr_dir,
            lr_dir=args.lr_dir,
            scale_factor=args.scale,
            architecture=args.arch,
        )

    train_ds, val_ds = dataset.get_tf_datasets(batch_size=args.batch_size)

    resume_path = args.resume
    if resume_path and Path(resume_path).exists():
        print(f"Resuming from checkpoint: {resume_path}")
        model = tf.keras.models.load_model(
            resume_path,
            custom_objects={'PixelShuffle': PixelShuffle},
            compile=False,
        )
    else:
        model = build_model(args.arch, scale_factor=args.scale)

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.MeanAbsoluteError(),
        metrics=[tf.keras.metrics.MeanSquaredError(name='mse')],
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    best_ckpt = os.path.join(
        args.checkpoint_dir, f'{args.arch}_x{args.scale}_best.keras'
    )
    final_ckpt = os.path.join(
        args.checkpoint_dir, f'{args.arch}_x{args.scale}_final.keras'
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            best_ckpt,
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1,
        ),
    ]

    print(f"\nTraining {args.arch.upper()} x{args.scale} for up to {args.epochs} epochs\n{'-' * 60}")
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(final_ckpt)
    print(f"\nDone. Best model saved:  {best_ckpt}")
    print(f"Done. Final model saved: {final_ckpt}")
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train super-resolution ANN')

    parser.add_argument('--arch', type=str, default='espcn', choices=['srcnn', 'espcn'])
    parser.add_argument('--scale', type=int, default=2, choices=[2, 3, 4, 8])

    # Directory-based loading
    parser.add_argument('--train_hr_dir', type=str, default='Data/DIV2K_train_HR')
    parser.add_argument('--valid_hr_dir', type=str, default='Data/DIV2K_valid_HR')
    parser.add_argument('--train_lr_root', type=str, default='Data/DIV2K_train_LR_bicubic')
    parser.add_argument('--valid_lr_root', type=str, default='Data/DIV2K_valid_LR_bicubic')

    parser.add_argument('--hr_dir', type=str, default=None)
    parser.add_argument('--lr_dir', type=str, default=None)
    parser.add_argument('--use_valid_as_main', action='store_true')

    # Optional patch-based loading
    parser.add_argument('--use_patches', action='store_true')
    parser.add_argument('--hr_patches_path', type=str, default=None)
    parser.add_argument('--lr_patches_path', type=str, default=None)

    # Training params
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--checkpoint_dir', type=str, default='outputs/checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to a .keras checkpoint to resume training from')

    args = parser.parse_args()

    # Auto-build patch paths if patch mode is enabled
    if args.use_patches:
        if args.hr_patches_path is None:
            args.hr_patches_path = f'Data/hr_patches_x{args.scale}.npy'
        if args.lr_patches_path is None:
            args.lr_patches_path = f'Data/lr_patches_x{args.scale}.npy'
    else:
        # Auto-build image directory paths
        if args.use_valid_as_main:
            if args.hr_dir is None:
                args.hr_dir = args.valid_hr_dir
            if args.lr_dir is None:
                args.lr_dir = f'{args.valid_lr_root}/X{args.scale}'
        else:
            if args.hr_dir is None:
                args.hr_dir = args.train_hr_dir
            if args.lr_dir is None:
                args.lr_dir = f'{args.train_lr_root}/X{args.scale}'

    train(args)