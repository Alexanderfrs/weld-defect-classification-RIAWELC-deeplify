from __future__ import annotations

import argparse

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from deeplify.data import RIAWELCDataModule
from deeplify.model import WeldDefectClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a RIAWELC weld defect classifier.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to RIAWELC images")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--project", type=str, default="riawelc-defect-classification")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")
    return parser.parse_args()


def parse_devices(devices: str) -> str | int | list[int]:
    if devices == "auto":
        return devices
    if "," in devices:
        return [int(device.strip()) for device in devices.split(",") if device.strip()]
    return int(devices)


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    datamodule = RIAWELCDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    model = WeldDefectClassifier(
        num_classes=4,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pretrained=not args.no_pretrained,
    )

    logger = WandbLogger(project=args.project, name=args.run_name)
    callbacks = [
        ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1, filename="best-{epoch}-{val_acc:.4f}"),
        EarlyStopping(monitor="val_acc", mode="max", patience=8),
    ]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=parse_devices(args.devices),
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
        fast_dev_run=args.fast_dev_run,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
