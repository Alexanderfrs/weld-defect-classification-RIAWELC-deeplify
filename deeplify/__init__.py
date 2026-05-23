"""PyTorch Lightning project for RIAWELC weld defect classification."""

from importlib import import_module

__all__ = ["RIAWELCDataModule", "WeldDefectClassifier"]


def __getattr__(name: str):
    if name == "RIAWELCDataModule":
        return import_module("deeplify.data").RIAWELCDataModule
    if name == "WeldDefectClassifier":
        return import_module("deeplify.model").WeldDefectClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
