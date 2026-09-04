"""
registry.py

Registro central de datasets y modelos.

Esta es la pieza clave del desacople pedido: train.py nunca importa
"CCPDDataset" ni "RTDetrAdapter" directamente, sino que le pide al registry
la clase asociada al nombre que figura en config.yaml (dataset.name /
model.name). Para correr el mismo experimento con otro dataset u otro
modelo, alcanza con:

  1. Crear el archivo nuevo (p.ej. data/mi_dataset.py o
     models/mi_modelo.py) implementando la interfaz correspondiente
     (BaseDetectionDataset / BaseModelAdapter).
  2. Decorar la clase con @register_dataset("mi_dataset") o
     @register_model("mi_modelo").
  3. Importarlo una vez en train.py (o en un __init__.py) para que el
     decorador se ejecute y quede registrado.
  4. Cambiar el "name" correspondiente en config.yaml.

No hace falta tocar el resto del pipeline.
"""
from typing import Callable, Dict, Type

DATASET_REGISTRY: Dict[str, Type] = {}
MODEL_REGISTRY: Dict[str, Type] = {}


def register_dataset(name: str) -> Callable:
    def _decorator(cls: Type) -> Type:
        if name in DATASET_REGISTRY:
            raise ValueError(f"Dataset '{name}' ya esta registrado.")
        DATASET_REGISTRY[name] = cls
        return cls
    return _decorator


def register_model(name: str) -> Callable:
    def _decorator(cls: Type) -> Type:
        if name in MODEL_REGISTRY:
            raise ValueError(f"Modelo '{name}' ya esta registrado.")
        MODEL_REGISTRY[name] = cls
        return cls
    return _decorator


def build_dataset(name: str, **kwargs):
    if name not in DATASET_REGISTRY:
        raise KeyError(
            f"Dataset '{name}' no registrado. Disponibles: {list(DATASET_REGISTRY)}"
        )
    return DATASET_REGISTRY[name](**kwargs)


def build_model_adapter(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Modelo '{name}' no registrado. Disponibles: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](**kwargs)
