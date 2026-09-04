"""
data/base_dataset.py

Interfaz que debe cumplir cualquier dataset de deteccion de objetos para
poder enchufarse a este pipeline, sin importar que modelo se entrene sobre
el (RT-DETR u otro).

Contrato:
  - class_names (property): lista ordenada de nombres de clase. La posicion
    de cada nombre en la lista ES el id de clase (entero, 0-based) usado
    para entrenar.
  - __len__
  - __getitem__(idx) -> (imagen, target), con:
        imagen: PIL.Image en modo RGB
        target: {
            "image_id": int,
            "boxes":  [[x1, y1, x2, y2], ...]   # pixeles, formato pascal_voc (xyxy)
            "labels": [int, ...]                 # un id de clase por caja
        }

Para agregar un dataset nuevo (otro dataset de matriculas, COCO, uno propio,
etc.) alcanza con heredar de esta clase, implementar estos metodos y
registrarla en registry.py con @register_dataset("nombre"). El resto del
pipeline (train.py, el adaptador de modelo) no necesita cambios.
"""
from abc import ABC, abstractmethod
from typing import List

from torch.utils.data import Dataset


class BaseDetectionDataset(Dataset, ABC):
    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int):
        raise NotImplementedError
