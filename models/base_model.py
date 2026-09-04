"""
models/base_model.py

Interfaz que debe cumplir cualquier "adaptador" de modelo para poder
enchufarse a este pipeline, sin importar el dataset que se use (CCPD u
otro).

Un adaptador envuelve un modelo concreto (RT-DETR de HuggingFace, o
cualquier otro detector que se agregue en el futuro) y expone una API
comun para que train.py pueda entrenarlo sin conocer sus detalles internos
(formato de anotaciones esperado, nombre de la clase de HF, etc.).

Para agregar un modelo nuevo: heredar de BaseModelAdapter, implementar
estos metodos y registrarlo con @register_model("nombre") en registry.py.
"""
from abc import ABC, abstractmethod
from typing import List


class BaseModelAdapter(ABC):
    @abstractmethod
    def build(self, class_names: List[str]) -> "BaseModelAdapter":
        """
        Instancia el modelo (y su preprocesador, si aplica) para el numero
        de clases del dataset actual. class_names viene de
        dataset.class_names (ver base_dataset.py). Debe devolver self.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def hf_model(self):
        """Modelo de HuggingFace (nn.Module), listo para pasarle a Trainer."""
        raise NotImplementedError

    @abstractmethod
    def collate_fn(self, batch):
        """
        Recibe una lista de (imagen_PIL, target) tal como los entrega el
        Dataset, y devuelve el diccionario de tensores que espera el
        forward() del modelo (pixel_values, labels, etc.). Este es el punto
        exacto donde se traduce el formato generico del dataset al formato
        especifico del modelo.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, output_dir: str) -> None:
        raise NotImplementedError
