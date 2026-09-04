"""
models/rtdetr_model.py

Adaptador de RT-DETR usando la implementacion oficial incluida en
HuggingFace Transformers.

Nota sobre licencias (importante para uso comercial):
------------------------------------------------------
Existen dos implementaciones distintas de RT-DETR muy usadas:

  1. Ultralytics (paquete "ultralytics", clase RTDETR): el CODIGO del
     paquete es AGPL-3.0, y Ultralytics ademas exige una licencia comercial
     paga para uso en productos cerrados. NO es lo que pediste.

  2. HuggingFace Transformers (clases RTDetrForObjectDetection /
     RTDetrImageProcessor, usadas aqui): tanto la libreria "transformers"
     como los checkpoints preentrenados publicados por el equipo de RT-DETR
     bajo el usuario "PekingU" en el Hub estan licenciados Apache-2.0, que
     SI permite uso comercial sin restricciones de copyleft. Es la opcion
     usada en este adaptador.

Checkpoints Apache-2.0 disponibles (organizacion "PekingU" en HF Hub):
    PekingU/rtdetr_r18vd
    PekingU/rtdetr_r34vd
    PekingU/rtdetr_r50vd        <- default
    PekingU/rtdetr_r101vd
    PekingU/rtdetr_v2_r18vd / r34vd / r50vd / r101vd   (RT-DETRv2)
"""
from typing import List, Optional, Union

from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

from models.base_model import BaseModelAdapter
from registry import register_model


@register_model("rtdetr")
class RTDetrAdapter(BaseModelAdapter):
    def __init__(
        self,
        checkpoint: str = "PekingU/rtdetr_r50vd",
        image_size: int = 640,
        hf_token: Optional[Union[str, bool]] = False,
    ):
        """
        hf_token: por defecto False, lo que le indica a huggingface_hub que
        NO use ningun token guardado localmente (variable de entorno,
        `huggingface-cli login`, etc.). Los checkpoints "PekingU/*" son
        publicos y no necesitan autenticacion; forzar False evita que un
        token viejo/vencido guardado en la maquina rompa la descarga con un
        401 aunque el repo sea publico. Si en el futuro se usa un
        checkpoint privado propio, pasar aca el token real como string.
        """
        self.checkpoint = checkpoint
        self.image_size = image_size
        self.hf_token = hf_token
        self._model = None
        self._processor = None

    def build(self, class_names: List[str]) -> "RTDetrAdapter":
        id2label = {i: name for i, name in enumerate(class_names)}
        label2id = {name: i for i, name in enumerate(class_names)}

        self._processor = RTDetrImageProcessor.from_pretrained(
            self.checkpoint,
            size={"height": self.image_size, "width": self.image_size},
            token=self.hf_token,
        )
        self._model = RTDetrForObjectDetection.from_pretrained(
            self.checkpoint,
            token=self.hf_token,
            num_labels=len(class_names),
            id2label=id2label,
            label2id=label2id,
            # la cabeza de clasificacion del checkpoint preentrenado (80
            # clases COCO) no coincide en tamaño con nuestras clases nuevas
            ignore_mismatched_sizes=True,
        )
        return self

    @property
    def hf_model(self):
        if self._model is None:
            raise RuntimeError("Llama a build(class_names) antes de usar el modelo.")
        return self._model

    @property
    def processor(self):
        if self._processor is None:
            raise RuntimeError("Llama a build(class_names) antes de usar el procesador.")
        return self._processor

    def collate_fn(self, batch):
        """
        Traduce (imagen_PIL, target generico) -> tensores para RT-DETR.

        RTDetrImageProcessor (igual que el procesador de DETR, del cual
        hereda) construye automaticamente pixel_values y labels a partir de
        anotaciones en formato COCO: bbox = [x, y, w, h] en pixeles, mas
        category_id por caja. Aqui simplemente convertimos nuestro formato
        generico (boxes en xyxy) a ese formato antes de llamar al procesador.
        """
        images, targets = zip(*batch)

        coco_annotations = []
        for target in targets:
            annotations = []
            for (x1, y1, x2, y2), label in zip(target["boxes"], target["labels"]):
                annotations.append({
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "category_id": label,
                    "area": max(0.0, (x2 - x1) * (y2 - y1)),
                    "iscrowd": 0,
                })
            coco_annotations.append({
                "image_id": target["image_id"],
                "annotations": annotations,
            })

        encoding = self.processor(
            images=list(images),
            annotations=coco_annotations,
            return_tensors="pt",
        )

        batch_out = {"pixel_values": encoding["pixel_values"], "labels": encoding["labels"]}
        if "pixel_mask" in encoding:
            batch_out["pixel_mask"] = encoding["pixel_mask"]
        return batch_out

    def save(self, output_dir: str) -> None:
        self.hf_model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
