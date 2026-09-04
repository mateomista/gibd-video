#!/usr/bin/env python
"""
predict.py

Inferencia sobre una imagen suelta usando un modelo ya entrenado con
train.py. Sirve tambien para mostrar que, gracias al desacople, cargar y
usar el modelo no depende de la clase de dataset usada en el entrenamiento.

Uso:
    python predict.py --model_dir outputs/rtdetr-ccpd/final_model --image foto.jpg
"""
import argparse
import os

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

import torch
from PIL import Image, ImageDraw
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Carpeta con el modelo guardado (final_model).")
    parser.add_argument("--image", required=True, help="Ruta a la imagen de entrada.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default="prediccion.jpg", help="Donde guardar la imagen con las cajas dibujadas.")
    args = parser.parse_args()

    processor = RTDetrImageProcessor.from_pretrained(args.model_dir)
    model = RTDetrForObjectDetection.from_pretrained(args.model_dir)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_object_detection(
        outputs, target_sizes=torch.tensor([image.size[::-1]]), threshold=args.threshold
    )[0]

    draw = ImageDraw.Draw(image)
    for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
        x1, y1, x2, y2 = [round(v) for v in box.tolist()]
        class_name = model.config.id2label[label_id.item()]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1 - 12)), f"{class_name} {score:.2f}", fill="red")
        print(f"{class_name}: {score:.2f} -> [{x1}, {y1}, {x2}, {y2}]")

    image.save(args.output)
    print(f"[ok] Imagen guardada en: {args.output}")


if __name__ == "__main__":
    main()
