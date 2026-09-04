#!/usr/bin/env python
"""
train.py

Punto de entrada unico para entrenar un modelo de deteccion. Ni el dataset
ni el modelo estan hardcodeados aca: train.py solo conoce las interfaces
BaseDetectionDataset y BaseModelAdapter, y le pide instancias concretas al
registry.py segun lo que diga config.yaml.

Para repetir el mismo experimento con otro dataset u otro modelo:
  1. Dataset propio: heredar de data.base_dataset.BaseDetectionDataset
     (ver data/ccpd_dataset.py como ejemplo) y registrarlo con
     @register_dataset("mi_dataset").
  2. Modelo propio: heredar de models.base_model.BaseModelAdapter (ver
     models/rtdetr_model.py como ejemplo) y registrarlo con
     @register_model("mi_modelo").
  3. Importar el archivo nuevo mas abajo (junto a los imports de
     data.ccpd_dataset / models.rtdetr_model) para que el decorador se
     ejecute, y cambiar dataset.name / model.name en config.yaml.
No hace falta modificar el resto de este script.

Uso:
    python train.py --config config.yaml
"""
import argparse
import os
from pathlib import Path

# IMPORTANTE: hay que setear esto ANTES de importar transformers/huggingface_hub.
# Si esta maquina tiene una sesion guardada (huggingface-cli login), por
# defecto se manda ese token en TODAS las llamadas al Hub, incluso a repos
# publicos como "PekingU/rtdetr_r50vd". Si ese token esta vencido, revocado
# o simplemente no tiene permisos, la descarga falla con 401 aunque el repo
# sea publico. Esta variable desactiva ese envio implicito.
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

import yaml
from torch.utils.data import Subset, random_split
from transformers import Trainer, TrainingArguments

# Importar los modulos concretos registra sus clases en registry.py (via los
# decoradores @register_dataset / @register_model). Para usar otro dataset u
# otro modelo, agregar aca su import (o mover estos imports a un
# __init__.py que los cargue todos automaticamente).
import data.ccpd_dataset  # noqa: F401
import models.rtdetr_model  # noqa: F401
from registry import build_dataset, build_model_adapter
from utils import set_seed


class _SplitView:
    """
    Envoltorio minimo sobre torch.utils.data.Subset que conserva el
    atributo `class_names` del dataset original. Se necesita porque
    random_split() devuelve un Subset "pelado" que no expone class_names,
    y el adaptador de modelo necesita esa lista para construir la cabeza
    de clasificacion con el numero de clases correcto.
    """

    def __init__(self, subset: Subset, class_names):
        self._subset = subset
        self.class_names = class_names

    def __len__(self):
        return len(self._subset)

    def __getitem__(self, idx):
        return self._subset[idx]


def build_train_val_datasets(cfg: dict):
    """
    Construye los datasets de train y val a partir de dataset.* en la
    config. Si train_params/val_params traen ambos un split_file (como los
    train.txt/val.txt oficiales de CCPD), se usan tal cual. Si no, se arma
    un unico dataset con common_params y se separa aleatoriamente segun
    dataset.val_ratio.
    """
    ds_cfg = cfg["dataset"]
    name = ds_cfg["name"]
    common = ds_cfg.get("common_params", {})
    train_params = ds_cfg.get("train_params", {}) or {}
    val_params = ds_cfg.get("val_params", {}) or {}

    # Se consideran "splits explicitos" tanto el caso de un split_file (como
    # los train.txt/val.txt de CCPD2019) como el caso de root_dir separados
    # por split (como CCPD2020, que ya viene con carpetas train/ y val/
    # fisicamente distintas). Alcanza con que train_params y val_params
    # traigan algo (split_file o root_dir propios) para no caer en el split
    # aleatorio de mas abajo.
    has_explicit_splits = bool(train_params) and bool(val_params)

    if has_explicit_splits:
        train_ds = build_dataset(name, **{**common, **train_params})
        val_ds = build_dataset(name, **{**common, **val_params})
        return train_ds, val_ds

    full_ds = build_dataset(name, **common)
    val_ratio = ds_cfg.get("val_ratio", 0.1)
    n_val = max(1, int(len(full_ds) * val_ratio))
    n_train = len(full_ds) - n_val
    train_subset, val_subset = random_split(full_ds, [n_train, n_val])
    return (
        _SplitView(train_subset, full_ds.class_names),
        _SplitView(val_subset, full_ds.class_names),
    )


def main():
    parser = argparse.ArgumentParser(description="Entrena un detector, desacoplado de dataset y modelo.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Ruta al archivo de configuracion.")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(cfg.get("seed", 42))

    # --- Dataset (elegido por config, no por codigo) ---
    train_ds, val_ds = build_train_val_datasets(cfg)
    class_names = train_ds.class_names
    print(f"[dataset] '{cfg['dataset']['name']}' -> {len(class_names)} clases: {class_names}")
    print(f"[dataset] train: {len(train_ds)} ejemplos | val: {len(val_ds)} ejemplos")

    # --- Modelo (elegido por config, no por codigo) ---
    model_cfg = cfg["model"]
    adapter = build_model_adapter(model_cfg["name"], **model_cfg.get("params", {}))
    adapter.build(class_names)
    print(f"[modelo] '{model_cfg['name']}' construido con checkpoint '{model_cfg['params'].get('checkpoint')}'")

    # --- Entrenamiento (via transformers.Trainer, funciona igual sin
    #     importar que dataset/modelo se hayan elegido arriba) ---
    tr_cfg = cfg.get("training", {})
    training_args = TrainingArguments(
        output_dir=cfg.get("output_dir", "./outputs"),
        num_train_epochs=tr_cfg.get("num_train_epochs", 20),
        per_device_train_batch_size=tr_cfg.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=tr_cfg.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=tr_cfg.get("gradient_accumulation_steps", 1),
        learning_rate=tr_cfg.get("learning_rate", 1e-4),
        weight_decay=tr_cfg.get("weight_decay", 1e-4),
        warmup_ratio=tr_cfg.get("warmup_ratio", 0.05),
        lr_scheduler_type=tr_cfg.get("lr_scheduler_type", "cosine"),
        logging_steps=tr_cfg.get("logging_steps", 50),
        eval_strategy=tr_cfg.get("eval_strategy", "epoch"),
        save_strategy=tr_cfg.get("save_strategy", "epoch"),
        save_total_limit=tr_cfg.get("save_total_limit", 3),
        fp16=tr_cfg.get("fp16", False),
        bf16=tr_cfg.get("bf16", False),
        dataloader_num_workers=tr_cfg.get("dataloader_num_workers", 4),
        report_to=tr_cfg.get("report_to", "none"),
        # Obligatorio: nuestro dataset no entrega columnas con nombre que
        # coincidan con la firma del modelo, sino (imagen, target) crudos
        # que el data_collator del adaptador convierte al formato correcto.
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=adapter.hf_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=adapter.collate_fn,
    )

    trainer.train()

    final_dir = str(Path(training_args.output_dir) / "final_model")
    adapter.save(final_dir)
    print(f"[ok] Modelo final y procesador guardados en: {final_dir}")


if __name__ == "__main__":
    main()
