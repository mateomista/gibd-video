# Entrenamiento desacoplado (dataset ↔ modelo) — CCPD + RT-DETR

Pipeline para entrenar un detector sobre el dataset [CCPD](https://github.com/detectRecog/CCPD)
usando **RT-DETR**, diseñado para que cambiar de dataset o de modelo sea
cuestión de editar `config.yaml`, sin tocar el resto del código.

## ⚠️ Aviso importante: el dataset no llegó a este entorno

Dijiste que adjuntabas el dataset, pero la carpeta de archivos subidos
llegó vacía — no tuve acceso a ningún archivo de CCPD. Todo lo que sigue
(parseo de nombres de archivo, estructura de carpetas, splits) está
implementado siguiendo **al pie de la letra la especificación publicada en
el repositorio oficial** (https://github.com/detectRecog/CCPD), no
verificado contra tus archivos reales. Si tu copia local difiere un poco
(por ejemplo, otra organización de carpetas), probablemente solo haga
falta ajustar `_infer_class_from_path` / `_load_from_split_file` en
`data/ccpd_dataset.py`. Te recomiendo correr `train.py` primero contra una
carpeta chica de prueba antes de lanzar el entrenamiento completo.

## ⚠️ Nota sobre la licencia de RT-DETR

Pediste específicamente un módulo con licencia que permita uso comercial
(tipo Apache-2.0). Ojo con esto porque **hay dos implementaciones de
RT-DETR muy distintas en cuanto a licencia**:

| Implementación | Licencia | ¿Sirve para uso comercial cerrado? |
|---|---|---|
| `ultralytics` (paquete `pip install ultralytics`, clase `RTDETR`) | **AGPL-3.0** + licencia comercial paga para producto cerrado | No, sin pagarles una licencia |
| `transformers` de HuggingFace (`RTDetrForObjectDetection`, checkpoints `PekingU/*`) | **Apache-2.0** | Sí |

Este proyecto usa la **segunda opción** (HuggingFace Transformers), que es
la que realmente permite uso comercial sin restricciones de copyleft. Todas
las demás dependencias (`torch`, `torchvision`, `pyyaml`, `pillow`,
`numpy`, `accelerate`) también son de licencia permisiva (BSD/MIT/Apache-2.0).

## Estructura del proyecto

```
config.yaml              # ÚNICO lugar donde se elige dataset y modelo a usar
registry.py               # registro central: nombre (string) -> clase concreta
train.py                  # script de entrenamiento, agnóstico a dataset/modelo
predict.py                 # inferencia rápida sobre una imagen
utils.py                   # semilla aleatoria, etc.

data/
  base_dataset.py          # interfaz que debe cumplir cualquier dataset
  ccpd_dataset.py           # implementación concreta para CCPD

models/
  base_model.py             # interfaz que debe cumplir cualquier modelo
  rtdetr_model.py            # implementación concreta para RT-DETR (HF, Apache-2.0)
```

**Cómo funciona el desacople:** `train.py` sólo conoce las interfaces
`BaseDetectionDataset` y `BaseModelAdapter`. Nunca importa `CCPDDataset` ni
`RTDetrAdapter` por su nombre de clase; se los pide a `registry.py` usando
el string que aparece en `config.yaml` (`dataset.name` / `model.name`).

## Cómo se decodifican las anotaciones de CCPD

CCPD no trae un archivo de anotaciones aparte: cada nombre de imagen
codifica el bounding box, los 4 vértices, el número de matrícula, brillo y
nitidez. Ejemplo oficial:

```
025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
```

`data/ccpd_dataset.py` parsea ese nombre y arma el bounding box. Como
"clase" del objeto detectado (necesaria para RT-DETR), hay tres modos
configurables vía `dataset.common_params.class_mode`:

- **`"subset"`** (default): la clase es la carpeta de origen de la imagen
  (`ccpd_base`, `ccpd_blur`, `ccpd_db`, `ccpd_fn`, `ccpd_rotate`,
  `ccpd_tilt`, `ccpd_challenge`, `ccpd_green`, ...) — es decir, los
  distintos escenarios/condiciones que distingue el propio dataset.
- **`"single"`**: todas las matrículas son una única clase
  `license_plate` (detección pura, sin distinguir subconjuntos).
- **`"province"`**: la clase es la provincia china codificada en el primer
  carácter de la matrícula (hasta 34 clases).

Elegí `"subset"` como default porque es la lectura más literal de "las
distintas clases que contiene el dataset" para un dataset que, en sí
mismo, sólo tiene un tipo de objeto (matrícula). Si lo que buscabas era
detección pura de matrículas sin distinguir origen, cambiá a `"single"`.

## Instalación

```bash
pip install -r requirements.txt
```

## Configuración

Editá `config.yaml`:

```yaml
dataset:
  name: "ccpd"
  common_params:
    root_dir: "/ruta/a/CCPD2019"
    class_mode: "subset"        # "subset" | "single" | "province"
  train_params:
    split_file: "/ruta/a/CCPD2019/split/train.txt"   # opcional
  val_params:
    split_file: "/ruta/a/CCPD2019/split/val.txt"      # opcional

model:
  name: "rtdetr"
  params:
    checkpoint: "PekingU/rtdetr_r50vd"   # Apache-2.0
    image_size: 640
```

Si no tenés los `split/train.txt` / `split/val.txt` oficiales, dejá
`train_params`/`val_params` sin `split_file`: el script arma un único
dataset y separa automáticamente `dataset.val_ratio` (10% por defecto) al
azar.

## Entrenar

```bash
python train.py --config config.yaml
```

El modelo final (pesos + preprocesador) queda en
`<output_dir>/final_model/`.

## Inferencia

```bash
python predict.py --model_dir outputs/rtdetr-ccpd/final_model --image alguna_foto.jpg
```

## Repetir el experimento con otro dataset o modelo

**Otro dataset** (p. ej. otro dataset de matrículas, o uno completamente
distinto):
1. Crear `data/mi_dataset.py`, heredar de `BaseDetectionDataset`
   (`data/base_dataset.py`), implementar `class_names`, `__len__`,
   `__getitem__`.
2. Decorar la clase con `@register_dataset("mi_dataset")`.
3. Importar el archivo en `train.py` (agregar `import data.mi_dataset`).
4. En `config.yaml`, poner `dataset.name: "mi_dataset"` y sus `params`.

**Otro modelo** (p. ej. otro detector con licencia permisiva):
1. Crear `models/mi_modelo.py`, heredar de `BaseModelAdapter`
   (`models/base_model.py`), implementar `build`, `hf_model`,
   `collate_fn`, `save`.
2. Decorar la clase con `@register_model("mi_modelo")`.
3. Importar el archivo en `train.py`.
4. En `config.yaml`, poner `model.name: "mi_modelo"` y sus `params`.

`train.py` no cambia en ningún caso.

## Limitaciones conocidas

- La métrica de validación que usa `Trainer` por defecto es la **loss**,
  no mAP. Para una evaluación tipo COCO (AP@0.5, etc.) habría que agregar
  `torchmetrics.detection.MeanAveragePrecision` sobre las predicciones de
  `model.eval()`, no incluido aquí para mantener el script enfocado en
  entrenamiento.
- No se incluyen augmentations (flip, brightness, etc.). Para CCPD en
  particular, **cuidado con el flip horizontal**: invierte el número de
  matrícula, así que si en algún momento se agrega un modelo de
  reconocimiento de caracteres sobre esto, no correspondería usarlo.
