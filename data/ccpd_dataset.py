"""
data/ccpd_dataset.py

Implementacion de BaseDetectionDataset para CCPD (Chinese City Parking
Dataset): https://github.com/detectRecog/CCPD

CCPD no trae un archivo de anotaciones aparte (ni COCO ni Pascal VOC): el
bounding box, los 4 vertices, el numero de matricula, el brillo y la
nitidez estan codificados en el propio nombre de archivo. Formato oficial
(ver README del repo):

    area-tilt-bbox-vertices-platenumber-brightness-blurriness.jpg

Ejemplo:
    025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg

Este archivo solo se encarga de convertir ese formato a la interfaz generica
(imagen, {"boxes": [...], "labels": [...]}) que espera el resto del
pipeline. No depende de RT-DETR ni de ningun otro modelo en particular.

IMPORTANTE: en este entorno no se recibio el dataset descargado (la carpeta
de uploads estaba vacia), asi que la implementacion sigue al pie de la letra
la especificacion publicada en el repositorio. Si tu copia local de CCPD
tiene una estructura de carpetas distinta a la oficial, revisa
`_infer_class_from_path` y `_load_from_split_file` mas abajo.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from data.base_dataset import BaseDetectionDataset
from registry import register_dataset

# Provincias chinas tal como las define la especificacion oficial de CCPD.
# El primer caracter de toda matricula china es una de estas 34 opciones
# (la ultima, "O", significa "sin caracter" y no deberia aparecer en la
# practica salvo error de anotacion).
PROVINCES = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京",
    "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏",
    "陕", "甘", "青", "宁", "新", "警", "学", "O",
]

# Subconjuntos oficiales publicados por los autores de CCPD (carpetas del
# .tar.xz). Se usan como nombre de clase por defecto (class_mode="subset").
KNOWN_CCPD_SUBSETS = [
    "ccpd_base", "ccpd_blur", "ccpd_challenge", "ccpd_db", "ccpd_fn",
    "ccpd_green", "ccpd_np", "ccpd_rotate", "ccpd_tilt", "ccpd_weather",
]


def _parse_ccpd_filename(path: Path) -> Optional[Dict]:
    """Extrae bbox y numero de matricula del nombre de archivo de CCPD."""
    fields = path.stem.split("-")
    if len(fields) < 5:
        # Algunos archivos (p.ej. negativos en ccpd_np) no siguen la
        # convencion completa; se descartan en vez de romper el entrenamiento.
        return None
    try:
        p1_str, p2_str = fields[2].split("_")
        x1, y1 = (int(v) for v in p1_str.split("&"))
        x2, y2 = (int(v) for v in p2_str.split("&"))
        plate_indices = [int(v) for v in fields[4].split("_")]
    except (ValueError, IndexError):
        return None

    return {
        "bbox": [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)],
        "plate_indices": plate_indices,
    }


@register_dataset("ccpd")
class CCPDDataset(BaseDetectionDataset):
    """
    class_mode define que se usa como "clase" de la (unica) caja detectada
    en cada imagen:

      - "subset"   (default): la clase es la carpeta de origen de la imagen
                    (ccpd_base, ccpd_blur, ccpd_db, ccpd_fn, ccpd_rotate,
                    ccpd_tilt, ccpd_challenge, ccpd_green, ...), es decir,
                    los distintos escenarios/condiciones que distingue el
                    dataset.
      - "single":  todas las matriculas son una unica clase "license_plate"
                    (deteccion pura, sin distincion de subconjunto).
      - "province": la clase es la provincia china codificada en el primer
                    caracter de la matricula (hasta 34 clases).
    """

    def __init__(
        self,
        root_dir: str,
        class_mode: str = "subset",
        split_file: Optional[str] = None,
        image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
        transform=None,
    ):
        if class_mode not in ("subset", "single", "province"):
            raise ValueError(f"class_mode desconocido: {class_mode}")

        self.root_dir = Path(root_dir)
        self.class_mode = class_mode
        self.transform = transform
        self.image_extensions = tuple(e.lower() for e in image_extensions)

        self._paths = (
            self._load_from_split_file(split_file)
            if split_file
            else self._scan_directory()
        )

        self._class_names = self._build_class_list()
        self._class_to_idx = {c: i for i, c in enumerate(self._class_names)}

        # Se parsea una sola vez al construir el dataset (no en cada
        # __getitem__), descartando silenciosamente los archivos que no
        # siguen la convencion de nombres de CCPD.
        self._samples = []
        for p in self._paths:
            parsed = _parse_ccpd_filename(p)
            if parsed is None:
                continue
            class_name = self._infer_class_from_path(p, parsed)
            if class_name not in self._class_to_idx:
                continue
            self._samples.append((p, parsed, class_name))

        if len(self._samples) == 0:
            raise RuntimeError(
                f"No se encontraron imagenes validas de CCPD en '{root_dir}'. "
                "Verifica root_dir, split_file y que los nombres de archivo "
                "sigan el formato oficial de CCPD."
            )

    # ------------------------------------------------------------------ #
    # Descubrimiento de archivos
    # ------------------------------------------------------------------ #
    def _scan_directory(self) -> List[Path]:
        return sorted(
            p for p in self.root_dir.rglob("*")
            if p.suffix.lower() in self.image_extensions
        )

    def _load_from_split_file(self, split_file: str) -> List[Path]:
        """
        Los archivos de la carpeta split/ de CCPD (train.txt, val.txt,
        test.txt) listan una ruta relativa por linea, p.ej.:
            ccpd_base/0116-3_11-... .jpg
        """
        lines = Path(split_file).read_text(encoding="utf-8").splitlines()
        paths = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            candidate = self.root_dir / line
            if not candidate.exists():
                # fallback: buscar por nombre de archivo si el split no
                # incluye la subcarpeta o la estructura local difiere
                matches = list(self.root_dir.rglob(Path(line).name))
                if matches:
                    candidate = matches[0]
            paths.append(candidate)
        return [p for p in paths if p.exists() and p.suffix.lower() in self.image_extensions]

    # ------------------------------------------------------------------ #
    # Resolucion de clases
    # ------------------------------------------------------------------ #
    def _infer_class_from_path(self, path: Path, parsed: Dict) -> str:
        if self.class_mode == "single":
            return "license_plate"
        if self.class_mode == "province":
            idx = parsed["plate_indices"][0]
            return PROVINCES[idx] if idx < len(PROVINCES) else "unknown"
        return path.parent.name.lower()  # class_mode == "subset"

    def _build_class_list(self) -> List[str]:
        if self.class_mode == "single":
            return ["license_plate"]
        if self.class_mode == "province":
            return list(PROVINCES)
        # subset: se listan los subconjuntos oficiales que realmente
        # aparecen en root_dir, mas cualquier carpeta extra (por si el
        # usuario organizo su copia local con otros nombres).
        found = {p.parent.name.lower() for p in self._paths}
        ordered = [s for s in KNOWN_CCPD_SUBSETS if s in found]
        extra = sorted(found - set(ordered))
        return ordered + extra

    # ------------------------------------------------------------------ #
    # Interfaz BaseDetectionDataset
    # ------------------------------------------------------------------ #
    @property
    def class_names(self) -> List[str]:
        return self._class_names

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        path, parsed, class_name = self._samples[idx]
        image = Image.open(path).convert("RGB")
        target = {
            "image_id": idx,
            "boxes": [parsed["bbox"]],
            "labels": [self._class_to_idx[class_name]],
        }
        if self.transform is not None:
            image, target = self.transform(image, target)
        return image, target
