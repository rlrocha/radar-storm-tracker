import os
import glob
import json
import numpy as np
import cv2

def dbz_to_norm(dbz: float) -> float:
    """Converte valor em dBZ para a escala normalizada [0.0, 1.0] usada no grid."""
    return float(np.clip(dbz, 0.0, 75.0) / 75.0)

def extract_storm_bboxes(
    npy_path: str,
    coords_dict: dict,
    min_dbz_threshold: float = 45.0,  # Limiar para convecção severa
    min_area_pixels: int = 16,        # Ignora núcleos minúsculos/ruído (< ~20 km²)
    closing_kernel_size: int = 3      # Kernel para fechar pequenos vazios na nuvem
):
    """
    Processa um array normalizado de radar e extrai caixas delimitadoras.
    Retorna caixas em pixels [xmin, ymin, xmax, ymax], formato YOLO e dados geográficos.
    """
    # Carrega array normalizado (256, 256)
    grid_norm = np.load(npy_path)
    h, w = grid_norm.shape

    # 1. Binarização pelo limiar físico
    threshold_norm = dbz_to_norm(min_dbz_threshold)
    binary_mask = (grid_norm >= threshold_norm).astype(np.uint8)

    # 2. Operações Morfológicas
    # Fechamento (Dilation -> Erosion) para consolidar ecos fragmentados da mesma tempestade
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel_size, closing_kernel_size))
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # 3. Análise de Componentes Conectados
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)

    detections = []
    lats = coords_dict.get('lats')
    lons = coords_dict.get('lons')

    # stats: [x_left, y_top, width, height, area]
    # label 0 é sempre o fundo (background)
    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area < min_area_pixels:
            continue

        xmin = int(stats[label_idx, cv2.CC_STAT_LEFT])
        ymin = int(stats[label_idx, cv2.CC_STAT_TOP])
        box_w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        box_h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        xmax = xmin + box_w
        ymax = ymin + box_h

        # Valores em dBZ real na região do núcleo
        cell_mask = (labels == label_idx)
        max_dbz = float(grid_norm[cell_mask].max() * 75.0)

        # Formato YOLO: [class_id, x_center, y_center, width, height] normalizados [0, 1]
        x_center = (xmin + xmax) / (2.0 * w)
        y_center = (ymin + ymax) / (2.0 * h)
        yolo_w = box_w / w
        yolo_h = box_h / h

        # Resgate de coordenadas geográficas do centróide
        cx_pixel = int(centroids[label_idx][0])
        cy_pixel = int(centroids[label_idx][1])
        
        geo_info = {}
        if lats is not None and lons is not None:
            geo_info = {
                "center_lat": float(lats[cy_pixel, cx_pixel]),
                "center_lon": float(lons[cy_pixel, cx_pixel])
            }

        detections.append({
            "class_id": 0,  # 0: "severe_storm_core"
            "bbox_pixel": [xmin, ymin, xmax, ymax],
            "bbox_yolo": [0, round(x_center, 6), round(y_center, 6), round(yolo_w, 6), round(yolo_h, 6)],
            "max_dbz": round(max_dbz, 1),
            "area_pixels": int(area),
            "geo": geo_info
        })

    return detections

def generate_dataset_labels(
    arrays_dir: str,
    coords_npz_path: str,
    output_labels_dir: str
):
    """
    Processa todos os arquivos .npy gerados e salva os rótulos (.txt e .json).
    """
    os.makedirs(output_labels_dir, exist_ok=True)
    coords_dict = {}
    if os.path.exists(coords_npz_path):
        coords_dict = np.load(coords_npz_path)

    npy_files = sorted(glob.glob(os.path.join(arrays_dir, "*.npy")))
    print(f"Gerando labels para {len(npy_files)} frames...")

    summary_labels = {}

    for npy_path in npy_files:
        base_name = os.path.splitext(os.path.basename(npy_path))[0]
        detections = extract_storm_bboxes(npy_path, coords_dict)

        # 1. Salva formato YOLO puro (.txt)
        txt_path = os.path.join(output_labels_dir, f"{base_name}.txt")
        with open(txt_path, 'w') as f:
            for d in detections:
                yolo_str = " ".join(map(str, d["bbox_yolo"]))
                f.write(f"{yolo_str}\n")

        # 2. Guarda metadados enriquecidos para inspeção e tracking
        summary_labels[base_name] = detections

    # Salva arquivo consolidado com metadados geoespaciais e físicos
    json_path = os.path.join(output_labels_dir, "dataset_metadata.json")
    with open(json_path, 'w') as f:
        json.dump(summary_labels, f, indent=2)

    print(f"Labels concluídos e salvos em '{output_labels_dir}'.")

if __name__ == "__main__":

    ARRAYS_DIR = "./data/processed/arrays"
    COORDS_PATH = "./data/processed/grid_coords.npz"
    OUTPUT_LABELS = "./data/processed/labels"

    generate_dataset_labels(ARRAYS_DIR, COORDS_PATH, OUTPUT_LABELS)