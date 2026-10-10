import os
import glob
import copy
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from PIL import Image

from src.model import create_radar_detector
from src.tracker import SortTracker
import pyart


def load_trained_model(checkpoint_path: str, device: torch.device):
    """
    Inicializa a arquitetura adaptada para 1 canal e carrega os pesos treinados.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint não encontrado em: {checkpoint_path}")

    model = create_radar_detector(num_classes=2, pretrained_backbone=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    print(f"[OK] Modelo carregado com sucesso de: {checkpoint_path}")
    return model


def get_radar_colormap():
    """
    Configura o colormap do radar com fundo branco para NaNs.
    """
    try:
        if 'NWSRef' in mpl.colormaps:
            base_cmap = mpl.colormaps['NWSRef']
        elif 'pyart_NWSRef' in mpl.colormaps:
            base_cmap = mpl.colormaps['pyart_NWSRef']
        else:
            base_cmap = mpl.colormaps.get('NWSRef', mpl.colormaps.get('pyart_NWSRef', mpl.colormaps['turbo']))
    except Exception:
        base_cmap = mpl.colormaps['turbo']

    cmap = copy.copy(base_cmap)
    cmap.set_bad(color='white')
    return cmap


def run_inference_pipeline(
    arrays_dir: str,
    checkpoint_path: str,
    output_gif_path: str = "./data/storm_tracking_inferred.gif",
    coords_npz_path: str = "./data/processed/grid_coords.npz",
    conf_threshold: float = 0.45
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executando inferência em: {device}")

    # 1. Carrega modelo treinado
    model = load_trained_model(checkpoint_path, device)

    # 2. Carrega coordenadas geográficas
    if not os.path.exists(coords_npz_path):
        raise FileNotFoundError(f"Arquivo de coordenadas não encontrado: {coords_npz_path}")

    coords = np.load(coords_npz_path)
    lats = coords.get('lats')
    lons = coords.get('lons')
    radar_lat = coords.get('radar_lat')
    radar_lon = coords.get('radar_lon')

    if lats is None or lons is None:
        raise ValueError("Coordenadas 'lats' e 'lons' não encontradas no arquivo .npz.")

    lon_min, lon_max = float(lons.min()), float(lons.max())
    lat_min, lat_max = float(lats.min()), float(lats.max())

    # 3. Inicializa o SORT tracker
    tracker = SortTracker(max_age=2, min_hits=1, iou_threshold=0.10)

    array_files = sorted(glob.glob(os.path.join(arrays_dir, "*.npy")))
    if not array_files:
        raise FileNotFoundError(f"Nenhum arquivo .npy encontrado em: {arrays_dir}")

    print(f"Processando {len(array_files)} frames cronológicos...")

    COLOR_PALETTE = [
        '#00FFCC', '#FF0055', '#FFD700', '#00BFFF', 
        '#FF8C00', '#ADFF2F', '#FF1493', '#E066FF'
    ]
    # Histórico em coordenadas geográficas: {track_id: [(lon, lat), ...]}
    track_histories = {}
    rendered_frames = []

    cmap = get_radar_colormap()
    proj = ccrs.PlateCarree()

    for frame_idx, npy_path in enumerate(array_files):
        # Leitura do grid de radar normalizado [0.0, 1.0]
        grid_norm = np.load(npy_path).astype(np.float32)
        grid_dbz = grid_norm * 75.0
        h, w = grid_norm.shape

        # ==========================================================
        # 1. INFERÊNCIA DO PYTORCH
        # ==========================================================
        tensor_img = torch.from_numpy(grid_norm).unsqueeze(0).to(device)

        with torch.no_grad():
            predictions = model([tensor_img])[0]

        scores = predictions['scores'].cpu().numpy()
        boxes = predictions['boxes'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()

        valid_mask = (scores >= conf_threshold) & (labels == 1)
        pred_boxes = boxes[valid_mask]
        pred_scores = scores[valid_mask]

        if len(pred_boxes) > 0:
            dets = np.column_stack([pred_boxes, pred_scores])
        else:
            dets = np.empty((0, 5), dtype=np.float32)

        # ==========================================================
        # 2. RASTREAMENTO (SORT)
        # ==========================================================
        tracked_objects = tracker.update(dets)

        # Mascara ecos fracos (<= 5 dBZ) para fundo branco
        grid_display = np.where(grid_dbz <= 5.0, np.nan, grid_dbz)

        # ==========================================================
        # 3. RENDERIZAÇÃO GEORREFERENCIADA COM CARTOPY
        # ==========================================================
        fig = plt.figure(figsize=(7, 7), dpi=120, facecolor='white')
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)

        # Malha de refletividade
        ax.pcolormesh(
            lons,
            lats,
            grid_display,
            cmap=cmap,
            vmin=5,
            vmax=75,
            transform=proj,
            shading='auto'
        )

        # Camadas Cartográficas
        ax.coastlines(resolution='50m', color='#333333', linewidth=1.0)
        ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='#666666', linewidth=0.6, linestyle=':')
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), edgecolor='#333333', linewidth=0.8)

        if radar_lat is not None and radar_lon is not None:
            ax.plot(radar_lon, radar_lat, marker='o', color='red', markersize=3.5, transform=proj)

        # ==========================================================
        # 4. DESENHO DAS DETECÇÕES, HISTÓRICO E NOWCASTING
        # ==========================================================
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id, vx, vy = obj
            track_id = int(track_id)
            color = COLOR_PALETTE[track_id % len(COLOR_PALETTE)]

            # Converte limites da caixa (pixels) para coordenadas geográficas
            x0_idx = int(np.clip(x1, 0, w - 1))
            y0_idx = int(np.clip(y1, 0, h - 1))
            x1_idx = int(np.clip(x2, 0, w - 1))
            y1_idx = int(np.clip(y2, 0, h - 1))

            box_lon_min = float(lons[y0_idx, x0_idx])
            box_lat_min = float(lats[y0_idx, x0_idx])
            box_lon_max = float(lons[y1_idx, x1_idx])
            box_lat_max = float(lats[y1_idx, x1_idx])

            geo_w = box_lon_max - box_lon_min
            geo_h = box_lat_max - box_lat_min

            # Centroide em coordenadas geográficas
            cx_px = int(np.clip((x1 + x2) / 2.0, 0, w - 1))
            cy_px = int(np.clip((y1 + y2) / 2.0, 0, h - 1))
            c_lon = float(lons[cy_px, cx_px])
            c_lat = float(lats[cy_px, cx_px])

            if track_id not in track_histories:
                track_histories[track_id] = []
            track_histories[track_id].append((c_lon, c_lat))

            # a) Caixa delimitadora georreferenciada
            rect = patches.Rectangle(
                (box_lon_min, box_lat_min),
                geo_w,
                geo_h,
                linewidth=2.2,
                edgecolor=color,
                facecolor='none',
                transform=proj
            )
            ax.add_patch(rect)

            # b) Rastro histórico georreferenciado
            history = track_histories[track_id]
            if len(history) > 1:
                hlons, hlats = zip(*history[-8:])
                ax.plot(hlons, hlats, color=color, linewidth=2.0, linestyle='--', alpha=0.85, transform=proj)

            # c) Vetor de Nowcasting estimado pelo Kalman
            future_px_x = int(np.clip((x1 + x2) / 2.0 + vx * 3.0, 0, w - 1))
            future_px_y = int(np.clip((y1 + y2) / 2.0 + vy * 3.0, 0, h - 1))
            pred_future_lon = float(lons[future_px_y, future_px_x])
            pred_future_lat = float(lats[future_px_y, future_px_x])

            ax.annotate(
                '',
                xy=(pred_future_lon, pred_future_lat),
                xytext=(c_lon, c_lat),
                xycoords=proj._as_mpl_transform(ax),
                textcoords=proj._as_mpl_transform(ax),
                arrowprops=dict(facecolor=color, edgecolor='white', width=1.5, headwidth=6, shrink=0.05)
            )

            # d) Tag com ID da tempestade e Lat/Lon
            # tag = f"Storm #{track_id}\n({c_lat:.2f}°, {c_lon:.2f}°)"
            tag = f"Storm #{track_id}"
            ax.text(
                box_lon_min,
                box_lat_min,
                tag,
                color='black',
                fontsize=7.5,
                weight='bold',
                verticalalignment='bottom',
                transform=proj,
                bbox=dict(boxstyle='square,pad=0.2', facecolor=color, alpha=0.9, edgecolor='none')
            )

        # ==========================================================
        # 5. GRIDLINES (Latitudes à esquerda e Longitudes abaixo)
        # ==========================================================
        gl = ax.gridlines(
            crs=proj,
            draw_labels=True,
            linewidth=0.5,
            color='gray',
            alpha=0.4,
            linestyle='--'
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = True
        gl.left_labels = True

        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 8.5, 'color': '#222222', 'weight': 'bold'}
        gl.ylabel_style = {'size': 8.5, 'color': '#222222', 'weight': 'bold'}

        base_name = os.path.splitext(os.path.basename(npy_path))[0]
        ax.set_title(
            # f"PyTorch Detection + SORT Tracking\nScan: {base_name} | Tracked Cells: {len(tracked_objects)}",
            f"Scan: {base_name} | Tracked Cells: {len(tracked_objects)}",
            fontsize=9.5,
            weight='bold',
            pad=10
        )

        plt.tight_layout()

        # Converte para PIL Image em memória
        fig.canvas.draw()
        frame_img = Image.frombuffer(
            'RGBA',
            fig.canvas.get_width_height(),
            fig.canvas.buffer_rgba(),
            'raw',
            'RGBA',
            0,
            1
        ).convert('RGB')
        rendered_frames.append(frame_img)
        plt.close(fig)

    # ==========================================================
    # 6. EXPORTAÇÃO DO GIF FINAL
    # ==========================================================
    if rendered_frames:
        os.makedirs(os.path.dirname(output_gif_path), exist_ok=True)
        rendered_frames[0].save(
            output_gif_path,
            save_all=True,
            append_images=rendered_frames[1:],
            duration=350,
            loop=0
        )
        print(f"\n[SUCESSO] Animação com inferência georreferenciada salva em: {output_gif_path}")


if __name__ == "__main__":

    ARRAYS_DIR = "./data/processed/arrays"
    CHECKPOINT = "./checkpoints/radar_detector_best.pth"
    OUTPUT_GIF = "./data/storm_tracking_inferred.gif"
    COORDS_NPZ = "./data/processed/grid_coords.npz"

    run_inference_pipeline(
        arrays_dir=ARRAYS_DIR,
        checkpoint_path=CHECKPOINT,
        output_gif_path=OUTPUT_GIF,
        coords_npz_path=COORDS_NPZ,
        conf_threshold=0.45
    )