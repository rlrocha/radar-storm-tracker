import os
import copy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pyart


def inspect_single_frame(
    npy_path: str,
    txt_path: str,
    coords_npz_path: str = None,
    output_png_path: str = None
):
    """
    Carrega o array de refletividade e os rótulos YOLO, plotando a grade georreferenciada
    com Cartopy (coastlines, estados, lat/lon nos eixos), caixas delimitadoras e pico de dBZ.
    """
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Arquivo de grid não encontrado: {npy_path}")

    # 1. Carrega o array normalizado [0.0, 1.0] e converte para dBZ
    grid_norm = np.load(npy_path).astype(np.float32)
    grid_dbz = grid_norm * 75.0
    h, w = grid_norm.shape

    # 2. Carrega as coordenadas geográficas
    lats, lons = None, None
    radar_lat, radar_lon = None, None
    if coords_npz_path and os.path.exists(coords_npz_path):
        coords = np.load(coords_npz_path)
        lats = coords.get('lats')
        lons = coords.get('lons')
        radar_lat = coords.get('radar_lat')
        radar_lon = coords.get('radar_lon')

    if lats is None or lons is None:
        raise ValueError("Coordenadas geográficas ('lats' e 'lons') são necessárias para plotagem cartográfica.")

    # 3. Preparação do colormap com fundo branco
    try:
        cmap = copy.copy(mpl.colormaps['NWSRef'])
    except KeyError:
        try:
            cmap = copy.copy(mpl.colormaps['pyart_NWSRef'])
        except KeyError:
            cmap = copy.copy(mpl.colormaps['turbo'])
    cmap.set_bad(color='white')

    # Mascara ruídos e áreas sem eco significativo
    grid_display = np.where(grid_dbz <= 5.0, np.nan, grid_dbz)

    # 4. Configura figura com projeção cartográfica
    fig = plt.figure(figsize=(8, 8), dpi=150)
    proj = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    lon_min, lon_max = float(lons.min()), float(lons.max())
    lat_min, lat_max = float(lats.min()), float(lats.max())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)

    # Plota a malha de refletividade
    mesh = ax.pcolormesh(
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

    # 5. Lê e desenha os bounding boxes projetados em Lat/Lon
    boxes_count = 0
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue

            class_id, xc, yc, bw, bh = map(float, parts)

            # Índices em pixel da matriz original
            box_w_px = bw * w
            box_h_px = bh * h
            x0_idx = int(np.clip((xc * w) - (box_w_px / 2.0), 0, w - 1))
            y0_idx = int(np.clip((yc * h) - (box_h_px / 2.0), 0, h - 1))
            x1_idx = int(np.clip((xc * w) + (box_w_px / 2.0), 0, w - 1))
            y1_idx = int(np.clip((yc * h) + (box_h_px / 2.0), 0, h - 1))

            # Pico de dBZ do núcleo
            box_subgrid = grid_dbz[y0_idx:y1_idx, x0_idx:x1_idx]
            max_dbz = box_subgrid.max() if box_subgrid.size > 0 else 0.0

            # Conversão dos limites em pixels para coordenadas geográficas reais
            box_lon_min = float(lons[y0_idx, x0_idx])
            box_lat_min = float(lats[y0_idx, x0_idx])
            box_lon_max = float(lons[y1_idx, x1_idx])
            box_lat_max = float(lats[y1_idx, x1_idx])

            geo_w = box_lon_max - box_lon_min
            geo_h = box_lat_max - box_lat_min

            # Desenha o retângulo no sistema de coordenadas PlateCarree
            rect = patches.Rectangle(
                (box_lon_min, box_lat_min),
                geo_w,
                geo_h,
                linewidth=2.0,
                edgecolor='#00FFAA',
                facecolor='none',
                linestyle='-',
                transform=proj
            )
            ax.add_patch(rect)

            # Centróide geográfico
            cx_idx = int(np.clip(xc * w, 0, w - 1))
            cy_idx = int(np.clip(yc * h, 0, h - 1))
            c_lat = float(lats[cy_idx, cx_idx])
            c_lon = float(lons[cy_idx, cx_idx])

            label_text = f"Cell #{boxes_count + 1}\n{max_dbz:.1f} dBZ\n({c_lat:.2f}°, {c_lon:.2f}°)"

            ax.text(
                box_lon_min,
                box_lat_min,
                label_text,
                color='black',
                fontsize=7.5,
                weight='bold',
                verticalalignment='bottom',
                transform=proj,
                bbox=dict(boxstyle='square,pad=0.2', facecolor='#00FFAA', alpha=0.85, edgecolor='none')
            )
            boxes_count += 1

    # 6. Gridlines: Latitudes à esquerda e Longitudes abaixo
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

    # Título e Barra de Cores
    frame_name = os.path.splitext(os.path.basename(npy_path))[0]
    ax.set_title(
        f"Radar Convective Core Detection | Frame: {frame_name}\nDetected Cells (>= 45 dBZ): {boxes_count}",
        fontsize=10.5,
        weight='bold',
        pad=10
    )

    cbar = plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Equivalent Reflectivity ($Z$) [dBZ]", fontsize=8.5)

    # 7. Salva ou exibe
    if output_png_path:
        os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
        plt.savefig(output_png_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"[OK] Visualização salva com sucesso em: {output_png_path}")
    else:
        plt.show()

    plt.close(fig)

if __name__ == "__main__":

    file_name = "KTLX20130520_225821_V06"

    # Caminhos de exemplo conforme a estrutura do projeto
    SAMPLE_NPY = f"./data/processed/arrays/{file_name}.npy"
    SAMPLE_TXT = f"./data/processed/labels/{file_name}.txt"
    COORDS_NPZ = f"./data/processed/grid_coords.npz"
    OUTPUT_IMG = f"./data/processed/inspections/preview_{file_name}.png"

    # Se você ainda não rodou o pipeline real e quer testar a função imediatamente,
    # descomente o bloco abaixo para gerar um dado sintético de teste:
    """
    os.makedirs("./data/processed/arrays", exist_ok=True)
    os.makedirs("./data/processed/labels", exist_ok=True)

    # Cria matriz vazia com 2 núcleos simulados
    dummy_grid = np.zeros((256, 256), dtype=np.float32)
    dummy_grid[100:140, 110:150] = 0.85  # ~63.7 dBZ
    dummy_grid[180:210, 60:85] = 0.70    # ~52.5 dBZ
    np.save(SAMPLE_NPY, dummy_grid)

    # Cria arquivo YOLO correspondente
    # formato: class xc yc w h
    with open(SAMPLE_TXT, "w") as f:
        f.write("0 0.5078 0.4687 0.1562 0.1562\n")
        f.write("0 0.2832 0.7617 0.0976 0.1171\n")
    """

    inspect_single_frame(
        npy_path=SAMPLE_NPY,
        txt_path=SAMPLE_TXT,
        coords_npz_path=COORDS_NPZ,
        output_png_path=OUTPUT_IMG
    )