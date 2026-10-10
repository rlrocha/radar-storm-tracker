import os
import copy
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pyart


def process_nexrad_file(
    file_path: str,
    output_dir_arrays: str,
    output_dir_images: str,
    grid_shape=(1, 256, 256),       # (z, y, x) -> 1 nível vertical, 256x256 horizontal
    grid_limits=((2000.0, 2000.0),  # altitude: 2 km acima do radar
                 (-150000.0, 150000.0), # y: -150 km a +150 km
                 (-150000.0, 150000.0)) # x: -150 km a +150 km
):
    """
    Lê um volume NEXRAD Level II, interpola para grade regular cartesiana 2D,
    salva o tensor normalizado para PyTorch (.npy) e gera a imagem com
    coastlines, estados e coordenadas de latitude/longitude nos eixos (.png).
    """
    base_name = os.path.basename(file_path).replace(".gz", "").replace(".tar", "")
    
    # -------------------------------------------------------------------------
    # 1. Leitura do Volume Polar Bruto
    # -------------------------------------------------------------------------
    try:
        radar = pyart.io.read_nexrad_archive(file_path)
    except Exception as e:
        print(f"[ERRO] Falha ao ler {file_path}: {e}")
        return

    ref_field = 'reflectivity'
    if ref_field not in radar.fields:
        ref_field = 'reflectivity_horizontal'
        if ref_field not in radar.fields:
            print(f"[AVISO] Campo de refletividade ausente em {file_path}. Pulando.")
            return

    # -------------------------------------------------------------------------
    # 2. Interpolação para Grade Cartesiana 2D (CAPPI a 2 km)
    # -------------------------------------------------------------------------
    grid = pyart.map.grid_from_radars(
        (radar,),
        grid_shape=grid_shape,
        grid_limits=grid_limits,
        fields=[ref_field],
        weighting_function='Barnes2'
    )

    # Extrai o plano horizontal 2D: shape (256, 256)
    refl_2d = grid.fields[ref_field]['data'][0]
    refl_array = np.ma.filled(refl_2d, fill_value=0.0).astype(np.float32)

    # -------------------------------------------------------------------------
    # 3. Normalização para o PyTorch (SEM NaNs para garantir estabilidade da Loss)
    # -------------------------------------------------------------------------
    refl_clipped = np.clip(refl_array, 0.0, 75.0)
    refl_norm = refl_clipped / 75.0  # Escala exata [0.0, 1.0]

    os.makedirs(output_dir_arrays, exist_ok=True)
    array_filename = os.path.join(output_dir_arrays, f"{base_name}.npy")
    np.save(array_filename, refl_norm)

    # -------------------------------------------------------------------------
    # 4. Extração e Armazenamento das Coordenadas Geográficas
    # -------------------------------------------------------------------------
    lats_2d = grid.point_latitude['data'][0]
    lons_2d = grid.point_longitude['data'][0]
    radar_lat = grid.radar_latitude['data'][0]
    radar_lon = grid.radar_longitude['data'][0]

    # Salva o arquivo de coordenadas de referência caso ainda não exista
    coords_path = os.path.join(os.path.dirname(output_dir_arrays), "grid_coords.npz")
    if not os.path.exists(coords_path):
        np.savez(
            coords_path,
            lats=lats_2d,
            lons=lons_2d,
            radar_lat=radar_lat,
            radar_lon=radar_lon
        )

    # -------------------------------------------------------------------------
    # 5. Preparação da Visualização (Fundo branco e máscara de ruído)
    # -------------------------------------------------------------------------
    # Valores <= 5 dBZ viram NaN apenas para display visual
    refl_display = np.where(refl_clipped <= 5.0, np.nan, refl_clipped)

    # Configuração do Colormap NWSRef com fundo branco para valores 'bad' (NaNs)
    try:
        cmap = copy.copy(mpl.colormaps['NWSRef'])
    except KeyError:
        try:
            cmap = copy.copy(mpl.colormaps['pyart_NWSRef'])
        except KeyError:
            cmap = copy.copy(mpl.colormaps['turbo'])
    cmap.set_bad(color='white')

    # -------------------------------------------------------------------------
    # 6. Renderização com Cartopy e Rótulos de Lat/Lon
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(7, 7), dpi=120)
    proj = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    # Delimita o enquadramento geográfico exato da varredura
    lon_min, lon_max = float(lons_2d.min()), float(lons_2d.max())
    lat_min, lat_max = float(lats_2d.min()), float(lats_2d.max())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)

    # Plota a malha de refletividade georreferenciada
    ax.pcolormesh(
        lons_2d,
        lats_2d,
        refl_display,
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

    # Marcador da posição da antena do radar
    ax.plot(radar_lon, radar_lat, marker='o', color='red', markersize=3, transform=proj)

    # Rótulos de Latitude (Esquerda) e Longitude (Abaixo)
    gl = ax.gridlines(
        crs=proj,
        draw_labels=True,
        linewidth=0.5,
        color='gray',
        alpha=0.4,
        linestyle='--'
    )
    gl.top_labels = False           # Desativa no topo
    gl.right_labels = False         # Desativa à direita
    gl.bottom_labels = True         # Ativa abaixo
    gl.left_labels = True           # Ativa à esquerda

    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 8.5, 'color': '#222222', 'weight': 'bold'}
    gl.ylabel_style = {'size': 8.5, 'color': '#222222', 'weight': 'bold'}

    # -------------------------------------------------------------------------
    # 7. Salvamento da Imagem
    # -------------------------------------------------------------------------
    os.makedirs(output_dir_images, exist_ok=True)
    img_filename = os.path.join(output_dir_images, f"{base_name}.png")
    
    # bbox_inches='tight' garante que as coordenadas externas sejam incluídas sem cortes
    plt.savefig(img_filename, bbox_inches='tight', dpi=120, facecolor='white')
    plt.close(fig)

    print(f"[OK] Processado: {base_name} | Max dBZ: {refl_array.max():.1f}")

def batch_process(raw_dir: str, processed_dir: str):
    """
    Processa todos os arquivos de radar em lote ordenados cronologicamente.
    """
    dir_arrays = os.path.join(processed_dir, "arrays")
    dir_images = os.path.join(processed_dir, "images")
    os.makedirs(dir_arrays, exist_ok=True)
    os.makedirs(dir_images, exist_ok=True)

    # Busca arquivos brutos (arquivos NEXRAD frequentemente não têm extensão padrão ou terminam em _V06)
    files = sorted(glob.glob(os.path.join(raw_dir, "*")))
    # Filtra subpastas caso existam
    files = [f for f in files if os.path.isfile(f) and not f.endswith('.png') and not f.endswith('.npy')]

    print(f"Iniciando processamento de {len(files)} arquivos...")
    for idx, f in enumerate(files):
        print(f"[{idx+1}/{len(files)}] ", end="")
        process_nexrad_file(f, dir_arrays, dir_images)

if __name__ == "__main__":
    
    RAW_INPUT_DIR = "./data/raw/ktlx_20130520"
    PROCESSED_OUTPUT_DIR = "./data/processed"

    batch_process(RAW_INPUT_DIR, PROCESSED_OUTPUT_DIR)