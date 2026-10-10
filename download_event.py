import os
from datetime import datetime
import nexradaws

def download_radar_event(radar_id: str, start_dt: datetime, end_dt: datetime, output_dir: str):
    """
    Consulta o bucket aberto da AWS e baixa varreduras NEXRAD Level II.
    """
    os.makedirs(output_dir, exist_ok=True)
    conn = nexradaws.NexradAwsInterface()

    print(f"Buscando scans para {radar_id} entre {start_dt} e {end_dt}...")
    scans = conn.get_avail_scans_in_range(start_dt, end_dt, radar_id)
    print(f"Total de arquivos encontrados: {len(scans)}")

    # Filtra arquivos MDM (metadados intermediários) se existirem
    scans_to_download = [s for s in scans if not s.filename.endswith('_MDM')]

    # Download concorrente nativo da biblioteca
    results = conn.download(scans_to_download, output_dir)
    # print(f"Download concluído: {results.total_number} arquivos salvos em '{output_dir}'.")
    print(f"Download concluído: {results.total} arquivos salvos em '{output_dir}'.")
    return results

if __name__ == "__main__":

    # Supercélula em Oklahoma (KTLX) na tarde de 20/05/2013
    RADAR = "KTLX"
    START = datetime(2013, 5, 20, 19, 0)  # Horário UTC
    END   = datetime(2013, 5, 20, 23, 0)
    OUTPUT_FOLDER = "./data/raw/ktlx_20130520"

    download_radar_event(RADAR, START, END, OUTPUT_FOLDER)