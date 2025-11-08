import requests
from pathlib import Path
from dagster import AssetExecutionContext, asset
from tqdm import tqdm


def download_file(url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(output_path, "wb") as f, tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    return output_path


@asset(group_name="download_data")
def download_kg(context: AssetExecutionContext) -> Path:
    url = "https://dataverse.harvard.edu/api/access/datafile/6180620"
    output_path = Path("download_data/asset_output/kg.csv")
    return download_file(url, output_path)
