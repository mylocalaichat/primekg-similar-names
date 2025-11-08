import requests
import shutil
from pathlib import Path
from dagster import AssetExecutionContext, asset
from tqdm import tqdm
import polars as pl
import ollama


def download_file(url: str, output_path: Path) -> Path:
    if output_path.parent.exists():
        shutil.rmtree(output_path.parent)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(output_path, "wb") as f, tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    return output_path


@asset(group_name="primekg_similar_names")
def download_kg(context: AssetExecutionContext) -> Path:
    url = "https://dataverse.harvard.edu/api/access/datafile/6180620"
    output_path = Path("primekg_similar_names/download_kg/asset_output/kg.csv")
    result = download_file(url, output_path)
    context.log.info(f"CSV saved to: {result.absolute()}")
    return result


@asset(group_name="primekg_similar_names")
def filter_disease_nodes(context: AssetExecutionContext, download_kg: Path) -> Path:
    output_path = Path("primekg_similar_names/filter_disease_nodes/asset_output/diseases.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    context.log.info("Reading CSV into polars")
    df = pl.read_csv(download_kg, infer_schema_length=0)

    context.log.info("Filtering and collecting disease names")
    x_diseases = df.filter(pl.col("x_type") == "disease").select("x_name").unique()
    y_diseases = df.filter(pl.col("y_type") == "disease").select("y_name").unique()

    diseases = pl.concat([
        x_diseases.rename({"x_name": "disease_name"}),
        y_diseases.rename({"y_name": "disease_name"})
    ]).unique().sort("disease_name")

    diseases.write_csv(output_path)
    context.log.info(f"CSV saved to: {output_path.absolute()}")

    return output_path


@asset(group_name="primekg_similar_names")
def disease_descriptions(context: AssetExecutionContext, filter_disease_nodes: Path) -> Path:
    output_path = Path("primekg_similar_names/disease_descriptions/asset_output/descriptions.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv(filter_disease_nodes)
    disease_names = df['disease_name'].to_list()

    context.log.info(f"Getting descriptions for {len(disease_names)} diseases using Ollama (qwen3:4b)")

    descriptions = []
    for disease_name in tqdm(disease_names, desc="Fetching disease descriptions", unit="disease"):
        try:
            prompt = f"""Provide a comprehensive medical description of {disease_name} in approximately 500 words. Include:
1. What the disease is and its medical definition
2. Common symptoms and clinical presentation
3. Known causes and risk factors
4. How it's typically diagnosed
5. Available treatment options and management strategies
6. Prognosis and potential complications

Be factual, medically accurate, and detailed."""
            response = ollama.chat(
                model='qwen3:4b',
                messages=[{'role': 'user', 'content': prompt}]
            )
            description = response['message']['content'].strip()
        except Exception as e:
            context.log.warning(f"Failed to get description for {disease_name}: {e}")
            description = "Description unavailable"

        descriptions.append({
            'disease_name': disease_name,
            'description': description
        })

    result_df = pl.DataFrame(descriptions)
    result_df.write_csv(output_path)
    context.log.info(f"Descriptions CSV saved to: {output_path.absolute()}")

    return output_path
