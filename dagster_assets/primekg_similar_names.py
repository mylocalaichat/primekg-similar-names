import requests
import shutil
from pathlib import Path
from dagster import AssetExecutionContext, asset
from tqdm import tqdm
import polars as pl
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cdist
import plotly.express as px
import numpy as np
import torch


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
def disease_embeddings(context: AssetExecutionContext, filter_disease_nodes: Path) -> Path:
    output_path = Path("primekg_similar_names/disease_embeddings/asset_output/embeddings.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv(filter_disease_nodes)
    disease_names = df['disease_name'].to_list()
    n = len(disease_names)

    distance_matrix = np.zeros((n, n), dtype=np.float32)

    batch_size = 100
    total_batches = (n + batch_size - 1) // batch_size

    for batch_start in tqdm(range(0, n, batch_size), total=total_batches, desc="Computing distances"):
        batch_end = min(batch_start + batch_size, n)
        batch_names = disease_names[batch_start:batch_end]
        remaining_names = disease_names[batch_end:]

        batch_dists = cdist(batch_names, remaining_names, scorer=Levenshtein.distance, workers=-1)
        distance_matrix[batch_start:batch_end, batch_end:] = batch_dists
        distance_matrix[batch_end:, batch_start:batch_end] = batch_dists.T

    context.log.info("Running MDS dimensionality reduction with PyTorch")

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    context.log.info(f"Using device: {device}")

    try:
        dist_tensor = torch.from_numpy(distance_matrix).to(device)
        embeddings = torch.randn(n, 2, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([embeddings], lr=1.0)

        for _ in tqdm(range(300), desc="MDS iterations"):
            optimizer.zero_grad()

            embedded_dist = torch.cdist(embeddings, embeddings)
            stress = torch.sum((embedded_dist - dist_tensor) ** 2)

            stress.backward()
            optimizer.step()

        embeddings_np = embeddings.detach().cpu().numpy()
    except NotImplementedError as e:
        if 'cdist_backward' in str(e) and device.type == 'mps':
            context.log.warning("MPS doesn't support cdist backward, falling back to CPU")
            device = torch.device('cpu')
            dist_tensor = torch.from_numpy(distance_matrix).to(device)
            embeddings = torch.randn(n, 2, device=device, requires_grad=True)
            optimizer = torch.optim.Adam([embeddings], lr=1.0)

            for _ in tqdm(range(300), desc="MDS iterations (CPU)"):
                optimizer.zero_grad()

                embedded_dist = torch.cdist(embeddings, embeddings)
                stress = torch.sum((embedded_dist - dist_tensor) ** 2)

                stress.backward()
                optimizer.step()

            embeddings_np = embeddings.detach().cpu().numpy()
        else:
            raise

    distance_matrix[distance_matrix == 0] = np.inf
    min_distances = distance_matrix.min(axis=1)

    df = df.with_columns([
        pl.Series("x", embeddings_np[:, 0]),
        pl.Series("y", embeddings_np[:, 1]),
        pl.Series("min_distance", min_distances)
    ])

    df.write_csv(output_path)
    context.log.info(f"CSV saved to: {output_path.absolute()}")

    return output_path


@asset(group_name="primekg_similar_names")
def disease_similarity_plot(context: AssetExecutionContext, disease_embeddings: Path) -> Path:
    output_path = Path("primekg_similar_names/disease_similarity_plot/asset_output/plot.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv(disease_embeddings)

    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='min_distance',
        hover_data=['disease_name', 'min_distance'],
        title='Disease Name Similarity (Levenshtein Distance)',
        color_continuous_scale='RdYlGn_r'
    )

    fig.update_traces(marker=dict(size=5, opacity=0.7))
    fig.update_layout(
        xaxis_title='MDS Dimension 1',
        yaxis_title='MDS Dimension 2',
        hovermode='closest',
        coloraxis_colorbar=dict(title="Min Distance")
    )

    fig.write_html(output_path)

    return output_path
