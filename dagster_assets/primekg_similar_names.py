import requests
import shutil
from pathlib import Path
from dagster import AssetExecutionContext, asset
from tqdm import tqdm
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import plotly.express as px


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
    return download_file(url, output_path)


@asset(group_name="primekg_similar_names")
def filter_disease_nodes(context: AssetExecutionContext, download_kg: Path) -> Path:
    output_path = Path("primekg_similar_names/filter_disease_nodes/asset_output/diseases.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv(download_kg, infer_schema_length=0)

    x_diseases = df.filter(pl.col("x_type") == "disease").select("x_name").unique()
    y_diseases = df.filter(pl.col("y_type") == "disease").select("y_name").unique()

    diseases = pl.concat([
        x_diseases.rename({"x_name": "disease_name"}),
        y_diseases.rename({"y_name": "disease_name"})
    ]).unique().sort("disease_name")

    diseases.write_csv(output_path)

    return output_path


@asset(group_name="primekg_similar_names")
def disease_embeddings(context: AssetExecutionContext, filter_disease_nodes: Path) -> Path:
    output_path = Path("primekg_similar_names/disease_embeddings/asset_output/embeddings.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv(filter_disease_nodes)
    disease_names = df['disease_name'].to_list()

    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
    tfidf_matrix = vectorizer.fit_transform(disease_names)

    svd = TruncatedSVD(n_components=2, random_state=42)
    embeddings = svd.fit_transform(tfidf_matrix)

    df = df.with_columns([
        pl.Series("x", embeddings[:, 0]),
        pl.Series("y", embeddings[:, 1])
    ])

    df.write_csv(output_path)

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
        hover_data=['disease_name'],
        title='Disease Name Similarity (2D Embedding)'
    )

    fig.update_traces(marker=dict(size=5, opacity=0.6))
    fig.update_layout(
        xaxis_title='Dimension 1',
        yaxis_title='Dimension 2',
        hovermode='closest'
    )

    fig.write_html(output_path)

    return output_path
