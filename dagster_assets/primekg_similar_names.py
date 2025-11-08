import requests
import shutil
from pathlib import Path
from dagster import AssetExecutionContext, asset
from tqdm import tqdm
import polars as pl
from llama_cpp import Llama
import numpy as np
from sklearn.decomposition import PCA
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
    ]).unique().sort("disease_name").head(100)

    diseases.write_csv(output_path)
    context.log.info(f"CSV saved to: {output_path.absolute()} (limited to 100 records)")

    return output_path


@asset(group_name="primekg_similar_names")
def disease_descriptions(context: AssetExecutionContext, filter_disease_nodes: Path) -> Path:
    output_path = Path("primekg_similar_names/disease_descriptions/asset_output/descriptions.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv(filter_disease_nodes)
    disease_names = df['disease_name'].to_list()

    # Model configuration - Using TinyLlama 1.1B for speed
    model_name = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    model_url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / model_name

    # Download model if it doesn't exist
    if not model_path.exists():
        context.log.info(f"Model not found at {model_path}. Downloading from {model_url}")
        response = requests.get(model_url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        with open(model_path, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading model") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        context.log.info(f"Model downloaded to: {model_path}")
    else:
        context.log.info(f"Using existing model at: {model_path}")

    # Load model
    context.log.info("Loading LLM model...")
    llm = Llama(
        model_path=str(model_path),
        n_ctx=2048,  # TinyLlama context window
        n_threads=8,
        n_gpu_layers=-1,  # Use all GPU layers if available
        n_batch=512,  # Batch size for prompt processing
        verbose=False
    )
    context.log.info("Model loaded successfully")

    context.log.info(f"Getting descriptions for {len(disease_names)} diseases using llama-cpp-python")

    descriptions = []
    for disease_name in tqdm(disease_names, desc="Fetching disease descriptions", unit="disease"):
        try:
            # Using TinyLlama chat format
            prompt = f"""<|system|>
You are a medical expert assistant.</s>
<|user|>
Write a brief medical description of the disease: {disease_name}

Include:
- What the disease is
- Main symptoms
- Causes
- Treatment options

Keep it concise and factual.</s>
<|assistant|>
"""

            output = llm(
                prompt,
                max_tokens=256,  # Reduced for faster, more focused generation
                temperature=0.3,  # Lower temperature for more factual responses
                stop=["</s>", "<|user|>", "<|system|>"],
                echo=False
            )
            description = output['choices'][0]['text'].strip()
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
    context.log.info(f"File URI: file://{output_path.absolute()}")

    return output_path


@asset(group_name="primekg_similar_names")
def disease_embeddings(context: AssetExecutionContext, disease_descriptions: Path) -> Path:
    output_path = Path("primekg_similar_names/disease_embeddings/asset_output/embeddings.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv(disease_descriptions)

    # Model configuration - Using nomic-embed-text for embeddings
    model_name = "nomic-embed-text-v1.5.Q4_K_M.gguf"
    model_url = "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf"
    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / model_name

    # Download model if it doesn't exist
    if not model_path.exists():
        context.log.info(f"Model not found at {model_path}. Downloading from {model_url}")
        response = requests.get(model_url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        with open(model_path, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading embedding model") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        context.log.info(f"Embedding model downloaded to: {model_path}")
    else:
        context.log.info(f"Using existing embedding model at: {model_path}")

    # Load embedding model
    context.log.info("Loading embedding model...")
    llm = Llama(
        model_path=str(model_path),
        n_ctx=8192,  # Nomic supports 8k context
        n_threads=8,
        n_gpu_layers=-1,
        embedding=True,  # Enable embedding mode
        verbose=False,
        n_batch=512
    )
    context.log.info("Embedding model loaded successfully")

    context.log.info(f"Generating embeddings for {len(df)} disease descriptions")

    embeddings_data = []
    for row in tqdm(df.iter_rows(named=True), total=len(df), desc="Generating embeddings", unit="disease"):
        try:
            disease_name = row['disease_name']
            description = row['description']

            # Generate embedding
            embedding = llm.embed(description)

            # Store as comma-separated string for CSV
            embedding_str = ','.join(map(str, embedding))

            embeddings_data.append({
                'disease_name': disease_name,
                'description': description,
                'embedding': embedding_str
            })
        except Exception as e:
            context.log.warning(f"Failed to generate embedding for {row['disease_name']}: {e}")
            embeddings_data.append({
                'disease_name': row['disease_name'],
                'description': row['description'],
                'embedding': ''
            })

    result_df = pl.DataFrame(embeddings_data)
    result_df.write_csv(output_path)
    context.log.info(f"Embeddings CSV saved to: {output_path.absolute()}")
    context.log.info(f"File URI: file://{output_path.absolute()}")

    return output_path


@asset(group_name="primekg_similar_names")
def disease_embeddings_viz(context: AssetExecutionContext, disease_embeddings: Path) -> Path:
    output_path = Path("primekg_similar_names/disease_embeddings_viz/asset_output/embeddings_viz.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    context.log.info("Reading embeddings CSV")
    df = pl.read_csv(disease_embeddings)

    # Filter out rows with empty embeddings
    df = df.filter(pl.col("embedding") != "")

    context.log.info(f"Processing {len(df)} disease embeddings for visualization")

    # Parse embeddings from comma-separated strings to numpy arrays
    embeddings_list = []
    disease_names = []
    descriptions = []

    for row in df.iter_rows(named=True):
        try:
            embedding_str = row['embedding']
            embedding = np.array([float(x) for x in embedding_str.split(',')])
            embeddings_list.append(embedding)
            disease_names.append(row['disease_name'])
            descriptions.append(row['description'][:100] + '...' if len(row['description']) > 100 else row['description'])
        except Exception as e:
            context.log.warning(f"Failed to parse embedding for {row['disease_name']}: {e}")

    if len(embeddings_list) < 2:
        context.log.error("Not enough valid embeddings for visualization")
        return output_path

    embeddings_matrix = np.array(embeddings_list)
    context.log.info(f"Embeddings matrix shape: {embeddings_matrix.shape}")

    # Reduce to 3D using PCA
    context.log.info("Reducing embeddings to 3D using PCA")
    pca = PCA(n_components=3, random_state=42)
    embeddings_3d = pca.fit_transform(embeddings_matrix)

    context.log.info(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.2%}")

    # Create DataFrame for plotting
    viz_df = pl.DataFrame({
        'disease_name': disease_names,
        'description': descriptions,
        'x': embeddings_3d[:, 0],
        'y': embeddings_3d[:, 1],
        'z': embeddings_3d[:, 2]
    })

    # Add node IDs (index)
    viz_df = viz_df.with_columns(
        pl.Series("node_id", range(len(viz_df)))
    )

    # Create 3D scatter plot
    context.log.info("Creating 3D visualization")
    fig = px.scatter_3d(
        viz_df,
        x='x',
        y='y',
        z='z',
        hover_data=['node_id', 'disease_name'],
        title='Disease Embeddings Visualization (3D PCA)',
        labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3'},
        color='disease_name',
        height=800
    )

    fig.update_traces(
        marker=dict(size=5, opacity=0.8),
        hovertemplate='<b>Node ID:</b> %{customdata[0]}<br><b>Name:</b> %{customdata[1]}<extra></extra>'
    )

    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        scene=dict(
            xaxis_title='Principal Component 1',
            yaxis_title='Principal Component 2',
            zaxis_title='Principal Component 3'
        )
    )

    fig.write_html(output_path)
    file_uri = f"file://{output_path.absolute()}"
    context.log.info(f"Visualization saved to: {output_path.absolute()}")
    context.log.info(f"Open in browser: {file_uri}")

    return output_path
