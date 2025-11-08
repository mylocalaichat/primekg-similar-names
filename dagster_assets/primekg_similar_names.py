import requests
import shutil
from pathlib import Path
from dagster import AssetExecutionContext, asset
from tqdm import tqdm
import polars as pl
from llama_cpp import Llama


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

    return output_path
