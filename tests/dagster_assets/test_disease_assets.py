from pathlib import Path
import pytest
import polars as pl
from dagster import build_asset_context
from dagster_assets.primekg_similar_names import filter_disease_nodes, disease_embeddings, disease_similarity_plot


def test_filter_disease_nodes(tmp_path):
    kg_file = tmp_path / "kg.csv"
    kg_file.write_text(
        "relation,display_relation,x_index,x_id,x_type,x_name,x_source,y_index,y_id,y_type,y_name,y_source\n"
        "indication,indication,1,DB001,drug,DrugA,DrugBank,2,5044,disease,hypertension,MONDO\n"
        "disease_protein,disease_protein,3,1001,disease,diabetes,MONDO,4,P001,gene/protein,GeneA,NCBI\n"
    )

    context = build_asset_context()
    result = filter_disease_nodes(context, kg_file)

    assert result.exists()
    df = pl.read_csv(result)
    assert 'disease_name' in df.columns
    assert len(df) == 2
    assert set(df['disease_name'].to_list()) == {'diabetes', 'hypertension'}


def test_disease_embeddings(tmp_path):
    diseases_file = tmp_path / "diseases.csv"
    df = pl.DataFrame({'disease_name': ['hypertension', 'diabetes', 'cancer']})
    df.write_csv(diseases_file)

    context = build_asset_context()
    result = disease_embeddings(context, diseases_file)

    assert result.exists()
    df_embeddings = pl.read_csv(result)
    assert 'x' in df_embeddings.columns
    assert 'y' in df_embeddings.columns
    assert 'disease_name' in df_embeddings.columns
    assert len(df_embeddings) == 3


def test_disease_similarity_plot(tmp_path):
    embeddings_file = tmp_path / "embeddings.csv"
    df = pl.DataFrame({
        'disease_name': ['hypertension', 'diabetes'],
        'x': [0.1, 0.2],
        'y': [0.3, 0.4]
    })
    df.write_csv(embeddings_file)

    context = build_asset_context()
    result = disease_similarity_plot(context, embeddings_file)

    assert result.exists()
    assert result.suffix == '.html'
