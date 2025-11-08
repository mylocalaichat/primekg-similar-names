from pathlib import Path
import pytest
import polars as pl
import os
from dagster import build_asset_context
from dagster_assets.primekg_similar_names import filter_disease_nodes


def test_filter_disease_nodes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

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
