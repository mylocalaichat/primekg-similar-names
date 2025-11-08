import polars as pl
from unittest.mock import patch
from dagster import build_asset_context
from dagster_assets.primekg_similar_names import filter_disease_nodes, disease_descriptions


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


def test_disease_descriptions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    diseases_file = tmp_path / "diseases.csv"
    df = pl.DataFrame({'disease_name': ['hypertension', 'diabetes']})
    df.write_csv(diseases_file)

    # Mock ollama.chat to avoid actual API calls
    mock_response = {
        'message': {
            'content': 'A chronic condition characterized by high blood pressure.'
        }
    }

    with patch('ollama.chat', return_value=mock_response):
        context = build_asset_context()
        result = disease_descriptions(context, diseases_file)

        assert result.exists()
        df_descriptions = pl.read_csv(result)
        assert 'disease_name' in df_descriptions.columns
        assert 'description' in df_descriptions.columns
        assert len(df_descriptions) == 2
        assert set(df_descriptions['disease_name'].to_list()) == {'diabetes', 'hypertension'}
