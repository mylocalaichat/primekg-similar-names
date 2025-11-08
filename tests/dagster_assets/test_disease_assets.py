import polars as pl
from unittest.mock import patch, MagicMock
from dagster import build_asset_context
from dagster_assets.primekg_similar_names import filter_disease_nodes, disease_descriptions, disease_embeddings, disease_embeddings_viz, disease_similarity_pairs, disease_clusters


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

    # Create a fake model file
    models_dir = tmp_path / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    fake_model = models_dir / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    fake_model.touch()

    # Mock Llama class to avoid loading actual model
    mock_llm_instance = MagicMock()
    mock_llm_instance.return_value = {
        'choices': [{
            'text': 'A chronic condition characterized by high blood pressure.'
        }]
    }

    with patch('dagster_assets.primekg_similar_names.Llama', return_value=mock_llm_instance):
        context = build_asset_context()
        result = disease_descriptions(context, diseases_file)

        assert result.exists()
        df_descriptions = pl.read_csv(result)
        assert 'disease_name' in df_descriptions.columns
        assert 'description' in df_descriptions.columns
        assert len(df_descriptions) == 2
        assert set(df_descriptions['disease_name'].to_list()) == {'diabetes', 'hypertension'}


def test_disease_embeddings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # Create input descriptions file
    descriptions_file = tmp_path / "descriptions.csv"
    df = pl.DataFrame({
        'disease_name': ['hypertension', 'diabetes'],
        'description': ['High blood pressure condition', 'Blood sugar disorder']
    })
    df.write_csv(descriptions_file)

    # Create a fake embedding model file
    models_dir = tmp_path / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    fake_model = models_dir / "nomic-embed-text-v1.5.Q4_K_M.gguf"
    fake_model.touch()

    # Mock Llama class for embeddings
    mock_llm_instance = MagicMock()
    mock_llm_instance.embed.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]  # Mock embedding vector

    with patch('dagster_assets.primekg_similar_names.Llama', return_value=mock_llm_instance):
        context = build_asset_context()
        result = disease_embeddings(context, descriptions_file)

        assert result.exists()
        df_embeddings = pl.read_csv(result)
        assert 'disease_name' in df_embeddings.columns
        assert 'description' in df_embeddings.columns
        assert 'embedding' in df_embeddings.columns
        assert len(df_embeddings) == 2
        assert set(df_embeddings['disease_name'].to_list()) == {'diabetes', 'hypertension'}


def test_disease_embeddings_viz(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # Create input embeddings file with mock embeddings (768 dimensions)
    embeddings_file = tmp_path / "embeddings.csv"
    # Create a simple 5-dimensional embedding for testing
    embedding1 = ','.join(['0.1'] * 5)
    embedding2 = ','.join(['0.2'] * 5)
    embedding3 = ','.join(['0.3'] * 5)

    df = pl.DataFrame({
        'disease_name': ['hypertension', 'diabetes', 'cancer'],
        'description': ['High blood pressure', 'Blood sugar disorder', 'Abnormal cell growth'],
        'embedding': [embedding1, embedding2, embedding3]
    })
    df.write_csv(embeddings_file)

    context = build_asset_context()
    result = disease_embeddings_viz(context, embeddings_file)

    assert result.exists()
    assert result.suffix == '.html'


def test_disease_similarity_pairs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # Create input embeddings file with mock embeddings
    embeddings_file = tmp_path / "embeddings.csv"
    # Create embeddings that are similar
    embedding1 = ','.join(['0.1'] * 5)
    embedding2 = ','.join(['0.11'] * 5)  # Very similar to embedding1
    embedding3 = ','.join(['0.9'] * 5)   # Different from others

    df = pl.DataFrame({
        'disease_name': ['hypertension', 'high blood pressure', 'diabetes'],
        'description': ['desc1', 'desc2', 'desc3'],
        'embedding': [embedding1, embedding2, embedding3]
    })
    df.write_csv(embeddings_file)

    context = build_asset_context()
    result = disease_similarity_pairs(context, embeddings_file)

    assert result.exists()
    df_pairs = pl.read_csv(result)
    assert 'disease_1' in df_pairs.columns
    assert 'disease_2' in df_pairs.columns
    assert 'similarity' in df_pairs.columns
    assert 'node_id_1' in df_pairs.columns
    assert 'node_id_2' in df_pairs.columns
    assert len(df_pairs) == 3  # 3 diseases = 3 pairs (3 choose 2)


def test_disease_clusters(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # Create input embeddings file with similar and dissimilar diseases
    embeddings_file = tmp_path / "embeddings.csv"
    # Create embeddings: first two are similar, third is different
    embedding1 = ','.join(['0.1'] * 10)
    embedding2 = ','.join(['0.11'] * 10)  # Very similar to embedding1
    embedding3 = ','.join(['0.9'] * 10)   # Different from others

    df = pl.DataFrame({
        'disease_name': ['hypertension', 'high blood pressure', 'diabetes'],
        'description': ['desc1', 'desc2', 'desc3'],
        'embedding': [embedding1, embedding2, embedding3]
    })
    df.write_csv(embeddings_file)

    context = build_asset_context()
    result = disease_clusters(context, embeddings_file)

    assert result.exists()
    df_clusters = pl.read_csv(result)
    assert 'node_id' in df_clusters.columns
    assert 'disease_name' in df_clusters.columns
    assert 'cluster_id' in df_clusters.columns
    assert 'is_noise' in df_clusters.columns
    assert len(df_clusters) == 3

    # Check that canonical mapping file was created
    mapping_file = result.parent / "canonical_mapping.csv"
    assert mapping_file.exists()
    df_mapping = pl.read_csv(mapping_file)
    assert 'original_disease' in df_mapping.columns
    assert 'canonical_disease' in df_mapping.columns
    assert 'is_canonical' in df_mapping.columns
