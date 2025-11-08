from pathlib import Path
from unittest.mock import Mock, mock_open
import pytest
from dagster_assets.download_data import download_file


def test_download_file(mocker):
    mock_response = Mock()
    mock_response.headers = {'content-length': '100'}
    mock_response.iter_content = Mock(return_value=[b'chunk1', b'chunk2'])

    mocker.patch('requests.get', return_value=mock_response)
    mocker.patch('builtins.open', mock_open())
    mocker.patch('dagster_assets.download_data.tqdm')

    url = "https://example.com/file.csv"
    output_path = Path("test_output/test.csv")

    result = download_file(url, output_path)

    mock_response.raise_for_status.assert_called_once()
    assert result == output_path


def test_download_file_raises_on_error(mocker):
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = Exception("Download failed")

    mocker.patch('requests.get', return_value=mock_response)

    url = "https://example.com/file.csv"
    output_path = Path("test_output/test.csv")

    with pytest.raises(Exception, match="Download failed"):
        download_file(url, output_path)
