from unittest.mock import patch, MagicMock
from src.upload import upload_dataset


@patch("src.upload.Dataset")
def test_upload_dataset_calls_push_to_hub_for_each_config(mock_dataset_cls):
    mock_ds = MagicMock()
    mock_dataset_cls.from_dict.return_value = mock_ds

    books = [{"id": "1", "title": "Test", "text": "Content"}]
    chapters = [{"id": "1", "chapter_index": 0, "text": "Content"}]
    paragraphs = [{"id": "1", "chapter_index": 0, "paragraph_index": 0, "text": "Content"}]

    upload_dataset("test-user/test-repo", books, chapters, paragraphs)

    assert mock_dataset_cls.from_dict.call_count == 3
    assert mock_ds.push_to_hub.call_count == 3

    config_names = [
        call.kwargs["config_name"]
        for call in mock_ds.push_to_hub.call_args_list
    ]
    assert "books" in config_names
    assert "chapters" in config_names
    assert "paragraphs" in config_names
