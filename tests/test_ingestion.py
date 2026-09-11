import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from src.file_service import FileIngestionService

def test_ingest_file(tmp_path):
    # Create dummy file
    file_path = tmp_path / "test.txt"
    file_path.write_text("Test content")
    
    # Mock memory
    mock_memory = MagicMock()
    
    # Mock get_memory to return our mock
    with patch("src.file_service.get_memory", return_value=mock_memory):
        # Mock settings for validation
        with patch("src.file_service.get_settings") as mock_settings:
            mock_settings.return_value.documents_path = tmp_path
            mock_settings.return_value.code_path = tmp_path
            
            with patch("src.file_service.get_file_whitelist") as mock_whitelist:
                mock_whitelist.return_value.allowed_extensions = {".txt"}
                mock_whitelist.return_value.blocked_patterns = []
                mock_whitelist.return_value.max_file_size_bytes = 1024 * 1024

                service = FileIngestionService()
                
                count = service.ingest_file(file_path)
                
                assert count == 1
                mock_memory.add_documents.assert_called_once()
                args, _ = mock_memory.add_documents.call_args
                docs = args[0]
                assert len(docs) == 1
                assert docs[0].content == "Test content"

def test_ingest_directory(tmp_path):
    # Create dummy files
    (tmp_path / "f1.txt").write_text("Content 1")
    (tmp_path / "f2.txt").write_text("Content 2")
    
    mock_memory = MagicMock()
    
    with patch("src.file_service.get_memory", return_value=mock_memory):
        with patch("src.file_service.get_settings") as mock_settings:
            mock_settings.return_value.documents_path = tmp_path
            mock_settings.return_value.code_path = tmp_path
            
            with patch("src.file_service.get_file_whitelist") as mock_whitelist:
                mock_whitelist.return_value.allowed_extensions = {".txt"}
                mock_whitelist.return_value.blocked_patterns = []
                mock_whitelist.return_value.max_file_size_bytes = 1024 * 1024

                service = FileIngestionService()
                results = service.ingest_directory(tmp_path)
                
                assert len(results) == 2
                assert mock_memory.add_documents.call_count == 2
