"""
Test cases for Audio Analyzer Tool.
Tests audio transcription, analysis, and error scenarios.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tools.audio_analyzer import (
    AudioAnalyzer,
    analyze_audio,
    get_audio_file_info,
    search_audio_transcript,
    transcribe_audio,
)


class TestAudioAnalyzer:
    """Test cases for AudioAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = AudioAnalyzer()
        self.test_task_id = "test_task_123"
        self.test_filename = "test_audio.mp3"
        self.test_audio_path = f"/tmp/{self.test_task_id}_{self.test_filename}"

    def test_init(self):
        """Test AudioAnalyzer initialization."""
        assert self.analyzer.llm is not None
        assert self.analyzer.transcription_model is not None

    def test_analyze_audio_file_not_found(self):
        """Test analyze_audio when audio file doesn't exist."""
        result = self.analyzer.analyze_audio(
            self.test_task_id, self.test_filename, "What is discussed in this audio?"
        )

        assert "Error: Audio file not found" in result
        assert self.test_audio_path in result

    def test_transcribe_audio_only_file_not_found(self):
        """Test transcribe_audio_only when audio file doesn't exist."""
        result = self.analyzer.transcribe_audio_only(
            self.test_task_id, self.test_filename
        )

        assert "Error: Audio file not found" in result
        assert self.test_audio_path in result

    def test_get_audio_file_info_fallback(self):
        """Test audio file info fallback when transcription fails."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_file.write(b"fake audio data")
            tmp_path = tmp_file.name

        try:
            # Test the fallback method directly
            result = self.analyzer._get_audio_file_info(tmp_path)

            assert "Audio file information:" in result
            assert "Size:" in result
            assert "Format: .mp3" in result
            assert "Automatic transcription not available" in result

        finally:
            os.unlink(tmp_path)

    @patch("subprocess.run")
    def test_transcribe_with_local_whisper_not_available(self, mock_run):
        """Test local whisper transcription when whisper is not available."""
        # Mock 'which whisper' to return non-zero (not found)
        mock_run.return_value = Mock(returncode=1)

        result = self.analyzer._transcribe_with_local_whisper("/fake/path.mp3")

        assert result is None
        mock_run.assert_called_once_with(
            ["which", "whisper"], capture_output=True, text=True, timeout=10
        )

    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    @patch("builtins.open")
    def test_transcribe_with_local_whisper_success(
        self, mock_open, mock_exists, mock_run
    ):
        """Test successful local whisper transcription."""
        # Mock 'which whisper' to return success
        mock_run.side_effect = [
            Mock(returncode=0),  # which whisper
            Mock(returncode=0),  # whisper command
        ]

        # Mock file operations
        mock_exists.return_value = True
        mock_file = Mock()
        mock_file.read.return_value = "This is a test transcript"
        mock_open.return_value.__enter__.return_value = mock_file

        result = self.analyzer._transcribe_with_local_whisper("/fake/path.mp3")

        assert result == "This is a test transcript"
        assert mock_run.call_count == 2
        mock_open.assert_called_once()

    @patch("subprocess.run")
    def test_transcribe_with_local_whisper_timeout(self, mock_run):
        """Test local whisper transcription timeout."""
        # Mock 'which whisper' to return success
        mock_run.side_effect = [
            Mock(returncode=0),  # which whisper
            subprocess.TimeoutExpired(cmd=["whisper"], timeout=300),  # timeout
        ]

        result = self.analyzer._transcribe_with_local_whisper("/fake/path.mp3")

        assert result == "Error: Audio transcription timed out"

    @patch("subprocess.run")
    def test_get_audio_duration_success(self, mock_run):
        """Test successful audio duration extraction."""
        mock_run.return_value = Mock(returncode=0, stdout="123.45\n")

        result = self.analyzer._get_audio_duration("/fake/path.mp3")

        assert result == 123.45
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_get_audio_duration_failure(self, mock_run):
        """Test audio duration extraction failure."""
        mock_run.return_value = Mock(returncode=1)

        result = self.analyzer._get_audio_duration("/fake/path.mp3")

        assert result is None

    @patch.object(AudioAnalyzer, "_transcribe_audio")
    def test_analyze_transcript_success(self, mock_transcribe):
        """Test successful transcript analysis."""
        mock_transcribe.return_value = "This is a test transcript about strawberry pie."

        # Mock LLM response
        mock_response = Mock()
        mock_response.content = (
            "The audio discusses strawberry pie recipes and ingredients."
        )
        self.analyzer.llm.invoke = Mock(return_value=mock_response)

        result = self.analyzer._analyze_transcript(
            "This is a test transcript about strawberry pie.",
            "What ingredients are needed?",
            "test_audio.mp3",
        )

        assert "strawberry pie recipes" in result
        self.analyzer.llm.invoke.assert_called_once()

    @patch.object(AudioAnalyzer, "_transcribe_audio")
    def test_search_transcript_success(self, mock_transcribe):
        """Test successful transcript search."""
        mock_transcribe.return_value = (
            "I need flour, sugar, and strawberries for the pie."
        )

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_file.write(b"fake audio data")
            tmp_path = tmp_file.name

        try:
            # Mock the audio path to point to our temp file
            with patch.object(
                self.analyzer,
                "_transcribe_audio",
                return_value="I need flour, sugar, and strawberries for the pie.",
            ):
                result = self.analyzer.search_transcript(
                    self.test_task_id, self.test_filename, "flour, sugar, strawberries"
                )

                assert "Found 'flour'" in result
                assert "Found 'sugar'" in result
                assert "Found 'strawberries'" in result

        finally:
            os.unlink(tmp_path)

    @patch.object(AudioAnalyzer, "_transcribe_audio")
    def test_search_transcript_no_matches(self, mock_transcribe):
        """Test transcript search with no matches."""
        mock_transcribe.return_value = "This audio is about something else entirely."

        result = self.analyzer.search_transcript(
            self.test_task_id, self.test_filename, "flour, sugar, strawberries"
        )

        assert "'flour' not found" in result
        assert "'sugar' not found" in result
        assert "'strawberries' not found" in result

    @patch("os.path.exists")
    @patch("os.path.getsize")
    def test_get_audio_info_success(self, mock_getsize, mock_exists):
        """Test successful audio info retrieval."""
        mock_exists.return_value = True
        mock_getsize.return_value = 1024000  # 1MB

        with patch.object(self.analyzer, "_get_audio_duration", return_value=123.45):
            result = self.analyzer.get_audio_info(self.test_task_id, self.test_filename)

            assert result["filename"] == self.test_filename
            assert result["format"] == ".mp3"
            assert result["size_bytes"] == 1024000
            assert result["duration_seconds"] == 123.45
            assert result["transcription_attempted"] is False

    def test_get_audio_info_file_not_found(self):
        """Test audio info when file doesn't exist."""
        result = self.analyzer.get_audio_info(self.test_task_id, self.test_filename)

        assert "error" in result
        assert "Audio file not found" in result["error"]

    @patch.object(AudioAnalyzer, "_transcribe_audio")
    def test_analyze_audio_with_transcript_success(self, mock_transcribe):
        """Test full audio analysis with successful transcription."""
        mock_transcribe.return_value = (
            "I need flour, sugar, and strawberries for the pie."
        )

        # Mock LLM response
        mock_response = Mock()
        mock_response.content = (
            "The speaker needs flour, sugar, and strawberries for making a pie."
        )
        self.analyzer.llm.invoke = Mock(return_value=mock_response)

        # Create a temporary file to simulate audio file existence
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_file.write(b"fake audio data")
            tmp_path = tmp_file.name

        try:
            # Mock the expected path
            with patch("os.path.exists", return_value=True):
                result = self.analyzer.analyze_audio(
                    self.test_task_id,
                    self.test_filename,
                    "What ingredients are needed for the pie?",
                )

                assert "Audio Analysis for" in result
                assert "flour, sugar, and strawberries" in result

        finally:
            os.unlink(tmp_path)

    @patch.object(AudioAnalyzer, "_transcribe_audio")
    def test_analyze_audio_transcription_failure(self, mock_transcribe):
        """Test audio analysis when transcription fails."""
        mock_transcribe.return_value = None

        with patch("os.path.exists", return_value=True):
            result = self.analyzer.analyze_audio(
                self.test_task_id, self.test_filename, "What is discussed?"
            )

            assert "Error: Could not transcribe audio file" in result
            assert self.test_filename in result


class TestAudioAnalyzerTools:
    """Test cases for audio analyzer tool functions."""

    @patch("src.tools.audio_analyzer._audio_analyzer")
    def test_analyze_audio_tool(self, mock_analyzer):
        """Test the analyze_audio tool function."""
        mock_analyzer.analyze_audio.return_value = "Audio analysis result"

        result = analyze_audio("test_task", "test.mp3", "What is discussed?") # type: ignore

        assert result == "Audio analysis result"
        mock_analyzer.analyze_audio.assert_called_once_with(
            "test_task", "test.mp3", "What is discussed?"
        )

    @patch("src.tools.audio_analyzer._audio_analyzer")
    def test_transcribe_audio_tool(self, mock_analyzer):
        """Test the transcribe_audio tool function."""
        mock_analyzer.transcribe_audio_only.return_value = "Transcription result"

        result = transcribe_audio("test_task", "test.mp3") # type: ignore

        assert result == "Transcription result"
        mock_analyzer.transcribe_audio_only.assert_called_once_with(
            "test_task", "test.mp3"
        )

    @patch("src.tools.audio_analyzer._audio_analyzer")
    def test_search_audio_transcript_tool(self, mock_analyzer):
        """Test the search_audio_transcript tool function."""
        mock_analyzer.search_transcript.return_value = "Search result"

        result = search_audio_transcript("test_task", "test.mp3", "term1, term2") # type: ignore

        assert result == "Search result"
        mock_analyzer.search_transcript.assert_called_once_with(
            "test_task", "test.mp3", "term1, term2"
        )

    @patch("src.tools.audio_analyzer._audio_analyzer")
    def test_get_audio_file_info_tool(self, mock_analyzer):
        """Test the get_audio_file_info tool function."""
        mock_analyzer.get_audio_info.return_value = {
            "filename": "test.mp3",
            "format": ".mp3",
            "size_bytes": 1024,
        }

        result = get_audio_file_info("test_task", "test.mp3") # type: ignore

        assert "Audio File Info:" in result
        assert "filename" in result
        mock_analyzer.get_audio_info.assert_called_once_with("test_task", "test.mp3")


class TestAudioAnalyzerEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = AudioAnalyzer()

    def test_analyze_audio_exception(self):
        """Test analyze_audio with general exception."""
        with patch.object(
            self.analyzer, "_transcribe_audio", side_effect=Exception("Test error")
        ):
            result = self.analyzer.analyze_audio(
                "test_task", "test.mp3", "test_question"
            )

            assert "Error analyzing audio file" in result
            assert "Test error" in result

    def test_transcribe_audio_only_exception(self):
        """Test transcribe_audio_only with general exception."""
        with patch.object(
            self.analyzer, "_transcribe_audio", side_effect=Exception("Test error")
        ):
            result = self.analyzer.transcribe_audio_only("test_task", "test.mp3")

            assert "Error transcribing audio file" in result
            assert "Test error" in result

    def test_search_transcript_exception(self):
        """Test search_transcript with general exception."""
        with patch.object(
            self.analyzer, "_transcribe_audio", side_effect=Exception("Test error")
        ):
            result = self.analyzer.search_transcript(
                "test_task", "test.mp3", "test_terms"
            )

            assert "Error searching transcript" in result
            assert "Test error" in result

    def test_get_audio_info_exception(self):
        """Test get_audio_info with general exception."""
        with patch("os.path.exists", side_effect=Exception("Test error")):
            result = self.analyzer.get_audio_info("test_task", "test.mp3")

            assert "error" in result
            assert "Test error" in result["error"]

    def test_analyze_transcript_exception(self):
        """Test _analyze_transcript with exception."""
        with patch.object(
            self.analyzer.llm, "invoke", side_effect=Exception("LLM error")
        ):
            result = self.analyzer._analyze_transcript(
                "test transcript", "test question", "test.mp3"
            )

            assert "Error analyzing transcript" in result
            assert "LLM error" in result

    def test_transcribe_audio_all_methods_fail(self):
        """Test when all transcription methods fail."""
        with patch.object(
            self.analyzer, "_transcribe_with_openai_whisper", return_value=None
        ):
            with patch.object(
                self.analyzer, "_transcribe_with_local_whisper", return_value=None
            ):
                with patch.object(
                    self.analyzer,
                    "_get_audio_file_info",
                    return_value="File info fallback",
                ):
                    result = self.analyzer._transcribe_audio("/fake/path.mp3")

                    assert result == "File info fallback"


if __name__ == "__main__":
    pytest.main([__file__])
