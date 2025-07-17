"""
Test cases for YouTube Analyzer Tool.
Tests video ID extraction, transcript handling, and error scenarios.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled

from ..youtube_analyzer import (
    YouTubeAnalyzer,
    analyze_youtube_video,
    extract_youtube_transcript,
    get_youtube_video_info,
    search_youtube_transcript,
)


class TestYouTubeAnalyzer:
    """Test cases for YouTubeAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = YouTubeAnalyzer()

    def test_init(self):
        """Test YouTubeAnalyzer initialization."""
        assert self.analyzer.llm is not None

    def test_extract_video_id_standard_url(self):
        """Test video ID extraction from standard YouTube URLs."""
        test_cases = [
            ("https://www.youtube.com/watch?v=L1vXCYZAYYM", "L1vXCYZAYYM"),
            ("https://youtube.com/watch?v=L1vXCYZAYYM", "L1vXCYZAYYM"),
            ("http://www.youtube.com/watch?v=L1vXCYZAYYM", "L1vXCYZAYYM"),
            ("https://www.youtube.com/watch?v=L1vXCYZAYYM&t=10s", "L1vXCYZAYYM"),
            ("https://www.youtube.com/watch?list=PLxxx&v=L1vXCYZAYYM", "L1vXCYZAYYM"),
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ]

        for url, expected_id in test_cases:
            result = self.analyzer._extract_video_id(url)
            assert result == expected_id, f"Failed for URL: {url}"

    def test_extract_video_id_short_url(self):
        """Test video ID extraction from short YouTube URLs."""
        test_cases = [
            ("https://youtu.be/L1vXCYZAYYM", "L1vXCYZAYYM"),
            ("http://youtu.be/L1vXCYZAYYM", "L1vXCYZAYYM"),
            ("youtu.be/L1vXCYZAYYM", "L1vXCYZAYYM"),
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ]

        for url, expected_id in test_cases:
            result = self.analyzer._extract_video_id(url)
            assert result == expected_id, f"Failed for URL: {url}"

    def test_extract_video_id_embed_url(self):
        """Test video ID extraction from embed URLs."""
        test_cases = [
            ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ]

        for url, expected_id in test_cases:
            result = self.analyzer._extract_video_id(url)
            assert result == expected_id, f"Failed for URL: {url}"

    def test_extract_video_id_invalid_url(self):
        """Test video ID extraction from invalid URLs."""
        test_cases = [
            "https://www.google.com",
            "https://www.youtube.com/channel/UCxxxxxx",
            "https://www.youtube.com/user/username",
            "not a url at all",
            "",
            "youtube.com/watch?v=invalid",  # Too short
            "youtube.com/watch?v=toolongvideoID123",  # Too long
        ]

        for url in test_cases:
            result = self.analyzer._extract_video_id(url)
            assert result is None, f"Should return None for invalid URL: {url}"

    @patch("src.tools.youtube_analyzer.YouTubeTranscriptApi.get_transcript")
    def test_get_transcript_success(self, mock_get_transcript):
        """Test successful transcript retrieval."""
        mock_transcript = [
            {"start": 0.0, "duration": 3.0, "text": "Hello world"},
            {"start": 3.0, "duration": 2.0, "text": "This is a test"},
        ]
        mock_get_transcript.return_value = mock_transcript

        result = self.analyzer._get_transcript("dQw4w9WgXcQ")

        assert result == mock_transcript
        mock_get_transcript.assert_called_once_with("dQw4w9WgXcQ")

    @patch("src.tools.youtube_analyzer.YouTubeTranscriptApi.get_transcript")
    def test_get_transcript_disabled(self, mock_get_transcript):
        """Test transcript retrieval when transcripts are disabled."""
        mock_get_transcript.side_effect = TranscriptsDisabled("video_id")

        result = self.analyzer._get_transcript("dQw4w9WgXcQ")

        assert result is None
        mock_get_transcript.assert_called_once_with("dQw4w9WgXcQ")

    @patch("src.tools.youtube_analyzer.YouTubeTranscriptApi.get_transcript")
    def test_get_transcript_not_found(self, mock_get_transcript):
        """Test transcript retrieval when transcript is not found."""
        mock_get_transcript.side_effect = NoTranscriptFound("video_id", [], "")

        result = self.analyzer._get_transcript("dQw4w9WgXcQ")

        assert result is None
        mock_get_transcript.assert_called_once_with("dQw4w9WgXcQ")

    @patch("src.tools.youtube_analyzer.YouTubeTranscriptApi.get_transcript")
    def test_get_transcript_general_error(self, mock_get_transcript):
        """Test transcript retrieval with general error."""
        mock_get_transcript.side_effect = Exception("Network error")

        result = self.analyzer._get_transcript("dQw4w9WgXcQ")

        assert result is None
        mock_get_transcript.assert_called_once_with("dQw4w9WgXcQ")

    def test_format_transcript(self):
        """Test transcript formatting."""
        transcript = [
            {"start": 0.0, "duration": 3.0, "text": "Hello world"},
            {"start": 65.5, "duration": 2.0, "text": "This is a test"},
            {"start": 3720.0, "duration": 1.0, "text": "Final message"},
        ]

        result = self.analyzer._format_transcript(transcript)

        expected = "[00:00] Hello world\n[01:05] This is a test\n[62:00] Final message"
        assert result == expected

    def test_format_transcript_empty(self):
        """Test transcript formatting with empty transcript."""
        result = self.analyzer._format_transcript([])
        assert result == ""

    @patch.object(YouTubeAnalyzer, "_get_transcript")
    @patch.object(YouTubeAnalyzer, "_extract_video_id")
    def test_analyze_youtube_video_success(self, mock_extract_id, mock_get_transcript):
        """Test successful YouTube video analysis."""
        mock_extract_id.return_value = "dQw4w9WgXcQ"
        mock_get_transcript.return_value = [
            {"start": 0.0, "duration": 3.0, "text": "Hello world"}
        ]

        # Mock the LLM response
        mock_response = Mock()
        mock_response.content = "This video says hello to the world"
        self.analyzer.llm.invoke = Mock(return_value=mock_response)

        result = self.analyzer.analyze_youtube_video(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "What does the video say?"
        )

        assert "YouTube Video Analysis" in result
        assert "This video says hello to the world" in result
        mock_extract_id.assert_called_once()
        mock_get_transcript.assert_called_once_with("dQw4w9WgXcQ")

    @patch.object(YouTubeAnalyzer, "_extract_video_id")
    def test_analyze_youtube_video_no_video_id(self, mock_extract_id):
        """Test YouTube video analysis with no video ID."""
        mock_extract_id.return_value = None

        result = self.analyzer.analyze_youtube_video(
            "https://www.google.com", "What does the video say?"
        )

        assert "Could not extract video ID" in result
        mock_extract_id.assert_called_once()

    @patch.object(YouTubeAnalyzer, "_get_transcript")
    @patch.object(YouTubeAnalyzer, "_extract_video_id")
    def test_analyze_youtube_video_no_transcript(
        self, mock_extract_id, mock_get_transcript
    ):
        """Test YouTube video analysis with no transcript available."""
        mock_extract_id.return_value = "dQw4w9WgXcQ"
        mock_get_transcript.return_value = None

        result = self.analyzer.analyze_youtube_video(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "What does the video say?"
        )

        assert "Could not retrieve transcript" in result
        mock_extract_id.assert_called_once()
        mock_get_transcript.assert_called_once_with("dQw4w9WgXcQ")

    def test_search_in_transcript(self):
        """Test searching for terms in transcript."""
        transcript = [
            {"start": 0.0, "text": "Hello world, this is a test"},
            {"start": 3.0, "text": "We are testing the search functionality"},
            {"start": 6.0, "text": "This should find the word test multiple times"},
            {"start": 9.0, "text": "Final message without keywords"},
        ]

        result = self.analyzer._search_in_transcript(transcript, ["test", "hello"])

        assert "test" in result
        assert "hello" in result
        assert "Found 2 matches" in result  # "test" appears twice
        assert "Found 1 matches" in result  # "hello" appears once
        assert "[00:00]" in result  # timestamp format

    def test_search_in_transcript_no_matches(self):
        """Test searching for terms with no matches."""
        transcript = [
            {"start": 0.0, "text": "Hello world"},
            {"start": 3.0, "text": "This is a test"},
        ]

        result = self.analyzer._search_in_transcript(transcript, ["nonexistent"])

        assert "nonexistent" in result
        assert "No matches found" in result

    @patch.object(YouTubeAnalyzer, "_get_transcript")
    @patch.object(YouTubeAnalyzer, "_extract_video_id")
    def test_get_video_info_success(self, mock_extract_id, mock_get_transcript):
        """Test getting video info successfully."""
        mock_extract_id.return_value = "dQw4w9WgXcQ"
        mock_get_transcript.return_value = [{"start": 0.0, "text": "test"}]

        result = self.analyzer.get_video_info(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )

        assert result["video_id"] == "dQw4w9WgXcQ"
        assert result["transcript_available"] is True
        assert result["transcript_entries"] == 1

    @patch.object(YouTubeAnalyzer, "_extract_video_id")
    def test_get_video_info_no_video_id(self, mock_extract_id):
        """Test getting video info with no video ID."""
        mock_extract_id.return_value = None

        result = self.analyzer.get_video_info("invalid_url")

        assert "error" in result
        assert "Could not extract video ID" in result["error"]


class TestYouTubeAnalyzerTools:
    """Test cases for YouTube analyzer tool functions."""

    @patch("src.tools.youtube_analyzer._youtube_analyzer")
    def test_analyze_youtube_video_tool(self, mock_analyzer):
        """Test the analyze_youtube_video tool function."""
        mock_analyzer.analyze_youtube_video.return_value = "Analysis result"

        result = analyze_youtube_video("https://www.youtube.com/watch?v=test", "Test question")  # type: ignore

        assert result == "Analysis result"
        mock_analyzer.analyze_youtube_video.assert_called_once_with(
            "https://www.youtube.com/watch?v=test", "Test question"
        )

    @patch("src.tools.youtube_analyzer._youtube_analyzer")
    def test_extract_youtube_transcript_tool(self, mock_analyzer):
        """Test the extract_youtube_transcript tool function."""
        mock_analyzer.extract_transcript_only.return_value = "Transcript result"

        result = extract_youtube_transcript("https://www.youtube.com/watch?v=test")

        assert result == "Transcript result"
        mock_analyzer.extract_transcript_only.assert_called_once_with(
            "https://www.youtube.com/watch?v=test"
        )

    @patch("src.tools.youtube_analyzer._youtube_analyzer")
    def test_search_youtube_transcript_tool(self, mock_analyzer):
        """Test the search_youtube_transcript tool function."""
        mock_analyzer.search_transcript.return_value = "Search result"

        result = search_youtube_transcript(
            "https://www.youtube.com/watch?v=test", "term1, term2"  # type: ignore
        )

        assert result == "Search result"
        mock_analyzer.search_transcript.assert_called_once_with(
            "https://www.youtube.com/watch?v=test", ["term1", "term2"]
        )

    @patch("src.tools.youtube_analyzer._youtube_analyzer")
    def test_get_youtube_video_info_tool(self, mock_analyzer):
        """Test the get_youtube_video_info tool function."""
        mock_analyzer.get_video_info.return_value = {
            "video_id": "test",
            "transcript_available": True,
        }

        result = get_youtube_video_info("https://www.youtube.com/watch?v=test")

        assert "YouTube Video Info:" in result
        assert "video_id" in result
        mock_analyzer.get_video_info.assert_called_once_with(
            "https://www.youtube.com/watch?v=test"
        )


class TestYouTubeAnalyzerEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = YouTubeAnalyzer()

    def test_analyze_youtube_video_exception(self):
        """Test analyze_youtube_video with exception."""
        # Mock _extract_video_id to raise an exception
        with patch.object(
            self.analyzer, "_extract_video_id", side_effect=Exception("Test error")
        ):
            result = self.analyzer.analyze_youtube_video("test_url", "test_question")

            assert "Error analyzing YouTube video" in result
            assert "Test error" in result

    def test_extract_transcript_only_exception(self):
        """Test extract_transcript_only with exception."""
        with patch.object(
            self.analyzer, "_extract_video_id", side_effect=Exception("Test error")
        ):
            result = self.analyzer.extract_transcript_only("test_url")

            assert "Error extracting transcript" in result
            assert "Test error" in result

    def test_search_transcript_exception(self):
        """Test search_transcript with exception."""
        with patch.object(
            self.analyzer, "_extract_video_id", side_effect=Exception("Test error")
        ):
            result = self.analyzer.search_transcript("test_url", ["term1"])

            assert "Error searching transcript" in result
            assert "Test error" in result

    def test_get_video_info_exception(self):
        """Test get_video_info with exception."""
        with patch.object(
            self.analyzer, "_extract_video_id", side_effect=Exception("Test error")
        ):
            result = self.analyzer.get_video_info("test_url")

            assert "error" in result
            assert "Test error" in result["error"]

    @patch.object(YouTubeAnalyzer, "_analyze_transcript")
    @patch.object(YouTubeAnalyzer, "_get_transcript")
    @patch.object(YouTubeAnalyzer, "_extract_video_id")
    def test_analyze_transcript_exception(
        self, mock_extract_id, mock_get_transcript, mock_analyze
    ):
        """Test _analyze_transcript with exception."""
        mock_extract_id.return_value = "test_id"
        mock_get_transcript.return_value = [{"start": 0.0, "text": "test"}]
        mock_analyze.side_effect = Exception("Analysis error")

        result = self.analyzer.analyze_youtube_video("test_url", "test_question")

        assert "Error analyzing YouTube video" in result
        assert "Analysis error" in result


if __name__ == "__main__":
    pytest.main([__file__])
