"""
YouTube Analyzer Tool for extracting and analyzing YouTube video content.
Handles transcript extraction and video content analysis for GAIA questions.
"""

import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from youtube_transcript_api._api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled

from ..llm_provider import get_llm


class YouTubeAnalyzer:
    """
    Handles YouTube video transcript extraction and content analysis.
    """

    def __init__(self):
        """Initialize the YouTube analyzer with LLM."""
        self.llm = get_llm()

    def analyze_youtube_video(self, url: str, question_context: str = "") -> str:
        """
        Analyze a YouTube video by extracting transcript and analyzing content.

        Args:
            url: YouTube video URL
            question_context: Context from the original question

        Returns:
            String containing video analysis results
        """
        try:
            # Extract video ID from URL
            video_id = self._extract_video_id(url)
            if not video_id:
                return f"Error: Could not extract video ID from URL: {url}"

            # Get transcript
            transcript = self._get_transcript(video_id)
            if not transcript:
                return f"""YouTube Video Analysis for {url}:
Video ID: {video_id}

Could not retrieve transcript for video {video_id}. Transcripts are disabled for this video.
Video URL: {url}
Suggestion: Try searching for information about this video or its content using web search."""

            # Analyze transcript content
            analysis = self._analyze_transcript(transcript, question_context)

            return f"YouTube Video Analysis for {url}:\n{analysis}"

        except Exception as e:
            return f"Error analyzing YouTube video: {str(e)}"

    def extract_transcript_only(self, url: str) -> str:
        """
        Extract just the transcript from a YouTube video.

        Args:
            url: YouTube video URL

        Returns:
            String containing the transcript text
        """
        try:
            # Extract video ID from URL
            video_id = self._extract_video_id(url)
            if not video_id:
                return f"Error: Could not extract video ID from URL: {url}"

            # Get transcript
            transcript = self._get_transcript(video_id)
            if not transcript:
                return f"""Transcript for {url}:
Video ID: {video_id}

Could not retrieve transcript for video {video_id}. Transcripts are disabled for this video.
Video URL: {url}
Suggestion: Try searching for information about this video or its content using web search."""

            # Format transcript
            formatted_transcript = self._format_transcript(transcript)

            return f"Transcript for {url}:\n{formatted_transcript}"

        except Exception as e:
            return f"Error extracting transcript: {str(e)}"

    def search_transcript(self, url: str, search_terms: List[str]) -> str:
        """
        Search for specific terms in a YouTube video transcript.

        Args:
            url: YouTube video URL
            search_terms: List of terms to search for

        Returns:
            String containing search results with context
        """
        try:
            # Extract video ID from URL
            video_id = self._extract_video_id(url)
            if not video_id:
                return f"Error: Could not extract video ID from URL: {url}"

            # Get transcript
            transcript = self._get_transcript(video_id)
            if not transcript:
                return f"""Search Results in {url}:
Video ID: {video_id}

Could not retrieve transcript for video {video_id}. Transcripts are disabled for this video.
Video URL: {url}
Suggestion: Try searching for information about this video or its content using web search."""

            # Search for terms
            results = self._search_in_transcript(transcript, search_terms)

            return f"Search Results in {url}:\n{results}"

        except Exception as e:
            return f"Error searching transcript: {str(e)}"

    def _extract_video_id(self, url: str) -> Optional[str]:
        """
        Extract video ID from YouTube URL.

        Args:
            url: YouTube URL

        Returns:
            Video ID string or None if not found
        """
        # Common YouTube URL patterns
        patterns = [
            r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
            r"youtube\.com/watch\?.*v=([a-zA-Z0-9_-]{11})",
            r"youtu\.be/([a-zA-Z0-9_-]{11})",
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return None

    def _get_transcript(self, video_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get transcript for a YouTube video.

        Args:
            video_id: YouTube video ID

        Returns:
            List of transcript entries or None if not available
        """
        try:
            # Try to get transcript
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return transcript

        except TranscriptsDisabled:
            print(f"Transcripts are disabled for video {video_id}")
            return None
        except NoTranscriptFound:
            print(f"No transcript found for video {video_id}")
            return None
        except Exception as e:
            print(f"Error getting transcript for video {video_id}: {e}")
            return None

    def _format_transcript(self, transcript: List[Dict[str, Any]]) -> str:
        """
        Format transcript entries into readable text.

        Args:
            transcript: List of transcript entries

        Returns:
            Formatted transcript string
        """
        formatted_lines = []

        for entry in transcript:
            start_time = entry.get("start", 0)
            duration = entry.get("duration", 0)
            text = entry.get("text", "")

            # Convert start time to minutes:seconds
            minutes = int(start_time // 60)
            seconds = int(start_time % 60)

            formatted_lines.append(f"[{minutes:02d}:{seconds:02d}] {text}")

        return "\n".join(formatted_lines)

    def _analyze_transcript(
        self, transcript: List[Dict[str, Any]], question_context: str
    ) -> str:
        """
        Analyze transcript content using LLM.

        Args:
            transcript: List of transcript entries
            question_context: Context from the original question

        Returns:
            String containing analysis results
        """
        try:
            # Format transcript for analysis
            formatted_transcript = self._format_transcript(transcript)

            # Create analysis prompt
            analysis_prompt = f"""
            You are analyzing a YouTube video transcript. Please provide a comprehensive analysis focusing on the specific question context.

            Question Context: {question_context}

            Transcript:
            {formatted_transcript}

            Please analyze the transcript and provide:
            1. Key topics and themes discussed
            2. Specific information relevant to the question
            3. Important facts, numbers, or details mentioned
            4. Timeline of key events or points (if applicable)
            5. Direct answer to the question based on the transcript content

            Be thorough and accurate in your analysis.
            """

            # Get LLM analysis
            messages = [SystemMessage(content=analysis_prompt)]
            response = self.llm.invoke(messages)

            # Extract content from response
            if hasattr(response, "content"):
                analysis = str(response.content).strip()
            else:
                analysis = str(response).strip()

            return analysis

        except Exception as e:
            return f"Error analyzing transcript: {str(e)}"

    def _search_in_transcript(
        self, transcript: List[Dict[str, Any]], search_terms: List[str]
    ) -> str:
        """
        Search for terms in transcript with context.

        Args:
            transcript: List of transcript entries
            search_terms: List of terms to search for

        Returns:
            String containing search results with context
        """
        results = []

        for term in search_terms:
            term_lower = term.lower()
            matches = []

            for i, entry in enumerate(transcript):
                text = entry.get("text", "").lower()

                if term_lower in text:
                    start_time = entry.get("start", 0)
                    minutes = int(start_time // 60)
                    seconds = int(start_time % 60)

                    # Get context (surrounding entries)
                    context_start = max(0, i - 2)
                    context_end = min(len(transcript), i + 3)

                    context_text = " ".join(
                        [
                            transcript[j].get("text", "")
                            for j in range(context_start, context_end)
                        ]
                    )

                    matches.append(
                        {
                            "time": f"{minutes:02d}:{seconds:02d}",
                            "text": entry.get("text", ""),
                            "context": context_text,
                        }
                    )

            if matches:
                results.append(f"\nTerm: '{term}' - Found {len(matches)} matches:")
                for match in matches:
                    results.append(f"  [{match['time']}] {match['text']}")
                    results.append(f"  Context: {match['context']}")
            else:
                results.append(f"\nTerm: '{term}' - No matches found")

        return "\n".join(results)

    def get_video_info(self, url: str) -> Dict[str, Any]:
        """
        Get basic information about a YouTube video.

        Args:
            url: YouTube video URL

        Returns:
            Dictionary with video information
        """
        video_id = self._extract_video_id(url)
        if not video_id:
            return {"error": "Could not extract video ID"}

        try:
            # Check if transcript is available
            transcript = self._get_transcript(video_id)
            transcript_available = transcript is not None
            transcript_length = len(transcript) if transcript else 0

            return {
                "video_id": video_id,
                "url": url,
                "transcript_available": transcript_available,
                "transcript_entries": transcript_length,
            }

        except Exception as e:
            return {"error": f"Error getting video info: {str(e)}"}


# Global instance
_youtube_analyzer = YouTubeAnalyzer()


@tool
def analyze_youtube_video(url: str, question_context: str = "") -> str:
    """
    Analyze a YouTube video by extracting transcript and analyzing content.

    Args:
        url: YouTube video URL
        question_context: Context from the original question

    Returns:
        String containing video analysis results
    """
    return _youtube_analyzer.analyze_youtube_video(url, question_context)


@tool
def extract_youtube_transcript(url: str) -> str:
    """
    Extract just the transcript from a YouTube video.

    Args:
        url: YouTube video URL

    Returns:
        String containing the transcript text
    """
    return _youtube_analyzer.extract_transcript_only(url)


@tool
def search_youtube_transcript(url: str, search_terms: str) -> str:
    """
    Search for specific terms in a YouTube video transcript.

    Args:
        url: YouTube video URL
        search_terms: Comma-separated list of terms to search for

    Returns:
        String containing search results with context
    """
    terms_list = [term.strip() for term in search_terms.split(",")]
    return _youtube_analyzer.search_transcript(url, terms_list)


@tool
def get_youtube_video_info(url: str) -> str:
    """
    Get basic information about a YouTube video.

    Args:
        url: YouTube video URL

    Returns:
        String containing video information
    """
    info = _youtube_analyzer.get_video_info(url)
    return f"YouTube Video Info: {info}"


# Export the tools for use in the main tools module
__all__ = [
    "analyze_youtube_video",
    "extract_youtube_transcript",
    "search_youtube_transcript",
    "get_youtube_video_info",
    "YouTubeAnalyzer",
]
