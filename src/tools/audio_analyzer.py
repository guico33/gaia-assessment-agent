"""
Audio Analyzer Tool for audio transcription and analysis.
Handles audio file transcription and content analysis for GAIA questions.
"""

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from ..config import Config
from ..llm_provider import get_llm


class AudioAnalyzer:
    """
    Handles audio file transcription and content analysis.
    """

    def __init__(self):
        """Initialize the audio analyzer with LLM and OpenAI for transcription."""
        self.llm = get_llm()
        # Use OpenAI for Whisper transcription
        self.transcription_model = ChatOpenAI(
            model="gpt-4o-audio-preview", temperature=0  # Model with audio capabilities
        )

    def analyze_audio(
        self, task_id: str, filename: str, question_context: str = ""
    ) -> str:
        """
        Analyze an audio file by transcribing and analyzing content.

        Args:
            task_id: The task identifier
            filename: Name of the audio file
            question_context: Context from the original question

        Returns:
            String containing audio analysis results
        """
        try:
            # Construct the audio file path
            audio_path = f"/tmp/{task_id}_{filename}"

            # Check if audio file exists
            if not os.path.exists(audio_path):
                return f"Error: Audio file not found at {audio_path}"

            # Get transcript
            transcript = self._transcribe_audio(audio_path)
            if not transcript:
                return f"Error: Could not transcribe audio file {filename}"

            # Analyze transcript content
            analysis = self._analyze_transcript(transcript, question_context, filename)

            return f"Audio Analysis for {filename}:\n{analysis}"

        except Exception as e:
            return f"Error analyzing audio file {filename}: {str(e)}"

    def transcribe_audio_only(self, task_id: str, filename: str) -> str:
        """
        Transcribe an audio file without additional analysis.

        Args:
            task_id: The task identifier
            filename: Name of the audio file

        Returns:
            String containing the transcription
        """
        try:
            # Construct the audio file path
            audio_path = f"/tmp/{task_id}_{filename}"

            # Check if audio file exists
            if not os.path.exists(audio_path):
                return f"Error: Audio file not found at {audio_path}"

            # Get transcript
            transcript = self._transcribe_audio(audio_path)
            if not transcript:
                return f"Error: Could not transcribe audio file {filename}"

            return f"Transcription for {filename}:\n{transcript}"

        except Exception as e:
            return f"Error transcribing audio file {filename}: {str(e)}"

    def _transcribe_audio(self, audio_path: str) -> Optional[str]:
        """
        Transcribe audio file using OpenAI Whisper or fallback method.

        Args:
            audio_path: Path to the audio file

        Returns:
            Transcribed text or None if failed
        """
        try:
            # Method 1: Try using OpenAI Whisper API (if available)
            transcript = self._transcribe_with_openai_whisper(audio_path)
            if transcript:
                return transcript

            # Method 2: Try using local whisper (if available)
            transcript = self._transcribe_with_local_whisper(audio_path)
            if transcript:
                return transcript

            # Method 3: Basic fallback - return file info
            return self._get_audio_file_info(audio_path)

        except Exception as e:
            return f"Error transcribing audio: {str(e)}"

    def _transcribe_with_openai_whisper(self, audio_path: str) -> Optional[str]:
        """
        Transcribe audio using OpenAI Whisper API.

        Args:
            audio_path: Path to the audio file

        Returns:
            Transcribed text or None if failed
        """
        try:
            # Check if OpenAI API key is available
            import os

            from openai import OpenAI

            api_key = Config.get_openai_api_key()
            if not api_key:
                print("OpenAI API key not found, skipping Whisper transcription")
                return None

            client = OpenAI(api_key=api_key)

            # Open and transcribe the audio file
            with open(audio_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1", file=audio_file, response_format="text"
                )

            return transcript.strip() if transcript else None

        except ImportError:
            print("OpenAI library not available, skipping Whisper transcription")
            return None
        except Exception as e:
            print(f"OpenAI Whisper transcription failed: {e}")
            return None

    def _transcribe_with_local_whisper(self, audio_path: str) -> Optional[str]:
        """
        Transcribe audio using local whisper installation.

        Args:
            audio_path: Path to the audio file

        Returns:
            Transcribed text or None if failed
        """
        try:
            # Check if whisper is available
            result = subprocess.run(
                ["which", "whisper"], capture_output=True, text=True, timeout=10
            )

            if result.returncode != 0:
                return None

            # Run whisper transcription
            result = subprocess.run(
                ["whisper", audio_path, "--model", "base", "--output_format", "txt"],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes timeout
            )

            if result.returncode == 0:
                # Find the generated text file
                txt_file = Path(audio_path).with_suffix(".txt")
                if txt_file.exists():
                    with open(txt_file, "r", encoding="utf-8") as f:
                        transcript = f.read().strip()
                    # Clean up the generated file
                    txt_file.unlink()
                    return transcript

            return None

        except subprocess.TimeoutExpired:
            return "Error: Audio transcription timed out"
        except Exception as e:
            print(f"Local whisper transcription failed: {e}")
            return None

    def _get_audio_file_info(self, audio_path: str) -> str:
        """
        Get basic information about an audio file as fallback.

        Args:
            audio_path: Path to the audio file

        Returns:
            String with file information and helpful suggestions
        """
        try:
            file_size = os.path.getsize(audio_path)
            file_ext = Path(audio_path).suffix
            duration = self._get_audio_duration(audio_path)

            info = f"Audio file information:\n- File: {Path(audio_path).name}\n- Size: {file_size} bytes\n- Format: {file_ext}"

            if duration:
                info += f"\n- Duration: {duration:.1f} seconds"

            info += f"\n\nNote: Automatic transcription not available. This could be due to:\n"
            info += f"- OpenAI API key not configured for Whisper transcription\n"
            info += f"- Local whisper installation not available\n"
            info += f"- Audio format compatibility issues\n\n"
            info += f"To analyze this audio file, you may need to:\n"
            info += f"1. Configure OpenAI API key for Whisper transcription\n"
            info += f"2. Install local whisper (pip install openai-whisper)\n"
            info += f"3. Manually transcribe the audio content\n"
            info += f"4. Use alternative audio analysis tools"

            return info

        except Exception as e:
            return f"Error getting audio file info: {str(e)}"

    def _analyze_transcript(
        self, transcript: str, question_context: str, filename: str
    ) -> str:
        """
        Analyze transcript content using LLM.

        Args:
            transcript: Transcribed text
            question_context: Context from the original question
            filename: Name of the audio file

        Returns:
            String containing analysis results
        """
        try:
            # Create analysis prompt
            analysis_prompt = f"""
            You are analyzing an audio file transcript. Please provide a comprehensive analysis focusing on the specific question context.

            Audio File: {filename}
            Question Context: {question_context}

            Transcript:
            {transcript}

            Please analyze the transcript and provide:
            1. Key topics and themes discussed
            2. Specific information relevant to the question
            3. Important facts, numbers, or details mentioned
            4. Speaker information (if identifiable)
            5. Timeline or sequence of events (if applicable)
            6. Direct answer to the question based on the transcript content

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

    def search_transcript(self, task_id: str, filename: str, search_terms: str) -> str:
        """
        Search for specific terms in an audio transcript.

        Args:
            task_id: The task identifier
            filename: Name of the audio file
            search_terms: Comma-separated list of terms to search for

        Returns:
            String containing search results
        """
        try:
            # Get transcript
            audio_path = f"/tmp/{task_id}_{filename}"
            transcript = self._transcribe_audio(audio_path)

            if not transcript:
                return f"Error: Could not transcribe audio file {filename}"

            # Search for terms
            terms_list = [term.strip().lower() for term in search_terms.split(",")]
            transcript_lower = transcript.lower()

            results = []
            for term in terms_list:
                if term in transcript_lower:
                    # Find context around the term
                    index = transcript_lower.find(term)
                    start = max(0, index - 100)
                    end = min(len(transcript), index + len(term) + 100)
                    context = transcript[start:end]

                    results.append(f"Found '{term}': ...{context}...")
                else:
                    results.append(f"'{term}' not found in transcript")

            return f"Search Results in {filename}:\n" + "\n".join(results)

        except Exception as e:
            return f"Error searching transcript: {str(e)}"

    def get_audio_info(self, task_id: str, filename: str) -> Dict[str, Any]:
        """
        Get basic information about an audio file.

        Args:
            task_id: The task identifier
            filename: Name of the audio file

        Returns:
            Dictionary with audio file information
        """
        try:
            audio_path = f"/tmp/{task_id}_{filename}"

            if not os.path.exists(audio_path):
                return {"error": "Audio file not found"}

            file_size = os.path.getsize(audio_path)
            file_ext = Path(audio_path).suffix

            # Try to get duration using ffprobe (if available)
            duration = self._get_audio_duration(audio_path)

            return {
                "filename": filename,
                "format": file_ext,
                "size_bytes": file_size,
                "duration_seconds": duration,
                "transcription_attempted": False,
            }

        except Exception as e:
            return {"error": f"Error getting audio info: {str(e)}"}

    def _get_audio_duration(self, audio_path: str) -> Optional[float]:
        """
        Get audio duration using ffprobe if available.

        Args:
            audio_path: Path to the audio file

        Returns:
            Duration in seconds or None if not available
        """
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "csv=p=0",
                    audio_path,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                return float(result.stdout.strip())

            return None

        except Exception:
            return None


# Global instance
_audio_analyzer = AudioAnalyzer()


@tool
def analyze_audio(task_id: str, filename: str, question_context: str = "") -> str:
    """
    Analyze an audio file by transcribing and analyzing content.

    Args:
        task_id: The task identifier
        filename: Name of the audio file
        question_context: Context from the original question

    Returns:
        String containing audio analysis results
    """
    return _audio_analyzer.analyze_audio(task_id, filename, question_context)


@tool
def transcribe_audio(task_id: str, filename: str) -> str:
    """
    Transcribe an audio file without additional analysis.

    Args:
        task_id: The task identifier
        filename: Name of the audio file

    Returns:
        String containing the transcription
    """
    return _audio_analyzer.transcribe_audio_only(task_id, filename)


@tool
def search_audio_transcript(task_id: str, filename: str, search_terms: str) -> str:
    """
    Search for specific terms in an audio transcript.

    Args:
        task_id: The task identifier
        filename: Name of the audio file
        search_terms: Comma-separated list of terms to search for

    Returns:
        String containing search results
    """
    return _audio_analyzer.search_transcript(task_id, filename, search_terms)


@tool
def get_audio_file_info(task_id: str, filename: str) -> str:
    """
    Get basic information about an audio file.

    Args:
        task_id: The task identifier
        filename: Name of the audio file

    Returns:
        String containing audio file information
    """
    info = _audio_analyzer.get_audio_info(task_id, filename)
    return f"Audio File Info: {info}"


# Export the tools for use in the main tools module
__all__ = [
    "analyze_audio",
    "transcribe_audio",
    "search_audio_transcript",
    "get_audio_file_info",
    "AudioAnalyzer",
]
