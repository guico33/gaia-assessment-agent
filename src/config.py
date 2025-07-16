"""
Configuration and environment setup for GAIA Agent.
"""

import os
from typing import Dict, List, Optional


class Config:
    """Configuration class for GAIA Agent."""

    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    # Model Configuration
    DEFAULT_MODEL_PROVIDER = "openai"  # or "anthropic"
    DEFAULT_OPENAI_MODEL = "gpt-4.1-mini-2025-04-14"
    DEFAULT_ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
    DEFAULT_TEMPERATURE = 0.1
    DEFAULT_MAX_ITERATIONS = 20

    # Agent Settings
    MAX_FILE_SIZE_MB = 10
    MAX_CONTEXT_LENGTH = 8000
    TIMEOUT_SECONDS = 300  # 5 minutes per question

    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of available LLM providers based on API keys."""
        providers = []
        if cls.OPENAI_API_KEY:
            providers.append("openai")
        if cls.ANTHROPIC_API_KEY:
            providers.append("anthropic")
        return providers

    @classmethod
    def get_default_provider(cls) -> Optional[str]:
        """Get the default provider to use."""
        available = cls.get_available_providers()
        if cls.DEFAULT_MODEL_PROVIDER in available:
            return cls.DEFAULT_MODEL_PROVIDER
        elif available:
            return available[0]
        return None

    @classmethod
    def check_environment(cls) -> Dict[str, bool]:
        """Check which environment variables and dependencies are available."""
        status = {
            "openai_api_key": bool(cls.OPENAI_API_KEY),
            "anthropic_api_key": bool(cls.ANTHROPIC_API_KEY),
            "tavily_api_key": bool(cls.TAVILY_API_KEY),
            "has_llm_provider": bool(cls.get_available_providers()),
        }

        # Check optional dependencies
        try:
            import wikipedia

            status["wikipedia"] = True
        except ImportError:
            status["wikipedia"] = False

        try:
            from youtube_transcript_api._api import YouTubeTranscriptApi  # type: ignore

            status["youtube_transcript"] = True
        except ImportError:
            status["youtube_transcript"] = False

        try:
            from PIL import Image

            status["pillow"] = True
        except ImportError:
            status["pillow"] = False

        try:
            import pandas as pd

            status["pandas"] = True
        except ImportError:
            status["pandas"] = False

        return status

    @classmethod
    def print_status(cls):
        """Print environment status for debugging."""
        print("\n" + "=" * 50)
        print("🔧 GAIA Agent Environment Status")
        print("=" * 50)

        status = cls.check_environment()
        providers = cls.get_available_providers()

        print(f"📊 Available LLM Providers: {providers if providers else 'None'}")
        print(f"🎯 Default Provider: {cls.get_default_provider() or 'None available'}")

        print("\n🔑 API Keys:")
        print(f"  OpenAI: {'✅' if status['openai_api_key'] else '❌'}")
        print(f"  Anthropic: {'✅' if status['anthropic_api_key'] else '❌'}")
        print(f"  Tavily: {'✅' if status['tavily_api_key'] else '❌'}")

        print("\n📦 Dependencies:")
        print(f"  Wikipedia: {'✅' if status['wikipedia'] else '❌'}")
        print(f"  YouTube Transcript: {'✅' if status['youtube_transcript'] else '❌'}")
        print(f"  Pillow (Images): {'✅' if status['pillow'] else '❌'}")
        print(f"  Pandas (Data): {'✅' if status['pandas'] else '❌'}")

        if not status["has_llm_provider"]:
            print("\n⚠️  WARNING: No LLM provider API keys found!")
            print("   Please set OPENAI_API_KEY or ANTHROPIC_API_KEY")

        if not status["tavily_api_key"]:
            print("\n💡 TIP: Set TAVILY_API_KEY for enhanced web search")

        print("=" * 50 + "\n")


def load_environment_file(file_path: str = ".env"):
    """Load environment variables from a .env file if it exists."""
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            for line in f:
                if "=" in line and not line.strip().startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value
        print(f"✅ Loaded environment variables from {file_path}")
    else:
        print(f"ℹ️  No {file_path} file found")


# Load .env file if present
load_environment_file()

# Example .env file content (for reference)
ENV_EXAMPLE = """
# Example .env file for GAIA Agent
# Copy this to .env and fill in your API keys

# LLM Provider (choose one or both)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Web Search (optional but recommended)
TAVILY_API_KEY=your_tavily_api_key_here

# Optional: Override default settings
DEFAULT_MODEL_PROVIDER=openai
DEFAULT_TEMPERATURE=0.1
"""
