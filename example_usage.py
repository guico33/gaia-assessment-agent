#!/usr/bin/env python3
"""
Example usage of the GAIA Agent for testing and demonstration.
"""

import os

from src.agent import BasicAgent, create_gaia_agent
from src.config import Config


def main():
    """Demonstrate GAIA agent usage with example questions."""

    print("🤖 GAIA Agent Example Usage")
    print("=" * 50)

    # Check environment
    Config.print_status()

    # Check if we have API keys
    if not Config.get_available_providers():
        print("❌ No API keys found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        return

    print("🚀 Initializing GAIA Agent...")

    # Create agent (will use available provider)
    try:
        agent = BasicAgent()
        print("✅ Agent initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return

    # Example questions from GAIA dataset
    example_questions = [
        # Simple math
        "What is 6 * 7?",
        # Text reversal puzzle
        '.rewsna eht sa "tfel" drow eht fo etisoppo eht etirw ,ecnetnes siht dnatsrednu uoy fI',
        # Research question (requires web search)
        "How many studio albums were published by Mercedes Sosa between 2000 and 2009?",
        # Mathematical reasoning
        "If I have 15 apples and I give away 3/5 of them, how many apples do I have left?",
    ]

    print("\n📝 Testing with sample GAIA questions...")
    print("-" * 50)

    for i, question in enumerate(example_questions, 1):
        print(f"\n🔍 Question {i}: {question}")
        print("⏳ Processing...")

        try:
            answer = agent(question)
            print(f"✅ Answer: {answer}")
        except Exception as e:
            print(f"❌ Error: {e}")

        print("-" * 30)

    print("\n🎉 Example usage complete!")
    print("\n💡 Tips:")
    print("- Set TAVILY_API_KEY for enhanced web search capabilities")
    print("- The agent supports multi-modal inputs (images, audio, files)")
    print("- For GAIA evaluation, run the full Gradio app with: uv run python app.py")


if __name__ == "__main__":
    main()
