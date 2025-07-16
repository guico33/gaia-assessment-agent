# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Hugging Face Gradio application template for agent evaluation assignments. The project consists of a simple Gradio web interface that allows users to test and submit agent implementations against an evaluation API.

## Core Architecture

- **Main Application**: `app.py` - Gradio web interface with Hugging Face OAuth integration
- **Agent Implementation**: `BasicAgent` class in `app.py:13-20` - The core agent logic that students modify
- **Evaluation Flow**: 
  1. Fetch questions from evaluation API (`DEFAULT_API_URL`)
  2. Run agent on each question
  3. Submit answers and receive score
- **Deployment**: Hugging Face Spaces with automatic environment variable injection

## Package Management Migration to uv

The project currently uses pip with `requirements.txt` but should be migrated to uv:

### Migration Commands
```bash
# Install uv if not available
curl -LsSf https://astral.sh/uv/install.sh | sh

# Convert requirements.txt to pyproject.toml
uv init --python 3.11
uv add gradio requests pandas

# Remove old files after verification
rm requirements.txt
```

### Development Commands (after uv migration)
```bash
# Install dependencies
uv sync

# Run the application locally
uv run python app.py

# Add new dependencies
uv add package_name

# Update dependencies
uv lock --upgrade
```

## Current Development Commands (pip-based)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## Key Implementation Areas

### Agent Development
- Modify the `BasicAgent` class in `app.py:13-20`
- The `__call__` method receives a question string and must return an answer string
- Agent instantiation happens in `run_and_submit_all` function at `app.py:43`

### Environment Variables
- `SPACE_ID`: Used for generating code repository links
- `SPACE_HOST`: Runtime URL for Hugging Face Spaces
- These are automatically set in Hugging Face Spaces deployment

### API Integration
- Questions endpoint: `{DEFAULT_API_URL}/questions`
- Submission endpoint: `{DEFAULT_API_URL}/submit`
- Requires valid Hugging Face OAuth profile for submissions

## Testing and Deployment

The application is designed to run on Hugging Face Spaces with automatic deployment. Local testing can be done by running `python app.py` directly.

## Important Notes

- Keep the agent submission format intact (task_id + submitted_answer)
- Maintain Hugging Face OAuth integration for user authentication
- The evaluation API expects specific JSON structure for submissions
- Space cloning workflow is expected for student submissions

## MCPs

- You have access to langchain and langraph documentation via MCPs.
- Use these resources to understand how to implement advanced agent features if needed.

## Agent Implementation Strategy

- We're using LangGraph to build the agent that is going to be evaluated against GAIA questions

## GAIA Agent Architecture

The project now includes a sophisticated LangGraph-based agent with the following components:

### Core Files
- **`agent.py`** - Main LangGraph agent implementation with ReAct architecture
- **`tools.py`** - Collection of specialized tools for GAIA questions
- **`config.py`** - Configuration management and environment setup
- **`app.py`** - Updated Gradio interface with agent integration

### Agent Capabilities

**Multi-Modal Tools:**
- **Web Search** - Tavily integration for research questions
- **Wikipedia Search** - Factual information lookup
- **Image Analysis** - AI-powered visual analysis using GPT-4 vision (chess, charts, etc.)
- **YouTube Analysis** - AI-powered video content analysis with transcript extraction
- **Spreadsheet Analysis** - Excel/CSV data processing and calculation
- **Code Execution** - Safe Python code execution for computational tasks
- **File Processing** - PDF, text, and document reading
- **Text Processing** - Reverse text and decode puzzles
- **Mathematical Calculations** - Safe expression evaluation

**Architecture Features:**
- **LangGraph ReAct Agent** - Tool selection and orchestration
- **Memory Management** - Conversation state persistence
- **Error Handling** - Robust error recovery and user feedback
- **Multi-Provider Support** - OpenAI and Anthropic compatibility
- **Progress Tracking** - Detailed logging for debugging

### Environment Setup

**Required API Keys:**
- `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` (at least one required)
- `TAVILY_API_KEY` (optional, for enhanced web search)

**Development Commands (uv-based):**
```bash
# Install dependencies
uv sync

# Set up environment (create .env file)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here  
TAVILY_API_KEY=your_tavily_key_here

# Run the application
uv run python app.py

# Check environment status
uv run python -c "from config import Config; Config.print_status()"

# Run tests
uv run pytest test_code_executor.py -v

# Run manual test suite
uv run python test_code_executor.py
```

### Testing the Agent

The agent can handle diverse GAIA question types:
- Mathematical reasoning and calculations
- Text puzzles and reversals
- Web research and fact-finding
- File analysis (images, audio, documents)
- Multi-step problem solving
- Code execution and computational tasks

The BasicAgent class maintains backward compatibility while providing advanced GAIA capabilities through LangGraph integration.

## MCPs

- You have access to langchain and langraph documentation via MCPs.
- Use these resources to understand how to implement advanced agent features if needed.

## Agent Implementation Strategy

- We're going to use LangGraph to build the agent that is going to be evaluated against GAIA questions