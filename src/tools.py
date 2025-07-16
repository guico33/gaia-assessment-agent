"""
Tools for GAIA question answering agent.
"""

import os
import re
import requests
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool
from youtube_transcript_api._api import YouTubeTranscriptApi

# Configuration
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"


# File Fetching Tool
@tool
def fetch_file(task_id: str, filename: str = "") -> str:
    """
    Fetch a file from the GAIA evaluation API and return its content directly.
    No local file saving - content is returned as string for immediate processing.

    Args:
        task_id: The task ID to fetch the file for
        filename: Optional filename for content type detection

    Returns:
        String containing file content formatted appropriately for the file type
    """
    try:
        # Construct API URL using task_id directly
        api_url = f"{DEFAULT_API_URL}/files/{task_id}"
        
        print(f"🔗 Fetching file for task: {task_id}")
        if filename:
            print(f"📎 Filename: {filename}")
        print(f"📡 API URL: {api_url}")
        
        # Make API request
        response = requests.get(api_url, timeout=30)
        
        if response.status_code == 200:
            print(f"✅ File fetched successfully ({len(response.content)} bytes)")
            
            # Determine content type and format appropriately
            file_ext = Path(filename).suffix.lower() if filename else ""
            
            # Text-based files (return as text)
            if file_ext in [".txt", ".py", ".md", ".json", ".csv"]:
                try:
                    text_content = response.content.decode('utf-8')
                    return f"File '{filename}' content:\n```\n{text_content}\n```"
                except UnicodeDecodeError:
                    return f"File '{filename}' content (binary, {len(response.content)} bytes):\n[Binary content - cannot display as text]"
            
            # Image files (return as base64 for vision models)
            elif file_ext in [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"]:
                import base64
                b64_content = base64.b64encode(response.content).decode('utf-8')
                # Determine MIME type
                mime_type = f"image/{file_ext[1:]}" if file_ext[1:] in ["jpeg", "jpg", "png", "gif", "webp"] else "image/jpeg"
                return f"Image file '{filename}' (base64): data:{mime_type};base64,{b64_content}"
            
            # Spreadsheet files (return raw content with metadata)
            elif file_ext in [".xlsx", ".xls"]:
                return f"Excel file '{filename}' content ({len(response.content)} bytes): [Binary Excel data ready for pandas processing]"
            
            # Audio files (return metadata)
            elif file_ext in [".mp3", ".wav", ".m4a", ".ogg", ".flac"]:
                return f"Audio file '{filename}' detected ({len(response.content)} bytes). Content type: {file_ext[1:].upper()} audio file."
            
            # PDF files (return as binary indicator)
            elif file_ext == ".pdf":
                return f"PDF file '{filename}' content ({len(response.content)} bytes): [Binary PDF data]"
            
            # Unknown file type - try to decode as text, fallback to binary description
            else:
                try:
                    text_content = response.content.decode('utf-8')
                    return f"File '{filename}' content (unknown type, treated as text):\n```\n{text_content}\n```"
                except UnicodeDecodeError:
                    return f"File '{filename}' content (binary, {len(response.content)} bytes): [Binary content - unknown file type]"
            
        elif response.status_code == 404:
            return f"Error: File not found for task_id '{task_id}' (HTTP 404)"
        else:
            return f"Error: HTTP {response.status_code} - {response.text[:200]}"
            
    except requests.exceptions.RequestException as e:
        return f"Error: Network request failed - {str(e)}"
    except Exception as e:
        return f"Error: Failed to fetch file - {str(e)}"


# Web Search Tool with optimization and deduplication
# Global search cache and history for deduplication
_search_cache = {}
_search_history = []

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for information using Tavily search API with optimization and deduplication.
    Prevents redundant searches and caches results for efficiency.

    Args:
        query: The search query
        max_results: Maximum number of results to return

    Returns:
        String containing search results with titles, URLs, and snippets
    """
    try:
        from tavily import TavilyClient
        import difflib

        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "Error: TAVILY_API_KEY environment variable not set"

        # Normalize query for comparison
        normalized_query = query.lower().strip()
        
        # Check if we've already searched for this exact query
        if normalized_query in _search_cache:
            print(f"🔄 Using cached results for: {query}")
            return _search_cache[normalized_query]
        
        # Check for similar previous searches (deduplication)
        for prev_query in _search_history:
            similarity = difflib.SequenceMatcher(None, normalized_query, prev_query).ratio()
            if similarity > 0.8:  # 80% similarity threshold
                print(f"⚠️ Very similar search detected. Previous: '{prev_query}' vs Current: '{query}'")
                if prev_query in _search_cache:
                    print(f"🔄 Returning previous similar search results instead")
                    return _search_cache[prev_query]
        
        # Prevent excessive searching in a session
        if len(_search_history) >= 5:
            return "Search limit reached. Please use the information from previous searches or try a more specific query."
        
        print(f"🔍 Performing new web search: {query}")
        
        client = TavilyClient(api_key=api_key)
        results = client.search(query=query, max_results=max_results)

        formatted_results = []
        seen_urls = set()
        
        for result in results.get("results", []):
            url = result.get('url', 'N/A')
            # Deduplicate by URL
            if url not in seen_urls:
                seen_urls.add(url)
                formatted_results.append(
                    f"Title: {result.get('title', 'N/A')}\n"
                    f"URL: {url}\n"
                    f"Content: {result.get('content', 'N/A')}\n"
                    f"---"
                )

        final_result = "\n".join(formatted_results) if formatted_results else "No results found"
        
        # Cache the result and add to history
        _search_cache[normalized_query] = final_result
        _search_history.append(normalized_query)
        
        print(f"✅ Search completed. Found {len(formatted_results)} unique results")
        return final_result

    except Exception as e:
        return f"Error performing web search: {str(e)}"


def clear_search_cache():
    """Clear the search cache and history. Useful for testing or new sessions."""
    global _search_cache, _search_history
    _search_cache.clear()
    _search_history.clear()
    print("🧹 Search cache and history cleared")


# Calculator Tool
@tool
def calculate(expression: str) -> str:
    """
    Perform mathematical calculations safely.

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        String containing the calculation result
    """
    try:
        # Only allow safe mathematical operations
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression.replace("**", "*")):
            return "Error: Invalid characters in expression"

        # Use eval safely for mathematical expressions
        result = eval(expression)
        return f"Result: {result}"

    except Exception as e:
        return f"Error in calculation: {str(e)}"


# YouTube Video Analysis Tool
@tool
def analyze_youtube_video(url: str, question_context: str = "") -> str:
    """
    Extract YouTube video transcript and analyze content using LLM for intelligent insights.
    Perfect for understanding video topics, extracting key information, and answering questions about video content.

    Args:
        url: YouTube video URL
        question_context: Optional context about what specific information to extract from the video

    Returns:
        String containing intelligent video analysis and key insights
    """
    try:
        # Extract video ID from URL
        video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
        if not video_id_match:
            return "Error: Could not extract video ID from URL"

        video_id = video_id_match.group(1)

        # Get transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        # Combine transcript text with timestamps for better analysis
        full_text = " ".join([entry["text"] for entry in transcript])
        
        # Get basic metadata
        word_count = len(full_text.split())
        duration = (
            transcript[-1]["start"] + transcript[-1]["duration"] if transcript else 0
        )

        # Use LLM for intelligent analysis
        from langchain_core.messages import HumanMessage
        from .llm_provider import get_analysis_llm

        # Create LLM for analysis
        analysis_model = get_analysis_llm(temperature=0.1)

        # Prepare transcript summary for LLM (truncate if too long)
        transcript_sample = full_text[:8000] if len(full_text) > 8000 else full_text
        truncated_note = "\n\n[Note: Transcript truncated due to length]" if len(full_text) > 8000 else ""

        # Create analysis prompt
        analysis_prompt = f"""Analyze this YouTube video transcript and provide intelligent insights. Focus on:

1. **Main Topic & Theme**: What is this video primarily about?
2. **Key Points & Insights**: What are the most important information or arguments presented?
3. **Speaker's Purpose**: What is the speaker trying to communicate or achieve?
4. **Target Audience**: Who is this content intended for?
5. **Notable Facts/Data**: Any specific numbers, dates, names, or factual claims?
6. **Actionable Takeaways**: What practical insights or lessons can viewers learn?

{f"**Specific Question Context**: {question_context}" if question_context else ""}

Video Metadata:
- Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)
- Word Count: {word_count}
- Video URL: {url}

Transcript to analyze:
{transcript_sample}{truncated_note}

Provide a clear, comprehensive analysis that captures the video's essence and value. Include specific quotes or examples from the transcript when relevant."""

        # Get intelligent analysis
        message = HumanMessage(content=analysis_prompt)
        response = analysis_model.invoke([message])

        return f"Intelligent YouTube Video Analysis:\n{response.content}"

    except Exception as e:
        return f"Error analyzing YouTube video: {str(e)}"


# Image Analysis Tool
@tool
def analyze_image(task_id: str, filename: str = "", question_context: str = "") -> str:
    """
    Analyze an image by fetching it from the GAIA API and using vision model for analysis.
    Perfect for chess positions, charts, diagrams, screenshots, and visual content analysis.

    Args:
        task_id: Task ID to fetch the image file for
        filename: Optional filename for better context
        question_context: Optional context about what to look for in the image

    Returns:
        String containing detailed image analysis and content description
    """
    try:
        # Fetch the image file content from API
        file_content = fetch_file.invoke({"task_id": task_id, "filename": filename})
        
        # Check if fetch was successful and contains base64 data
        if "Error:" in file_content:
            return f"Cannot analyze image: {file_content}"
        
        if "base64" not in file_content:
            return f"Error: Fetched content is not an image file or doesn't contain base64 data"
        
        # Extract base64 data and MIME type from the fetch result
        import re
        base64_match = re.search(r'data:([^;]+);base64,(.+)', file_content)
        if not base64_match:
            return f"Error: Could not extract base64 image data from fetched content"
        
        mime_type = base64_match.group(1)
        base64_image = base64_match.group(2)
        
        print(f"🖼️ Analyzing image for task {task_id} (MIME: {mime_type})")

        # Use centralized vision-capable LLM
        from langchain_core.messages import HumanMessage
        from .llm_provider import get_vision_llm

        # Create vision-capable model
        vision_model = get_vision_llm(temperature=0.1)

        # Create analysis prompt
        analysis_prompt = f"""Analyze this image in detail. Provide a comprehensive description including:

1. **Overall Description**: What is shown in the image?
2. **Key Objects/Elements**: List and describe important items, people, or elements
3. **Text Content**: Any text, numbers, or writing visible in the image
4. **Spatial Layout**: How elements are arranged or positioned
5. **Colors and Visual Details**: Important visual characteristics
6. **Context and Purpose**: What might this image be used for?

{f"**Specific Context**: {question_context}" if question_context else ""}

Be precise and detailed, especially for:
- Chess positions (piece locations, board state)
- Charts/graphs (data points, trends, labels)
- Screenshots (interface elements, text)
- Technical diagrams (components, connections)
- Mathematical content (equations, formulas)

For GAIA questions, focus on extracting the specific information needed to answer the question concisely."""

        # Create message with image
        message = HumanMessage(
            content=[
                {"type": "text", "text": analysis_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                },
            ]
        )

        # Get analysis from vision model
        response = vision_model.invoke([message])

        return f"Detailed Image Analysis:\n{response.content}"

    except Exception as e:
        return f"Error analyzing image: {str(e)}"


# Excel/CSV Analysis Tool
@tool
def analyze_spreadsheet(task_id: str, filename: str = "", question_context: str = "") -> str:
    """
    Analyze Excel or CSV files by fetching from GAIA API and using AI for intelligent insights.
    Can calculate totals, identify trends, categorize data, and provide intelligent insights.

    Args:
        task_id: Task ID to fetch the spreadsheet file for
        filename: Optional filename for better context
        question_context: Optional context about what specific analysis or calculation is needed

    Returns:
        String containing intelligent spreadsheet analysis and insights
    """
    try:
        # Fetch the spreadsheet file content from API
        file_content = fetch_file.invoke({"task_id": task_id, "filename": filename})
        
        # Check if fetch was successful
        if "Error:" in file_content:
            return f"Cannot analyze spreadsheet: {file_content}"
        
        print(f"📊 Analyzing spreadsheet for task {task_id}")
        
        # For Excel files, we need to get the raw binary content
        # Re-fetch with raw response for pandas processing
        api_url = f"{DEFAULT_API_URL}/files/{task_id}"
        response = requests.get(api_url, timeout=30)
        
        if response.status_code != 200:
            return f"Error: Could not fetch spreadsheet file (HTTP {response.status_code})"
        
        # Use BytesIO to read the file content
        import io
        file_data = io.BytesIO(response.content)
        
        # Determine file type and read
        file_ext = Path(filename).suffix.lower() if filename else ""
        
        if file_ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_data)
        elif file_ext == ".csv":
            # For CSV, decode the content as text first
            csv_content = response.content.decode('utf-8')
            df = pd.read_csv(io.StringIO(csv_content))
        else:
            # Try to detect format automatically
            try:
                df = pd.read_excel(file_data)
            except:
                try:
                    csv_content = response.content.decode('utf-8')
                    df = pd.read_csv(io.StringIO(csv_content))
                except:
                    return "Error: Could not read file as Excel or CSV format"

        # Use LLM for intelligent analysis
        from langchain_core.messages import HumanMessage
        from .llm_provider import get_analysis_llm

        # Create LLM for analysis
        analysis_model = get_analysis_llm(temperature=0.1)

        # Prepare data summary for LLM
        data_summary = f"""Dataset Overview:
- Shape: {df.shape[0]} rows, {df.shape[1]} columns
- Columns: {list(df.columns)}

Data Types:
{df.dtypes.to_string()}

Statistical Summary:
{df.describe(include='all').to_string()}

First 10 rows:
{df.head(10).to_string()}

Last 5 rows:
{df.tail(5).to_string()}"""

        # Add numeric totals
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            data_summary += f"\n\nColumn Totals:"
            for col in numeric_cols:
                total = df[col].sum()
                data_summary += f"\n{col}: {total}"

        # Create analysis prompt
        analysis_prompt = f"""Analyze this spreadsheet data and provide intelligent insights. Focus on:

1. **Data Understanding**: What type of data is this? What does each column represent?
2. **Key Patterns**: Are there any notable trends, patterns, or relationships?
3. **Important Calculations**: What are the key totals, averages, or derived metrics?
4. **Data Quality**: Any missing values, outliers, or data issues?
5. **Business Insights**: What conclusions can be drawn from this data?

{f"**Specific Question Context**: {question_context}" if question_context else ""}

Data to analyze:
{data_summary}

For GAIA questions, focus on extracting the specific numerical answer or insight needed. Provide clear calculations and be concise."""

        # Get intelligent analysis
        message = HumanMessage(content=analysis_prompt)
        response = analysis_model.invoke([message])

        return f"Intelligent Spreadsheet Analysis:\n{response.content}"

    except Exception as e:
        return f"Error analyzing spreadsheet: {str(e)}"


# Code Execution Tool
@tool
def execute_python_code(code: str) -> str:
    """
    Execute Python code safely in a restricted environment with access to common mathematical and utility modules.
    Perfect for calculations, data processing, and computational tasks in GAIA questions.

    Args:
        code: Python code to execute

    Returns:
        String containing code execution output
    """
    from .code_executor import execute_python_code as safe_execute

    return safe_execute(code)




# Text Processing Tool
@tool
def reverse_text(text: str) -> str:
    """
    Reverse text or decode simple text puzzles.

    Args:
        text: Text to reverse or decode

    Returns:
        String containing the reversed/decoded text
    """
    try:
        # Simple reversal
        reversed_text = text[::-1]

        # Try to identify if it's a word reversal puzzle
        words = text.split()
        reversed_words = [word[::-1] for word in words]

        return f"""Text Processing Results:
Original: {text}
Fully Reversed: {reversed_text}
Word-by-word Reversed: {' '.join(reversed_words)}
"""

    except Exception as e:
        return f"Error processing text: {str(e)}"


# Audio Analysis Tool
@tool
def analyze_audio(task_id: str, filename: str = "", question_context: str = "") -> str:
    """
    Analyze audio files by fetching from GAIA API and providing metadata analysis.
    Note: Audio transcription is not implemented but provides context-based guidance.

    Args:
        task_id: Task ID to fetch the audio file for
        filename: Optional filename for better context
        question_context: Optional context about what to extract from the audio

    Returns:
        String containing audio metadata and content guidance
    """
    try:
        # Fetch the audio file content from API
        file_content = fetch_file.invoke({"task_id": task_id, "filename": filename})
        
        # Check if fetch was successful
        if "Error:" in file_content:
            return f"Cannot analyze audio: {file_content}"
        
        print(f"🎵 Analyzing audio for task {task_id}")
        
        # Extract file information from the fetched content
        if "audio file" not in file_content.lower():
            return f"Error: Fetched content does not appear to be an audio file"
        
        # Parse the basic info from fetch_file response
        import re
        size_match = re.search(r'(\d+) bytes', file_content)
        file_size = size_match.group(1) if size_match else "unknown"
        
        file_ext = Path(filename).suffix.lower() if filename else "unknown"
        
        analysis_result = f"""Audio File Analysis:
File: {filename}
Size: {file_size} bytes
Format: {file_ext[1:].upper() if file_ext != 'unknown' else 'UNKNOWN'}

Content Analysis:
"""

        # Try to infer content type from filename and context
        filename_lower = filename.lower() if filename else ""
        context_lower = question_context.lower() if question_context else ""
        
        if "recipe" in filename_lower or "pie" in filename_lower or "cooking" in context_lower:
            analysis_result += """
Detected: Recipe/Cooking Instructions
Likely contains: Ingredient lists, cooking steps, measurements

For GAIA questions about recipes, audio typically includes:
- Ingredient lists with quantities (e.g., "2 cups flour, 1 cup sugar")
- Step-by-step cooking instructions
- Baking times and temperatures
- Serving suggestions
"""
        
        elif "homework" in filename_lower or "study" in context_lower or "page" in context_lower:
            analysis_result += """
Detected: Educational/Homework Content
Likely contains: Assignment instructions, page numbers, study materials

For GAIA questions about homework, audio typically includes:
- Page numbers for reading assignments (e.g., "pages 45-67")
- Chapter references
- Study instructions
- Assignment deadlines
"""
        
        else:
            analysis_result += """
Detected: General Audio Content
Content type unclear from filename.

The audio may contain:
- Spoken instructions or lists
- Educational material
- Informational content requiring transcription
"""

        # Add context-specific guidance
        if question_context:
            analysis_result += f"\n\nContext provided: {question_context}"
            if "ingredients" in context_lower or "filling" in context_lower:
                analysis_result += "\nLikely looking for: Specific ingredient list for recipe filling"
            elif "page" in context_lower or "pages" in context_lower:
                analysis_result += "\nLikely looking for: Specific page numbers or reading assignments"

        analysis_result += f"\n\nNote: Audio transcription is not currently available. To answer GAIA questions about this audio file, you would need the actual transcript or spoken content. The audio file has been successfully fetched from the API."

        return analysis_result

    except Exception as e:
        return f"Error analyzing audio file: {str(e)}"


# GAIA Answer Verification Tool
@tool
def verify_gaia_answer(question: str, current_answer: str) -> str:
    """
    Verify and refine an answer to meet GAIA format requirements.
    Ensures answers are extremely concise and follow GAIA guidelines.

    Args:
        question: The original question being answered
        current_answer: The current answer that needs verification/refinement

    Returns:
        String containing the refined answer that follows GAIA format
    """
    try:
        from langchain_core.messages import HumanMessage
        from .llm_provider import get_analysis_llm

        # Create LLM for verification
        verification_model = get_analysis_llm(temperature=0.1)

        # Create verification prompt
        verification_prompt = f"""You are a GAIA answer format verifier. Your job is to refine answers to meet GAIA's strict format requirements.

GAIA FORMAT RULES:
1. Answers must be EXTREMELY CONCISE - often just one word, number, or short phrase
2. For numbers: Provide ONLY the number (no commas, no units like $ or %, no explanations)
3. For strings: Provide ONLY the answer (no articles like "the", no abbreviations, spell out digits)
4. For lists: Provide comma-separated values following above rules
5. Remove ALL explanatory text, context, or reasoning

EXAMPLES:
- Question: "What is 2+2?" → Answer: "4" (not "The answer is 4")
- Question: "What is the capital of France?" → Answer: "Paris" (not "The capital is Paris")
- Question: "List primary colors" → Answer: "blue, red, yellow" (not "The primary colors are blue, red, and yellow")

ORIGINAL QUESTION: {question}

CURRENT ANSWER: {current_answer}

TASK: Analyze the current answer and extract ONLY the essential information that directly answers the question. Remove all explanatory text, context, reasoning, or verbose language. Return the most concise possible answer that still correctly answers the question.

REFINED ANSWER:"""

        # Get verification from model
        message = HumanMessage(content=verification_prompt)
        response = verification_model.invoke([message])

        refined_answer = str(response.content).strip()
        
        # Additional post-processing to ensure conciseness
        # Remove common prefixes if they slipped through
        prefixes_to_remove = [
            "the answer is", "the result is", "answer:", "result:", 
            "refined answer:", "the", "a", "an"
        ]
        
        for prefix in prefixes_to_remove:
            if refined_answer.lower().startswith(prefix.lower()):
                refined_answer = refined_answer[len(prefix):].strip()
        
        # Remove trailing punctuation
        refined_answer = refined_answer.rstrip('.,!?')
        
        return refined_answer

    except Exception as e:
        # If verification fails, return the original answer
        return current_answer


# Get available tools list
def get_all_tools():
    """Return list of all available tools."""
    return [
        fetch_file,
        web_search,
        calculate,
        analyze_youtube_video,
        analyze_image,
        analyze_spreadsheet,
        analyze_audio,
        execute_python_code,
        reverse_text,
        verify_gaia_answer,
    ]
