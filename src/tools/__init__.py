"""
Main tools module for the GAIA Agent.
Provides unified access to all tools and the get_all_tools() function.
"""

from typing import List, Dict, Any
from langchain_core.tools import BaseTool

# Import all tools from individual modules
from .file_fetcher import fetch_file, get_file_info
from .web_search import (
    web_search, 
    search_wikipedia, 
    search_specific_site, 
    search_academic, 
    search_news
)
from .code_agent import (
    execute_python_code,
    solve_math_problem,
    analyze_data,
    calculate_expression,
    process_computational_task
)
from .answer_verifier import (
    verify_gaia_answer,
    assess_answer_quality,
    suggest_answer_improvements
)
from .image_analyzer import (
    analyze_image,
    analyze_chess_position,
    analyze_chart_or_graph,
    analyze_document_image
)
from .youtube_analyzer import (
    analyze_youtube_video,
    extract_youtube_transcript,
    search_youtube_transcript,
    get_youtube_video_info
)
from .audio_analyzer import (
    analyze_audio,
    transcribe_audio,
    search_audio_transcript,
    get_audio_file_info
)
from .spreadsheet_analyzer import (
    analyze_spreadsheet,
    get_spreadsheet_summary,
    query_spreadsheet_data,
    get_spreadsheet_info
)

# Tool categories for organized access
CORE_TOOLS = [
    fetch_file,
    get_file_info,
    verify_gaia_answer,
    assess_answer_quality,
    suggest_answer_improvements
]

SEARCH_TOOLS = [
    web_search,
    search_wikipedia,
    search_specific_site,
    search_academic,
    search_news
]

CODE_TOOLS = [
    execute_python_code,
    solve_math_problem,
    analyze_data,
    calculate_expression,
    process_computational_task
]

ANALYSIS_TOOLS = [
    analyze_image,
    analyze_chess_position,
    analyze_chart_or_graph,
    analyze_document_image,
    analyze_youtube_video,
    extract_youtube_transcript,
    search_youtube_transcript,
    get_youtube_video_info,
    analyze_audio,
    transcribe_audio,
    search_audio_transcript,
    get_audio_file_info,
    analyze_spreadsheet,
    get_spreadsheet_summary,
    query_spreadsheet_data,
    get_spreadsheet_info
]

def get_all_tools() -> List[BaseTool]:
    """
    Get all available tools for the GAIA Agent.
    
    Returns:
        List of all available tools
    """
    all_tools = []
    all_tools.extend(CORE_TOOLS)
    all_tools.extend(SEARCH_TOOLS)
    all_tools.extend(CODE_TOOLS)
    all_tools.extend(ANALYSIS_TOOLS)
    
    return all_tools

def get_tools_by_category(category: str) -> List[BaseTool]:
    """
    Get tools by category.
    
    Args:
        category: Tool category ('core', 'search', 'code', 'analysis')
        
    Returns:
        List of tools in the specified category
    """
    category_map = {
        'core': CORE_TOOLS,
        'search': SEARCH_TOOLS,
        'code': CODE_TOOLS,
        'analysis': ANALYSIS_TOOLS
    }
    
    return category_map.get(category.lower(), [])

def get_tool_by_name(name: str) -> BaseTool:
    """
    Get a specific tool by name.
    
    Args:
        name: Name of the tool
        
    Returns:
        The requested tool or None if not found
    """
    all_tools = get_all_tools()
    
    for tool in all_tools:
        if tool.name == name:
            return tool
    
    return None

def get_tools_info() -> Dict[str, Any]:
    """
    Get information about all available tools.
    
    Returns:
        Dictionary containing tool information
    """
    all_tools = get_all_tools()
    
    info = {
        'total_tools': len(all_tools),
        'categories': {
            'core': len(CORE_TOOLS),
            'search': len(SEARCH_TOOLS),
            'code': len(CODE_TOOLS),
            'analysis': len(ANALYSIS_TOOLS)
        },
        'tools': [
            {
                'name': tool.name,
                'description': tool.description,
                'category': _get_tool_category(tool.name)
            }
            for tool in all_tools
        ]
    }
    
    return info

def _get_tool_category(tool_name: str) -> str:
    """
    Get the category of a tool by its name.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Category string
    """
    for tool in CORE_TOOLS:
        if tool.name == tool_name:
            return 'core'
    
    for tool in SEARCH_TOOLS:
        if tool.name == tool_name:
            return 'search'
    
    for tool in CODE_TOOLS:
        if tool.name == tool_name:
            return 'code'
    
    for tool in ANALYSIS_TOOLS:
        if tool.name == tool_name:
            return 'analysis'
    
    return 'unknown'

# Legacy function for backward compatibility
def clear_search_cache():
    """
    Clear search cache (placeholder for backward compatibility).
    """
    pass

# Export everything for direct import
__all__ = [
    # Main functions
    'get_all_tools',
    'get_tools_by_category',
    'get_tool_by_name',
    'get_tools_info',
    'clear_search_cache',
    
    # Core tools
    'fetch_file',
    'get_file_info',
    'verify_gaia_answer',
    'assess_answer_quality',
    'suggest_answer_improvements',
    
    # Search tools
    'web_search',
    'search_wikipedia',
    'search_specific_site',
    'search_academic',
    'search_news',
    
    # Code tools
    'execute_python_code',
    'solve_math_problem',
    'analyze_data',
    'calculate_expression',
    'process_computational_task',
    
    # Analysis tools
    'analyze_image',
    'analyze_chess_position',
    'analyze_chart_or_graph',
    'analyze_document_image',
    'analyze_youtube_video',
    'extract_youtube_transcript',
    'search_youtube_transcript',
    'get_youtube_video_info',
    'analyze_audio',
    'transcribe_audio',
    'search_audio_transcript',
    'get_audio_file_info',
    'analyze_spreadsheet',
    'get_spreadsheet_summary',
    'query_spreadsheet_data',
    'get_spreadsheet_info',
    
    # Tool categories
    'CORE_TOOLS',
    'SEARCH_TOOLS',
    'CODE_TOOLS',
    'ANALYSIS_TOOLS'
]