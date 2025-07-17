"""
File Fetcher Tool for retrieving files from GAIA API submissions.
Handles various file types and provides content extraction.
"""

import requests
from typing import Optional, Dict, Any
from pathlib import Path
from langchain_core.tools import tool
from ..config import Config


class FileFetcher:
    """
    Handles fetching files from GAIA API and extracting content.
    """
    
    def __init__(self):
        """Initialize the file fetcher with API configuration."""
        self.api_url = Config.DEFAULT_API_URL
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GAIA-Agent/1.0'
        })
    
    def fetch_file(self, task_id: str, filename: str) -> str:
        """
        Fetch a file from the GAIA API.
        
        Args:
            task_id: The task identifier
            filename: Name of the file to fetch
            
        Returns:
            String containing file content or error message
        """
        if not filename or not filename.strip():
            return "No file specified"
        
        try:
            # Construct the file URL using the correct endpoint format
            file_url = f"{self.api_url}/files/{task_id}"
            
            # Fetch the file
            response = self.session.get(file_url, timeout=30)
            response.raise_for_status()
            
            # Determine file type and extract content
            file_extension = Path(filename).suffix.lower()
            
            if file_extension in ['.txt', '.md', '.py', '.js', '.json', '.csv']:
                # Text-based files
                return self._extract_text_content(response, filename)
            elif file_extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
                # Image files - return file info for later processing
                return self._handle_image_file(response, filename, task_id)
            elif file_extension in ['.mp3', '.wav', '.flac', '.m4a']:
                # Audio files - return file info for later processing  
                return self._handle_audio_file(response, filename, task_id)
            elif file_extension in ['.xlsx', '.xls']:
                # Excel files - return file info for later processing
                return self._handle_excel_file(response, filename, task_id)
            elif file_extension == '.pdf':
                # PDF files - return file info for later processing
                return self._handle_pdf_file(response, filename, task_id)
            else:
                # Unknown file type
                return f"File fetched: {filename} (binary content, {len(response.content)} bytes)"
                
        except requests.exceptions.RequestException as e:
            return f"Error fetching file {filename}: {str(e)}"
        except Exception as e:
            return f"Error processing file {filename}: {str(e)}"
    
    def _extract_text_content(self, response: requests.Response, filename: str) -> str:
        """Extract text content from response."""
        try:
            content = response.text
            return f"File: {filename}\nContent:\n{content}"
        except Exception as e:
            return f"Error reading text file {filename}: {str(e)}"
    
    def _handle_image_file(self, response: requests.Response, filename: str, task_id: str) -> str:
        """Handle image file by saving for later analysis."""
        try:
            # Save file temporarily for analysis
            file_path = f"/tmp/{task_id}_{filename}"
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            return f"Image file fetched: {filename} (saved to {file_path} for analysis)"
        except Exception as e:
            return f"Error handling image file {filename}: {str(e)}"
    
    def _handle_audio_file(self, response: requests.Response, filename: str, task_id: str) -> str:
        """Handle audio file by saving for later analysis."""
        try:
            # Save file temporarily for analysis
            file_path = f"/tmp/{task_id}_{filename}"
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            return f"Audio file fetched: {filename} (saved to {file_path} for transcription)"
        except Exception as e:
            return f"Error handling audio file {filename}: {str(e)}"
    
    def _handle_excel_file(self, response: requests.Response, filename: str, task_id: str) -> str:
        """Handle Excel file by saving for later analysis."""
        try:
            # Save file temporarily for analysis
            file_path = f"/tmp/{task_id}_{filename}"
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            return f"Excel file fetched: {filename} (saved to {file_path} for analysis)"
        except Exception as e:
            return f"Error handling Excel file {filename}: {str(e)}"
    
    def _handle_pdf_file(self, response: requests.Response, filename: str, task_id: str) -> str:
        """Handle PDF file by saving for later analysis."""
        try:
            # Save file temporarily for analysis
            file_path = f"/tmp/{task_id}_{filename}"
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            return f"PDF file fetched: {filename} (saved to {file_path} for analysis)"
        except Exception as e:
            return f"Error handling PDF file {filename}: {str(e)}"
    
    def get_file_info(self, task_id: str, filename: str) -> Dict[str, Any]:
        """
        Get basic information about a file without downloading it.
        
        Args:
            task_id: The task identifier
            filename: Name of the file
            
        Returns:
            Dictionary with file information
        """
        if not filename or not filename.strip():
            return {"error": "No file specified"}
        
        file_extension = Path(filename).suffix.lower()
        
        return {
            "filename": filename,
            "extension": file_extension,
            "type": self._get_file_type(file_extension),
            "task_id": task_id
        }
    
    def _get_file_type(self, extension: str) -> str:
        """Determine file type from extension."""
        if extension in ['.txt', '.md', '.py', '.js', '.json', '.csv']:
            return "text"
        elif extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
            return "image"
        elif extension in ['.mp3', '.wav', '.flac', '.m4a']:
            return "audio"
        elif extension in ['.xlsx', '.xls']:
            return "spreadsheet"
        elif extension == '.pdf':
            return "pdf"
        else:
            return "unknown"


# Global instance
_file_fetcher = FileFetcher()


@tool
def fetch_file(task_id: str, filename: str) -> str:
    """
    Fetch a file from the GAIA API for analysis.
    
    Args:
        task_id: The task identifier
        filename: Name of the file to fetch
        
    Returns:
        String containing file content or processing information
    """
    return _file_fetcher.fetch_file(task_id, filename)


@tool
def get_file_info(task_id: str, filename: str) -> str:
    """
    Get information about a file without downloading it.
    
    Args:
        task_id: The task identifier
        filename: Name of the file
        
    Returns:
        String containing file information
    """
    info = _file_fetcher.get_file_info(task_id, filename)
    return f"File info: {info}"


# Export the tools for use in the main tools module
__all__ = [
    'fetch_file',
    'get_file_info',
    'FileFetcher'
]