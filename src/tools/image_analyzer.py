"""
Image Analyzer Tool for visual analysis using GPT-4 vision capabilities.
Handles chess positions, charts, diagrams, and other visual content in GAIA questions.
"""

import base64
import io
import os
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from PIL import Image
from pydantic import SecretStr

from ..config import Config


class ImageAnalyzer:
    """
    Handles image analysis using GPT-4 vision capabilities.
    """

    def __init__(self):
        # Use OpenAI for vision capabilities
        api_key = Config.get_default_api_key()
        self.vision_model = ChatOpenAI(
            model=Config.get_default_model(),  # GPT-4 with vision
            api_key=SecretStr(api_key) if api_key else None,
            temperature=0,
        )

    def analyze_image(
        self, task_id: str, filename: str, question_context: str = ""
    ) -> str:
        """
        Analyze an image file using GPT-4 vision.

        Args:
            task_id: The task identifier
            filename: Name of the image file
            question_context: Context from the original question

        Returns:
            String containing image analysis results
        """
        try:
            # Construct the image file path
            image_path = f"/tmp/{task_id}_{filename}"

            # Check if image exists
            if not os.path.exists(image_path):
                return f"Error: Image file not found at {image_path}"

            # Encode image to base64
            image_base64 = self._encode_image_to_base64(image_path)
            if not image_base64:
                return f"Error: Could not encode image {filename}"

            # Create analysis prompt
            analysis_prompt = self._create_analysis_prompt(filename, question_context)

            # Analyze the image
            message = HumanMessage(
                content=[
                    {"type": "text", "text": analysis_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ]
            )

            response = self.vision_model.invoke([message])

            # Extract and return the analysis
            if hasattr(response, "content"):
                analysis = str(response.content).strip()
            else:
                analysis = str(response).strip()

            return f"Image Analysis for {filename}:\n{analysis}"

        except Exception as e:
            return f"Error analyzing image {filename}: {str(e)}"

    def analyze_chess_position(
        self, task_id: str, filename: str, question_context: str = ""
    ) -> str:
        """
        Analyze a chess position image.

        Args:
            task_id: The task identifier
            filename: Name of the chess image file
            question_context: Context from the original question

        Returns:
            String containing chess position analysis
        """
        chess_prompt = f"""
        You are analyzing a chess position image. Please provide:
        1. Current position description (piece placement)
        2. Whose turn it is (if determinable)
        3. Any special conditions (check, checkmate, stalemate)
        4. Answer any specific questions about the position
        
        Question context: {question_context}
        
        Focus on accuracy and be specific about piece locations using standard chess notation.
        """

        return self._analyze_with_custom_prompt(task_id, filename, chess_prompt)

    def analyze_chart_or_graph(
        self, task_id: str, filename: str, question_context: str = ""
    ) -> str:
        """
        Analyze a chart or graph image.

        Args:
            task_id: The task identifier
            filename: Name of the chart/graph image file
            question_context: Context from the original question

        Returns:
            String containing chart/graph analysis
        """
        chart_prompt = f"""
        You are analyzing a chart or graph image. Please provide:
        1. Type of chart/graph (bar, line, pie, scatter, etc.)
        2. Key data points and values
        3. Trends or patterns visible
        4. Axis labels and units
        5. Answer any specific questions about the data
        
        Question context: {question_context}
        
        Be precise with numbers and data readings.
        """

        return self._analyze_with_custom_prompt(task_id, filename, chart_prompt)

    def analyze_document_or_text(
        self, task_id: str, filename: str, question_context: str = ""
    ) -> str:
        """
        Analyze a document or text image using OCR capabilities.

        Args:
            task_id: The task identifier
            filename: Name of the document image file
            question_context: Context from the original question

        Returns:
            String containing document text and analysis
        """
        document_prompt = f"""
        You are analyzing a document or text image. Please provide:
        1. Extract all readable text accurately
        2. Identify document type if possible
        3. Highlight key information relevant to the question
        4. Answer any specific questions about the content
        
        Question context: {question_context}
        
        Focus on accurate text extraction and comprehension.
        """

        return self._analyze_with_custom_prompt(task_id, filename, document_prompt)

    def _analyze_with_custom_prompt(
        self, task_id: str, filename: str, custom_prompt: str
    ) -> str:
        """
        Analyze image with a custom prompt.

        Args:
            task_id: The task identifier
            filename: Name of the image file
            custom_prompt: Custom analysis prompt

        Returns:
            String containing analysis results
        """
        try:
            # Construct the image file path
            image_path = f"/tmp/{task_id}_{filename}"

            # Check if image exists
            if not os.path.exists(image_path):
                return f"Error: Image file not found at {image_path}"

            # Encode image to base64
            image_base64 = self._encode_image_to_base64(image_path)
            if not image_base64:
                return f"Error: Could not encode image {filename}"

            # Analyze the image
            message = HumanMessage(
                content=[
                    {"type": "text", "text": custom_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ]
            )

            response = self.vision_model.invoke([message])

            # Extract and return the analysis
            if hasattr(response, "content"):
                analysis = str(response.content).strip()
            else:
                analysis = str(response).strip()

            return f"Image Analysis for {filename}:\n{analysis}"

        except Exception as e:
            return f"Error analyzing image {filename}: {str(e)}"

    def _create_analysis_prompt(self, filename: str, question_context: str) -> str:
        """
        Create a general analysis prompt based on filename and context.

        Args:
            filename: Name of the image file
            question_context: Context from the original question

        Returns:
            String containing the analysis prompt
        """
        # Determine likely image type from filename and context
        filename_lower = filename.lower()
        context_lower = question_context.lower()

        if "chess" in filename_lower or "chess" in context_lower:
            return self._get_chess_prompt(question_context)
        elif any(
            word in filename_lower or word in context_lower
            for word in ["chart", "graph", "plot", "data"]
        ):
            return self._get_chart_prompt(question_context)
        elif any(
            word in filename_lower or word in context_lower
            for word in ["text", "document", "page", "letter"]
        ):
            return self._get_document_prompt(question_context)
        else:
            return self._get_general_prompt(question_context)

    def _get_chess_prompt(self, question_context: str) -> str:
        """Get chess-specific analysis prompt."""
        return f"""
        Analyze this chess position image carefully. Provide:
        1. Current board position with piece locations
        2. Whose turn it is (if determinable)
        3. Any checks, threats, or special conditions
        4. Answer the specific question about this position
        
        Question: {question_context}
        
        Use standard chess notation and be precise.
        """

    def _get_chart_prompt(self, question_context: str) -> str:
        """Get chart/graph analysis prompt."""
        return f"""
        Analyze this chart or graph image carefully. Provide:
        1. Type of visualization
        2. Key data points and values
        3. Trends and patterns
        4. Axis information and units
        5. Answer the specific question about this data
        
        Question: {question_context}
        
        Be precise with numerical readings.
        """

    def _get_document_prompt(self, question_context: str) -> str:
        """Get document/text analysis prompt."""
        return f"""
        Analyze this document or text image carefully. Provide:
        1. Extract all readable text accurately
        2. Identify document structure and type
        3. Highlight relevant information
        4. Answer the specific question about this content
        
        Question: {question_context}
        
        Focus on accurate text extraction.
        """

    def _get_general_prompt(self, question_context: str) -> str:
        """Get general image analysis prompt."""
        return f"""
        Analyze this image carefully and thoroughly. Describe:
        1. What you see in the image
        2. Key visual elements and details
        3. Any text, numbers, or symbols
        4. Spatial relationships and layout
        5. Answer the specific question about this image
        
        Question: {question_context}
        
        Be detailed and accurate in your description.
        """

    def _encode_image_to_base64(self, image_path: str) -> Optional[str]:
        """
        Encode an image file to base64 string.

        Args:
            image_path: Path to the image file

        Returns:
            Base64 encoded string or None if error
        """
        try:
            # Open and potentially resize image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Resize if too large (GPT-4 vision has size limits)
                max_size = 1024
                if img.width > max_size or img.height > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

                # Convert to base64
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                img_bytes = buffer.getvalue()

                return base64.b64encode(img_bytes).decode("utf-8")

        except Exception as e:
            print(f"Error encoding image: {e}")
            return None

    def get_image_info(self, task_id: str, filename: str) -> Dict[str, Any]:
        """
        Get basic information about an image file.

        Args:
            task_id: The task identifier
            filename: Name of the image file

        Returns:
            Dictionary with image information
        """
        try:
            image_path = f"/tmp/{task_id}_{filename}"

            if not os.path.exists(image_path):
                return {"error": "Image file not found"}

            with Image.open(image_path) as img:
                return {
                    "filename": filename,
                    "format": img.format,
                    "mode": img.mode,
                    "size": img.size,
                    "width": img.width,
                    "height": img.height,
                }

        except Exception as e:
            return {"error": f"Error getting image info: {str(e)}"}


# Global instance
_image_analyzer = ImageAnalyzer()


@tool
def analyze_image(task_id: str, filename: str, question_context: str = "") -> str:
    """
    Analyze an image file using GPT-4 vision capabilities.

    Args:
        task_id: The task identifier
        filename: Name of the image file
        question_context: Context from the original question

    Returns:
        String containing image analysis results
    """
    return _image_analyzer.analyze_image(task_id, filename, question_context)


@tool
def analyze_chess_position(
    task_id: str, filename: str, question_context: str = ""
) -> str:
    """
    Analyze a chess position image specifically.

    Args:
        task_id: The task identifier
        filename: Name of the chess image file
        question_context: Context from the original question

    Returns:
        String containing chess position analysis
    """
    return _image_analyzer.analyze_chess_position(task_id, filename, question_context)


@tool
def analyze_chart_or_graph(
    task_id: str, filename: str, question_context: str = ""
) -> str:
    """
    Analyze a chart or graph image specifically.

    Args:
        task_id: The task identifier
        filename: Name of the chart/graph image file
        question_context: Context from the original question

    Returns:
        String containing chart/graph analysis
    """
    return _image_analyzer.analyze_chart_or_graph(task_id, filename, question_context)


@tool
def analyze_document_image(
    task_id: str, filename: str, question_context: str = ""
) -> str:
    """
    Analyze a document or text image using OCR capabilities.

    Args:
        task_id: The task identifier
        filename: Name of the document image file
        question_context: Context from the original question

    Returns:
        String containing document text and analysis
    """
    return _image_analyzer.analyze_document_or_text(task_id, filename, question_context)


# Export the tools for use in the main tools module
__all__ = [
    "analyze_image",
    "analyze_chess_position",
    "analyze_chart_or_graph",
    "analyze_document_image",
    "ImageAnalyzer",
]
