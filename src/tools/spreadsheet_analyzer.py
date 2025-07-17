"""
Spreadsheet Analyzer Tool for Excel and CSV file analysis.
Handles data processing, calculations, and content analysis for GAIA questions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from ..llm_provider import get_llm


class SpreadsheetAnalyzer:
    """
    Handles spreadsheet file analysis and data processing.
    """
    
    def __init__(self):
        """Initialize the spreadsheet analyzer with LLM."""
        self.llm = get_llm()
    
    def analyze_spreadsheet(self, task_id: str, filename: str, question_context: str = "") -> str:
        """
        Analyze a spreadsheet file and provide insights based on the question context.
        
        Args:
            task_id: The task identifier
            filename: Name of the spreadsheet file
            question_context: Context from the original question
            
        Returns:
            String containing spreadsheet analysis results
        """
        try:
            # Construct the file path
            file_path = f"/tmp/{task_id}_{filename}"
            
            # Check if file exists
            if not Path(file_path).exists():
                return f"Error: Spreadsheet file not found at {file_path}"
            
            # Load the spreadsheet
            df = self._load_spreadsheet(file_path)
            if df is None:
                return f"Error: Could not load spreadsheet {filename}"
            
            # Get basic information about the data
            data_info = self._get_data_info(df)
            
            # Analyze the data based on question context
            analysis = self._analyze_data(df, question_context, filename)
            
            return f"Spreadsheet Analysis for {filename}:\n{data_info}\n\nAnalysis:\n{analysis}"
            
        except Exception as e:
            return f"Error analyzing spreadsheet {filename}: {str(e)}"
    
    def get_data_summary(self, task_id: str, filename: str) -> str:
        """
        Get a summary of the spreadsheet data.
        
        Args:
            task_id: The task identifier
            filename: Name of the spreadsheet file
            
        Returns:
            String containing data summary
        """
        try:
            # Construct the file path
            file_path = f"/tmp/{task_id}_{filename}"
            
            # Load the spreadsheet
            df = self._load_spreadsheet(file_path)
            if df is None:
                return f"Error: Could not load spreadsheet {filename}"
            
            # Generate summary
            summary = self._generate_data_summary(df)
            
            return f"Data Summary for {filename}:\n{summary}"
            
        except Exception as e:
            return f"Error getting data summary: {str(e)}"
    
    def query_data(self, task_id: str, filename: str, query: str) -> str:
        """
        Query the spreadsheet data based on a specific request.
        
        Args:
            task_id: The task identifier
            filename: Name of the spreadsheet file
            query: Query or question about the data
            
        Returns:
            String containing query results
        """
        try:
            # Construct the file path
            file_path = f"/tmp/{task_id}_{filename}"
            
            # Load the spreadsheet
            df = self._load_spreadsheet(file_path)
            if df is None:
                return f"Error: Could not load spreadsheet {filename}"
            
            # Process the query
            result = self._process_query(df, query)
            
            return f"Query Result for '{query}' in {filename}:\n{result}"
            
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    def _load_spreadsheet(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Load a spreadsheet file into a pandas DataFrame.
        
        Args:
            file_path: Path to the spreadsheet file
            
        Returns:
            pandas DataFrame or None if failed
        """
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.csv':
                # Try different encodings for CSV
                encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        return df
                    except UnicodeDecodeError:
                        continue
                
                # If all encodings fail, try with errors='ignore'
                df = pd.read_csv(file_path, encoding='utf-8', encoding_errors='ignore')
                return df
                
            elif file_ext in ['.xlsx', '.xls']:
                # Load Excel file
                df = pd.read_excel(file_path)
                return df
                
            else:
                print(f"Unsupported file format: {file_ext}")
                return None
                
        except Exception as e:
            print(f"Error loading spreadsheet: {e}")
            return None
    
    def _get_data_info(self, df: pd.DataFrame) -> str:
        """
        Get basic information about the DataFrame.
        
        Args:
            df: pandas DataFrame
            
        Returns:
            String containing data information
        """
        try:
            info_parts = []
            
            # Basic shape
            info_parts.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Column names and types
            info_parts.append(f"Columns: {list(df.columns)}")
            
            # Data types
            dtype_info = df.dtypes.to_dict()
            info_parts.append(f"Data types: {dtype_info}")
            
            # Missing values
            missing_values = df.isnull().sum()
            if missing_values.any():
                info_parts.append(f"Missing values: {missing_values[missing_values > 0].to_dict()}")
            
            # First few rows
            info_parts.append(f"First 3 rows:\n{df.head(3).to_string()}")
            
            return "\n".join(info_parts)
            
        except Exception as e:
            return f"Error getting data info: {str(e)}"
    
    def _analyze_data(self, df: pd.DataFrame, question_context: str, filename: str) -> str:
        """
        Analyze the data using LLM based on question context.
        
        Args:
            df: pandas DataFrame
            question_context: Context from the original question
            filename: Name of the file
            
        Returns:
            String containing analysis results
        """
        try:
            # Prepare data summary for LLM
            data_summary = self._generate_data_summary(df)
            
            # Get sample data
            sample_data = df.head(10).to_string()
            
            # Create analysis prompt
            analysis_prompt = f"""
            You are analyzing a spreadsheet file with the following characteristics:

            File: {filename}
            Question Context: {question_context}

            Data Summary:
            {data_summary}

            Sample Data (first 10 rows):
            {sample_data}

            Please analyze this data and provide:
            1. Key insights about the data structure and content
            2. Relevant statistics and patterns
            3. Specific answers to the question based on the data
            4. Any calculations or computations needed
            5. Notable trends or relationships in the data

            Focus on answering the specific question while providing comprehensive analysis.
            """
            
            # Get LLM analysis
            messages = [SystemMessage(content=analysis_prompt)]
            response = self.llm.invoke(messages)
            
            # Extract content from response
            if hasattr(response, 'content'):
                analysis = str(response.content).strip()
            else:
                analysis = str(response).strip()
            
            return analysis
            
        except Exception as e:
            return f"Error analyzing data: {str(e)}"
    
    def _generate_data_summary(self, df: pd.DataFrame) -> str:
        """
        Generate a comprehensive data summary.
        
        Args:
            df: pandas DataFrame
            
        Returns:
            String containing data summary
        """
        try:
            summary_parts = []
            
            # Basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                summary_parts.append("Numeric Columns Statistics:")
                for col in numeric_cols:
                    stats = df[col].describe()
                    summary_parts.append(f"  {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, min={stats['min']}, max={stats['max']}")
            
            # Categorical columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                summary_parts.append("Categorical Columns:")
                for col in categorical_cols:
                    unique_count = df[col].nunique()
                    top_values = df[col].value_counts().head(3)
                    summary_parts.append(f"  {col}: {unique_count} unique values, top: {top_values.to_dict()}")
            
            # Data quality
            missing_data = df.isnull().sum()
            if missing_data.any():
                summary_parts.append(f"Missing data: {missing_data[missing_data > 0].to_dict()}")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def _process_query(self, df: pd.DataFrame, query: str) -> str:
        """
        Process a specific query about the data.
        
        Args:
            df: pandas DataFrame
            query: Query string
            
        Returns:
            String containing query results
        """
        try:
            # Use LLM to understand the query and generate appropriate pandas code
            code_prompt = f"""
            You are a data analyst. Given this query about a pandas DataFrame, generate Python code to answer it.

            Query: {query}

            DataFrame info:
            - Shape: {df.shape}
            - Columns: {list(df.columns)}
            - Data types: {df.dtypes.to_dict()}

            Sample data:
            {df.head(3).to_string()}

            Generate Python code that uses the DataFrame (variable name 'df') to answer the query.
            Provide only the code, no explanations.
            """
            
            messages = [SystemMessage(content=code_prompt)]
            response = self.llm.invoke(messages)
            
            if hasattr(response, 'content'):
                code = str(response.content).strip()
            else:
                code = str(response).strip()
            
            # Clean up code (remove markdown formatting)
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].split("```")[0].strip()
            
            # Execute the code safely
            result = self._execute_pandas_code(df, code)
            
            return result
            
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    def _execute_pandas_code(self, df: pd.DataFrame, code: str) -> str:
        """
        Execute pandas code safely.
        
        Args:
            df: pandas DataFrame
            code: Python code to execute
            
        Returns:
            String containing execution results
        """
        try:
            # Create a restricted environment
            safe_globals = {
                'df': df,
                'pd': pd,
                'np': np,
                'print': print,
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'list': list,
                'dict': dict,
                'sum': sum,
                'min': min,
                'max': max,
                'round': round
            }
            
            # Capture output
            import io
            import sys
            
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            try:
                # Execute the code
                exec(code, safe_globals)
                output = captured_output.getvalue()
                
                if output:
                    return output.strip()
                else:
                    # If no output, try to evaluate the last expression
                    lines = code.strip().split('\n')
                    if lines:
                        last_line = lines[-1]
                        if not last_line.strip().startswith(('print', 'df.', 'pd.')):
                            result = eval(last_line, safe_globals)
                            return str(result)
                    
                    return "Code executed successfully (no output)"
                    
            finally:
                sys.stdout = old_stdout
                
        except Exception as e:
            return f"Error executing code: {str(e)}"
    
    def get_spreadsheet_info(self, task_id: str, filename: str) -> Dict[str, Any]:
        """
        Get basic information about a spreadsheet file.
        
        Args:
            task_id: The task identifier
            filename: Name of the spreadsheet file
            
        Returns:
            Dictionary with spreadsheet information
        """
        try:
            file_path = f"/tmp/{task_id}_{filename}"
            
            if not Path(file_path).exists():
                return {"error": "Spreadsheet file not found"}
            
            # Load the spreadsheet
            df = self._load_spreadsheet(file_path)
            if df is None:
                return {"error": "Could not load spreadsheet"}
            
            return {
                "filename": filename,
                "rows": df.shape[0],
                "columns": df.shape[1],
                "column_names": list(df.columns),
                "data_types": df.dtypes.to_dict(),
                "missing_values": df.isnull().sum().to_dict()
            }
            
        except Exception as e:
            return {"error": f"Error getting spreadsheet info: {str(e)}"}


# Global instance
_spreadsheet_analyzer = SpreadsheetAnalyzer()


@tool
def analyze_spreadsheet(task_id: str, filename: str, question_context: str = "") -> str:
    """
    Analyze a spreadsheet file and provide insights based on the question context.
    
    Args:
        task_id: The task identifier
        filename: Name of the spreadsheet file
        question_context: Context from the original question
        
    Returns:
        String containing spreadsheet analysis results
    """
    return _spreadsheet_analyzer.analyze_spreadsheet(task_id, filename, question_context)


@tool
def get_spreadsheet_summary(task_id: str, filename: str) -> str:
    """
    Get a summary of the spreadsheet data.
    
    Args:
        task_id: The task identifier
        filename: Name of the spreadsheet file
        
    Returns:
        String containing data summary
    """
    return _spreadsheet_analyzer.get_data_summary(task_id, filename)


@tool
def query_spreadsheet_data(task_id: str, filename: str, query: str) -> str:
    """
    Query the spreadsheet data based on a specific request.
    
    Args:
        task_id: The task identifier
        filename: Name of the spreadsheet file
        query: Query or question about the data
        
    Returns:
        String containing query results
    """
    return _spreadsheet_analyzer.query_data(task_id, filename, query)


@tool
def get_spreadsheet_info(task_id: str, filename: str) -> str:
    """
    Get basic information about a spreadsheet file.
    
    Args:
        task_id: The task identifier
        filename: Name of the spreadsheet file
        
    Returns:
        String containing spreadsheet information
    """
    info = _spreadsheet_analyzer.get_spreadsheet_info(task_id, filename)
    return f"Spreadsheet Info: {info}"


# Export the tools for use in the main tools module
__all__ = [
    'analyze_spreadsheet',
    'get_spreadsheet_summary',
    'query_spreadsheet_data',
    'get_spreadsheet_info',
    'SpreadsheetAnalyzer'
]