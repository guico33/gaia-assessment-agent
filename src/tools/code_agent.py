"""
Code Agent Tool using smolagents CodeAgent for safe code execution.
Replaces the custom code executor with a more robust solution.
"""

from typing import Any, Dict, Optional

from langchain_core.tools import tool
from smolagents import CodeAgent, LiteLLMModel

from ..config import Config


class CodeAgentTool:
    """
    Wrapper for smolagents CodeAgent that provides safe code execution
    and generation capabilities for GAIA questions.
    """

    def __init__(self):
        """Initialize the CodeAgent with the configured LLM."""
        # Wrap the LLM in a format compatible with smolagents
        model = LiteLLMModel(
            model_id=Config.get_default_model(), api_key=Config.get_default_api_key()
        )  # Use a default model_id
        self.agent = CodeAgent(model=model, tools=[])  # Start with no additional tools

    def execute_code_task(self, task_description: str, context: str = "") -> str:
        """
        Execute a code-related task using the CodeAgent.

        Args:
            task_description: Description of the computational task
            context: Additional context from the question or previous analysis

        Returns:
            String result of the code execution
        """
        try:
            # Construct the full task prompt
            full_task = f"""
            Task: {task_description}
            
            Context: {context}
            
            Please write and execute Python code to solve this task.
            Provide the final answer as a clear, concise result.
            """

            # Execute the task using CodeAgent
            result = self.agent.run(full_task)

            # Extract the final result
            return str(result)

        except Exception as e:
            return f"Error executing code task: {str(e)}"

    def generate_and_execute_code(self, code_prompt: str) -> str:
        """
        Generate and execute code based on a specific prompt.

        Args:
            code_prompt: Specific code generation prompt

        Returns:
            String result of the code execution
        """
        try:
            result = self.agent.run(code_prompt)
            return str(result)

        except Exception as e:
            return f"Error generating and executing code: {str(e)}"


# Global instance
_code_agent = CodeAgentTool()


@tool
def execute_python_code(code: str) -> str:
    """
    Execute Python code safely using smolagents CodeAgent.

    Args:
        code: Python code to execute

    Returns:
        String containing code execution output
    """
    return _code_agent.generate_and_execute_code(f"Execute this Python code: {code}")


@tool
def solve_math_problem(problem: str, context: str = "") -> str:
    """
    Solve a mathematical problem using code generation and execution.

    Args:
        problem: Description of the mathematical problem
        context: Additional context or constraints

    Returns:
        String containing the solution
    """
    task = f"Solve this mathematical problem: {problem}"
    return _code_agent.execute_code_task(task, context)


@tool
def analyze_data(data_description: str, question: str, context: str = "") -> str:
    """
    Analyze data and answer questions using code generation.

    Args:
        data_description: Description of the data to analyze
        question: Specific question about the data
        context: Additional context or file information

    Returns:
        String containing the analysis result
    """
    task = f"Analyze this data: {data_description}. Answer: {question}"
    return _code_agent.execute_code_task(task, context)


@tool
def calculate_expression(expression: str, context: str = "") -> str:
    """
    Calculate a mathematical expression using safe code execution.

    Args:
        expression: Mathematical expression to calculate
        context: Additional context or variable definitions

    Returns:
        String containing the calculation result
    """
    task = f"Calculate this expression: {expression}"
    return _code_agent.execute_code_task(task, context)


@tool
def process_computational_task(task_description: str, context: str = "") -> str:
    """
    Process any computational task that requires code execution.

    Args:
        task_description: Description of the computational task
        context: Additional context from the question

    Returns:
        String containing the task result
    """
    return _code_agent.execute_code_task(task_description, context)


# Export the tools for use in the main tools module
__all__ = [
    "execute_python_code",
    "solve_math_problem",
    "analyze_data",
    "calculate_expression",
    "process_computational_task",
    "CodeAgentTool",
]
