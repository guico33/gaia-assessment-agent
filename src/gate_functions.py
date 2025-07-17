"""
Gate Functions for LangGraph Conditional Routing.
These functions use LLM-based decision making with multi-shot prompts
to determine workflow routing instead of string pattern matching.
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import SystemMessage

from .llm_provider import get_llm


class WorkflowState(TypedDict):
    """State schema for the workflow agent - defined here to avoid circular imports"""

    question: str
    task_id: str
    file_name: Optional[str]
    context: str
    tools_used: List[str]
    step_count: int
    final_answer: str
    tool_results: Dict[str, str]
    current_analysis: str
    processing_complete: bool


class GateFunctions:
    """
    LLM-based gate functions for workflow routing decisions.
    """

    def __init__(self):
        """Initialize gate functions with LLM."""
        self.llm = get_llm()

    def determine_next_step(
        self, state: WorkflowState
    ) -> Literal["video", "calculation", "reasoning", "web_search", "file"]:
        """
        Determine the next step in the workflow based on the question and context.

        Args:
            state: Current workflow state

        Returns:
            Next step to take: "video", "calculation", "reasoning", "web_search", or "file"
        """
        try:
            # Check if we previously failed at YouTube video processing
            tools_used = state.get("tools_used", [])
            current_analysis = state.get("current_analysis", "")

            # If we tried video analysis and it failed due to transcript issues, try web search
            if (
                "analyze_youtube_video" in tools_used
                and "web_search" not in tools_used
                and any(
                    pattern in current_analysis
                    for pattern in [
                        "Transcripts are disabled",
                        "Could not retrieve transcript",
                        "No transcript found",
                    ]
                )
            ):
                print("📺➡️🔍 YouTube transcript failed, routing to web search")
                return "web_search"

            # If we tried audio analysis and it failed due to transcription issues, try web search
            if (
                "analyze_audio" in tools_used
                and "web_search" not in tools_used
                and any(
                    pattern in current_analysis
                    for pattern in [
                        "Automatic transcription not available",
                        "Could not transcribe audio file",
                        "OpenAI API key not configured",
                        "Local whisper installation not available",
                    ]
                )
            ):
                print("🎵➡️🔍 Audio transcription failed, routing to web search")
                return "web_search"

            # Check if file is available for file processing
            file_name = state.get("file_name", "")
            has_file = file_name and file_name.strip() and file_name != "None"

            # Create multi-shot prompt for next step determination
            available_options = "video, calculation, reasoning, web_search"
            if has_file:
                available_options += ", file"

            prompt = f"""
            You are a workflow router for a GAIA agent. Based on the question and context, determine the next step.

            EXAMPLES:
            Question: "What is the result of 2 + 2 * 3?"
            Context: Basic arithmetic question
            Decision: calculation

            Question: "What does the person in this image look like?"
            File: image.jpg
            Decision: file

            Question: "What is said at 2:30 in this YouTube video?"
            Context: Question mentions YouTube video
            Decision: video

            Question: "What is the capital of France?"
            Context: General knowledge question
            Decision: web_search

            Question: "Why did the Roman Empire fall?"
            Context: Complex historical question requiring analysis
            Decision: reasoning

            CURRENT TASK:
            Question: {state["question"]}
            File: {state.get("file_name", "None")}
            Context: {state.get("context", "")}
            Step Count: {state["step_count"]}
            Tools Used: {state.get("tools_used", [])}

            IMPORTANT: You can only choose "file" if there is an actual file attached (File is not "None").
            
            Available options: {available_options}
            
            Based on this information, what is the next step?
            Respond with ONLY one word from the available options.
            """

            messages = [SystemMessage(content=prompt)]
            response = self.llm.invoke(messages)

            # Extract decision from response
            if hasattr(response, "content"):
                decision = str(response.content).strip().lower()
            else:
                decision = str(response).strip().lower()

            # Validate decision based on available options
            valid_decisions = ["video", "calculation", "reasoning", "web_search"]
            if has_file:
                valid_decisions.append("file")

            # If LLM chose "file" but no file is available, redirect to reasoning
            if decision == "file" and not has_file:
                print("⚠️ LLM chose 'file' but no file attached, routing to reasoning")
                return "reasoning"

            if decision in valid_decisions:
                return decision  # type: ignore
            else:
                # Fallback to web_search if invalid decision
                return "web_search"

        except Exception as e:
            print(f"Error in determine_next_step: {e}")
            return "web_search"  # Safe fallback

    def assess_answer_quality(
        self, state: WorkflowState
    ) -> Literal["satisfactory", "need_further_processing"]:
        """
        Assess if the current answer is satisfactory or needs more processing.
        Uses math-specific logic and LLM-based assessment to prevent infinite loops.

        Args:
            state: Current workflow state

        Returns:
            "satisfactory" if ready for GAIA formatting, "need_further_processing" if more work needed
        """
        try:
            # Hard limits to prevent infinite loops
            max_steps = 10
            max_retries = 3

            # If we've exceeded limits, force satisfactory
            if state["step_count"] >= max_steps:
                print(f"⚠️ Step limit reached ({max_steps}), forcing completion")
                return "satisfactory"

            # Count how many times we've been in this loop
            tools_used = state.get("tools_used", [])
            retry_count = tools_used.count(tools_used[-1]) if tools_used else 0

            if retry_count >= max_retries:
                print(f"⚠️ Retry limit reached ({max_retries}), forcing completion")
                return "satisfactory"

            # Get current answers and analysis
            current_answer = state.get("final_answer", "").strip()
            current_analysis = state.get("current_analysis", "").strip()
            question = state.get("question", "").lower()

            # Check for error patterns that suggest we should stop trying
            error_patterns = [
                "Error:",
                "failed:",
                "Could not",
                "Unable to",
                "No transcript found",
                "Transcripts are disabled",
                "not found",
                "Permission denied",
            ]

            analysis_has_error = any(
                pattern in current_analysis for pattern in error_patterns
            )

            # QUICK CHECKS: Math-specific logic for numerical answers
            if self._is_math_question(question):
                # Check final answer first
                if current_answer and self._is_numerical_answer(current_answer):
                    print(
                        f"✅ Math question with numerical answer detected: '{current_answer}' - accepting as satisfactory"
                    )
                    return "satisfactory"

                # Check if current_analysis contains a clear numerical result
                if current_analysis and self._is_numerical_answer(current_analysis):
                    print(
                        f"✅ Math question with numerical analysis detected - accepting as satisfactory"
                    )
                    return "satisfactory"

            # Special handling for YouTube transcript failures - try web search fallback
            youtube_transcript_failure = any(
                pattern in current_analysis
                for pattern in [
                    "Transcripts are disabled",
                    "Could not retrieve transcript",
                    "No transcript found",
                ]
            )

            if (
                youtube_transcript_failure
                and "analyze_youtube_video" in tools_used
                and "web_search" not in tools_used
            ):
                print(f"🔄 YouTube transcript failed, attempting web search fallback")
                return "need_further_processing"

            # Special handling for audio transcription failures - try web search fallback
            audio_transcription_failure = any(
                pattern in current_analysis
                for pattern in [
                    "Automatic transcription not available",
                    "Could not transcribe audio file",
                    "OpenAI API key not configured",
                    "Local whisper installation not available",
                ]
            )

            if (
                audio_transcription_failure
                and "analyze_audio" in tools_used
                and "web_search" not in tools_used
            ):
                print(f"🔄 Audio transcription failed, attempting web search fallback")
                return "need_further_processing"

            if analysis_has_error and retry_count >= 2:
                print(f"⚠️ Repeated errors detected, forcing completion")
                return "satisfactory"

            # If we have a reasonable answer (not just error), consider it satisfactory
            if (
                current_answer
                and current_answer != "No answer yet"
                and not any(pattern in current_answer for pattern in error_patterns)
            ):
                print(
                    f"✅ Reasonable answer detected: '{current_answer[:50]}...' - accepting as satisfactory"
                )
                return "satisfactory"

            # If we have some analysis content, even if not perfect, consider it satisfactory
            if current_analysis and len(current_analysis) > 50:
                # But don't accept error messages as satisfactory unless we've tried alternatives
                if not analysis_has_error or retry_count >= 1:
                    print(
                        f"✅ Substantial analysis content detected ({len(current_analysis)} chars) - accepting as satisfactory"
                    )
                    return "satisfactory"

            # If we successfully used web search after YouTube failure, consider it satisfactory
            if (
                "web_search" in tools_used
                and "analyze_youtube_video" in tools_used
                and current_analysis
                and len(current_analysis) > 100
            ):
                print(
                    f"✅ Web search fallback successful after YouTube failure - accepting as satisfactory"
                )
                return "satisfactory"

            # LLM-based assessment for complex cases
            if state["step_count"] >= 3 and (current_answer or current_analysis):
                llm_decision = self._llm_assess_quality(
                    question, current_answer, current_analysis, tools_used
                )
                if llm_decision == "satisfactory":
                    print(f"✅ LLM assessment: answer quality is sufficient")
                    return "satisfactory"

            # Only try more processing if we haven't hit limits and have no substantial content
            if state["step_count"] < max_steps and retry_count < max_retries:
                print(
                    f"🔄 Need further processing (step {state['step_count']}, retry {retry_count})"
                )
                return "need_further_processing"
            else:
                print(f"⚠️ Limits reached, forcing completion")
                return "satisfactory"

        except Exception as e:
            print(f"Error in assess_answer_quality: {e}")
            # Always return satisfactory on error to prevent infinite loops
            return "satisfactory"

    def _is_math_question(self, question: str) -> bool:
        """
        Check if the question is primarily mathematical.

        Args:
            question: The question text (already lowercased)

        Returns:
            True if this appears to be a math question
        """
        math_keywords = [
            "+",
            "-",
            "*",
            "/",
            "=",
            "×",
            "÷",
            "plus",
            "minus",
            "times",
            "divided",
            "calculate",
            "compute",
            "solve",
            "add",
            "subtract",
            "multiply",
            "divide",
            "sum",
            "product",
            "difference",
            "quotient",
            "equation",
            "formula",
            "percentage",
            "percent",
            "%",
            "average",
            "mean",
            "total",
            "result",
            "number",
            "digit",
            "integer",
            "decimal",
            "fraction",
            "math",
            "mathematical",
        ]

        return any(keyword in question for keyword in math_keywords)

    def _is_numerical_answer(self, text: str) -> bool:
        """
        Check if the text contains a clear numerical answer.

        Args:
            text: Text to check for numerical content

        Returns:
            True if text appears to be primarily numerical
        """
        import re

        if not text:
            return False

        # Clean the text
        text_clean = text.strip()

        # Check for pure numbers (integers or decimals)
        if re.match(r"^-?\d+\.?\d*$", text_clean):
            return True

        # Check for numbers with units or simple text
        number_patterns = [
            r"\b\d+\.?\d*\b",  # Any number
            r"\b\d+\s*(percent|%|dollars?|\$|cents?|points?|units?)\b",  # Number with unit
            r"^\s*\$?\d+\.?\d*\s*$",  # Money amounts
            r"^\s*\d+\.?\d*\s*(kg|g|lb|oz|m|cm|ft|in|l|ml)\s*$",  # Measurements
        ]

        # If it's a short text (< 20 chars) and contains numbers, likely numerical
        if len(text_clean) < 20:
            for pattern in number_patterns:
                if re.search(pattern, text_clean, re.IGNORECASE):
                    return True

        # Check if the text is mostly numbers (for longer responses)
        numbers_found = re.findall(r"\d+\.?\d*", text_clean)
        if numbers_found and len("".join(numbers_found)) >= len(text_clean) * 0.3:
            return True

        # Special case: if text contains mathematical language and numbers, likely numerical
        if numbers_found and len(text_clean) < 100:  # Reasonably short text
            math_indicator_patterns = [
                r"\b(equals?|is|=|result|answer|sum|total|product|difference)\s*\d+",
                r"\d+\s*[+\-*/×÷]\s*\d+\s*=\s*\d+",
                r"(calculate|compute|solve).*\d+",
                r"\d+.*\b(percent|%|dollars?|\$|cents?)\b",
            ]

            for pattern in math_indicator_patterns:
                if re.search(pattern, text_clean, re.IGNORECASE):
                    return True

        return False

    def _llm_assess_quality(
        self,
        question: str,
        current_answer: str,
        current_analysis: str,
        tools_used: list,
    ) -> str:
        """
        Use LLM to assess answer quality for complex cases.

        Args:
            question: The original question
            current_answer: Current final answer
            current_analysis: Current analysis
            tools_used: List of tools that have been used

        Returns:
            "satisfactory" or "need_further_processing"
        """
        try:
            assessment_prompt = f"""
            You are assessing whether an AI agent has provided a sufficient answer to a question.
            
            Question: {question}
            Current Answer: {current_answer or "No final answer yet"}
            Current Analysis: {current_analysis or "No analysis yet"}
            Tools Used: {tools_used}
            
            EXAMPLES OF SATISFACTORY ANSWERS:
            Question: "What is 15 + 25?"
            Answer: "40"
            Assessment: satisfactory
            
            Question: "What color is the sky?"
            Answer: "The sky is blue during clear weather conditions."
            Assessment: satisfactory
            
            Question: "Analyze this image of a chess board"
            Answer: "The image shows a chess position with the white king on e1 and black pieces positioned for attack."
            Assessment: satisfactory
            
            Question: "What does the speaker say about economics in the audio?"
            Answer: "Error: Could not transcribe audio file"
            Assessment: need_further_processing
            
            ASSESSMENT CRITERIA:
            1. Does the answer directly address the question?
            2. Is the answer specific and informative?
            3. For math questions: Is there a numerical result?
            4. For analysis questions: Is there substantive content?
            5. Are there obvious errors that could be corrected?
            
            Consider "satisfactory" if:
            - The answer directly addresses the question
            - For math: Contains a numerical result
            - For factual questions: Provides relevant information
            - For analysis: Contains substantive insights
            
            Consider "need_further_processing" if:
            - Answer is clearly wrong or incomplete
            - Contains only error messages that could be worked around
            - No substantial information provided
            - Answer doesn't address the question at all
            
            Respond with ONLY: satisfactory or need_further_processing
            """

            messages = [SystemMessage(content=assessment_prompt)]
            response = self.llm.invoke(messages)

            # Extract decision from response
            if hasattr(response, "content"):
                decision = str(response.content).strip().lower()
            else:
                decision = str(response).strip().lower()

            if "satisfactory" in decision:
                return "satisfactory"
            elif "need_further_processing" in decision:
                return "need_further_processing"
            else:
                # Default to satisfactory to prevent infinite loops
                return "satisfactory"

        except Exception as e:
            print(f"Error in LLM quality assessment: {e}")
            # Default to satisfactory on error to prevent infinite loops
            return "satisfactory"

    def route_file_analysis(
        self, state: WorkflowState
    ) -> Literal["image", "audio", "excel", "text", "skip"]:
        """
        Route file analysis to the appropriate tool based on file type and question.

        Args:
            state: Current workflow state

        Returns:
            File analysis route: "image", "audio", "excel", "text", or "skip"
        """
        try:
            file_name = state.get("file_name", "")

            # If no file, skip
            if not file_name:
                return "skip"

            # Create multi-shot prompt for file routing
            prompt = f"""
            You are a file analysis router. Based on the file name and question, determine the appropriate analysis tool.

            EXAMPLES:
            File: "chess_board.png"
            Question: "What chess piece is on e4?"
            Route: image

            File: "data.xlsx"
            Question: "What is the sum of column A?"
            Route: excel

            File: "speech.mp3"
            Question: "What does the speaker say about economics?"
            Route: audio

            File: "document.pdf"
            Question: "What is the main topic?"
            Route: text

            File: "chart.jpg"
            Question: "What is the highest value?"
            Route: image

            CURRENT TASK:
            File: {file_name}
            Question: {state["question"]}
            Context: {state.get("context", "")}

            Based on the file extension and question, which analysis tool should be used?
            Consider:
            - File extension (.png, .jpg, .mp3, .xlsx, .pdf, etc.)
            - Question requirements (visual analysis, data calculation, audio transcription, text extraction)

            Respond with ONLY: image, audio, excel, text, or skip
            """

            messages = [SystemMessage(content=prompt)]
            response = self.llm.invoke(messages)

            # Extract decision from response
            if hasattr(response, "content"):
                decision = str(response.content).strip().lower()
            else:
                decision = str(response).strip().lower()

            # Validate decision and provide fallback based on file extension
            valid_routes = ["image", "audio", "excel", "text", "skip"]
            if decision in valid_routes:
                return decision  # type: ignore
            else:
                # Fallback based on file extension
                if file_name:
                    ext = file_name.lower().split(".")[-1] if "." in file_name else ""
                    if ext in ["png", "jpg", "jpeg", "gif", "bmp", "webp"]:
                        return "image"
                    elif ext in ["mp3", "wav", "flac", "m4a"]:
                        return "audio"
                    elif ext in ["xlsx", "xls", "csv"]:
                        return "excel"
                    elif ext in ["pdf", "txt", "md"]:
                        return "text"

                return "skip"

        except Exception as e:
            print(f"Error in route_file_analysis: {e}")
            return "skip"

    def should_continue_processing(
        self, state: WorkflowState
    ) -> Literal["continue", "stop"]:
        """
        Determine if processing should continue or stop based on limits and progress.

        Args:
            state: Current workflow state

        Returns:
            "continue" to keep processing, "stop" to end workflow
        """
        try:
            # Check hard limits first
            if state["step_count"] >= 8:  # Maximum steps
                return "stop"

            if len(state.get("tools_used", [])) >= 5:  # Maximum tool uses
                return "stop"

            # If we have a final answer, we can stop
            if (
                state.get("final_answer")
                and state.get("final_answer") != "No answer yet"
            ):
                return "stop"

            # Create LLM-based decision for complex cases
            prompt = f"""
            You are a workflow continuation assessor. Determine if processing should continue or stop.

            EXAMPLES:
            Question: "What is 2 + 2?"
            Answer: "4"
            Tools Used: ["calculate_expression"]
            Steps: 2
            Decision: stop

            Question: "Analyze the image"
            Answer: "Error: Could not analyze image"
            Tools Used: ["fetch_file"]
            Steps: 2
            Decision: continue

            Question: "What is the capital of France?"
            Answer: "Paris"
            Tools Used: ["web_search"]
            Steps: 3
            Decision: stop

            CURRENT TASK:
            Question: {state["question"]}
            Current Answer: {state.get("final_answer", "No answer yet")}
            Tools Used: {state.get("tools_used", [])}
            Steps: {state["step_count"]}

            Should processing continue or stop?
            Consider:
            1. Do we have a reasonable answer?
            2. Are there obvious errors that could be fixed?
            3. Have we tried enough approaches?
            4. Are we making progress?

            Respond with ONLY: continue or stop
            """

            messages = [SystemMessage(content=prompt)]
            response = self.llm.invoke(messages)

            # Extract decision from response
            if hasattr(response, "content"):
                decision = str(response.content).strip().lower()
            else:
                decision = str(response).strip().lower()

            # Validate decision
            if "continue" in decision:
                return "continue"
            elif "stop" in decision:
                return "stop"
            else:
                # Default to continue if uncertain, but respect hard limits
                return "continue"

        except Exception as e:
            print(f"Error in should_continue_processing: {e}")
            return "stop"  # Safe fallback


# Global instance
gate_functions = GateFunctions()


# Export individual functions for use in workflow
def determine_next_step(
    state: WorkflowState,
) -> Literal["video", "calculation", "reasoning", "web_search", "file"]:
    """Gate function for determining next workflow step."""
    return gate_functions.determine_next_step(state)


def assess_answer_quality(
    state: WorkflowState,
) -> Literal["satisfactory", "need_further_processing"]:
    """Gate function for assessing answer quality."""
    return gate_functions.assess_answer_quality(state)


def route_file_analysis(
    state: WorkflowState,
) -> Literal["image", "audio", "excel", "text", "skip"]:
    """Gate function for routing file analysis."""
    return gate_functions.route_file_analysis(state)


def should_continue_processing(state: WorkflowState) -> Literal["continue", "stop"]:
    """Gate function for determining if processing should continue."""
    return gate_functions.should_continue_processing(state)


# Export all functions
__all__ = [
    "GateFunctions",
    "determine_next_step",
    "assess_answer_quality",
    "route_file_analysis",
    "should_continue_processing",
    "gate_functions",
]
