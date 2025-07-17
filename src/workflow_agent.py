"""
New GAIA Agent Implementation using LangGraph with LLM-based conditional routing.
Follows the workflow diagram with proper conditional edges instead of string matching.
"""

import re
from typing import Dict, List, Literal, Optional

from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph

from .config import Config
from .gate_functions import (
    WorkflowState,
    assess_answer_quality,
    determine_next_step,
    route_file_analysis,
    should_continue_processing,
)
from .llm_provider import get_llm
from .tools import get_all_tools


class WorkflowAgent:
    """
    New GAIA Agent using LangGraph with LLM-based conditional routing.
    """

    def __init__(self):
        """Initialize the new workflow agent."""
        # Print current LLM provider information
        current_provider = Config.get_default_provider()
        available_providers = Config.get_available_providers()

        print(f"🤖 New GAIA Agent Initialized")
        print(f"📡 Current LLM Provider: {current_provider}")
        print(f"🔧 Available Providers: {available_providers}")

        if current_provider == "openai":
            print(f"🎯 Model: {Config.DEFAULT_OPENAI_MODEL}")
        elif current_provider == "anthropic":
            print(f"🎯 Model: {Config.DEFAULT_ANTHROPIC_MODEL}")

        print(f"🌡️ Temperature: {Config.DEFAULT_TEMPERATURE}")
        print("-" * 50)

        self.llm = get_llm()
        self.tools = get_all_tools()
        self.graph = self._create_graph()

    def _create_graph(self):
        """Create the LangGraph workflow with conditional edges."""
        workflow = StateGraph(WorkflowState)

        # Add nodes for each workflow step
        workflow.add_node("initialize", self._initialize_processing)
        workflow.add_node("fetch_file", self._fetch_file)
        workflow.add_node("analyze_image", self._analyze_image)
        workflow.add_node("analyze_audio", self._analyze_audio)
        workflow.add_node("analyze_excel", self._analyze_excel)
        workflow.add_node("analyze_text", self._analyze_text)
        workflow.add_node("extract_video_transcript", self._extract_video_transcript)
        workflow.add_node("simple_calculation", self._simple_calculation)
        workflow.add_node("call_llm_reasoning", self._call_llm_reasoning)
        workflow.add_node("web_search", self._web_search)
        workflow.add_node("assess_answer_quality", self._assess_answer_quality_node)
        workflow.add_node(
            "process_answer_to_gaia_format", self._process_answer_to_gaia_format
        )

        # Set entry point
        workflow.set_entry_point("initialize")

        # Add conditional edges following the workflow diagram
        workflow.add_conditional_edges(
            "initialize",
            determine_next_step,
            {
                "file": "fetch_file",
                "video": "extract_video_transcript",
                "calculation": "simple_calculation",
                "reasoning": "call_llm_reasoning",
                "web_search": "web_search",
            },
        )

        # File processing routes to appropriate analyzer
        workflow.add_conditional_edges(
            "fetch_file",
            route_file_analysis,
            {
                "image": "analyze_image",
                "audio": "analyze_audio",
                "excel": "analyze_excel",
                "text": "analyze_text",
                "skip": "assess_answer_quality",
            },
        )

        # All analysis nodes route to answer quality assessment
        for node in [
            "analyze_image",
            "analyze_audio",
            "analyze_excel",
            "analyze_text",
            "extract_video_transcript",
            "simple_calculation",
            "call_llm_reasoning",
            "web_search",
        ]:
            workflow.add_edge(node, "assess_answer_quality")

        # Answer quality assessment routes based on LLM decision
        workflow.add_conditional_edges(
            "assess_answer_quality",
            assess_answer_quality,
            {
                "satisfactory": "process_answer_to_gaia_format",
                "need_further_processing": "initialize",  # Loop back for more processing
            },
        )

        # Final answer processing ends the workflow
        workflow.add_edge("process_answer_to_gaia_format", END)

        return workflow.compile()

    def _initialize_processing(self, state: WorkflowState) -> WorkflowState:
        """Initialize or continue processing."""
        print(f"🔄 Step {state['step_count'] + 1} - Initialize processing")

        # Check if we should stop processing
        if should_continue_processing(state) == "stop":
            return {
                **state,
                "processing_complete": True,
                "step_count": state["step_count"] + 1,
            }

        # Initialize context if first time
        if state["step_count"] == 0:
            context = f"Question: {state['question']}\n"
            if state.get("file_name"):
                context += f"File available: {state['file_name']}\n"

            return {
                **state,
                "context": context,
                "step_count": state["step_count"] + 1,
            }

        # Continue processing
        return {**state, "step_count": state["step_count"] + 1}

    def _fetch_file(self, state: WorkflowState) -> WorkflowState:
        """Fetch file from GAIA API."""
        print(
            f"📁 Step {state['step_count'] + 1} - Fetching file: {state.get('file_name', 'N/A')}"
        )

        try:
            from .tools import fetch_file

            file_result = fetch_file.invoke(
                {"task_id": state["task_id"], "filename": state.get("file_name", "")}
            )

            updated_context = state["context"] + f"\nFile fetched: {file_result}\n"

            return {
                **state,
                "context": updated_context,
                "step_count": state["step_count"] + 1,
                "tools_used": state["tools_used"] + ["fetch_file"],
                "tool_results": {**state["tool_results"], "fetch_file": file_result},
            }

        except Exception as e:
            print(f"❌ File fetch error: {e}")
            error_context = state["context"] + f"\nFile fetch failed: {str(e)}\n"
            return {
                **state,
                "context": error_context,
                "step_count": state["step_count"] + 1,
            }

    def _analyze_image(self, state: WorkflowState) -> WorkflowState:
        """Analyze image file."""
        print(f"🖼️ Step {state['step_count'] + 1} - Analyzing image")

        try:
            from .tools import analyze_image

            result = analyze_image.invoke(
                {
                    "task_id": state["task_id"],
                    "filename": state.get("file_name", ""),
                    "question_context": state["question"],
                }
            )

            updated_context = state["context"] + f"\nImage Analysis: {result}\n"

            return {
                **state,
                "context": updated_context,
                "current_analysis": result,
                "step_count": state["step_count"] + 1,
                "tools_used": state["tools_used"] + ["analyze_image"],
                "tool_results": {**state["tool_results"], "analyze_image": result},
            }

        except Exception as e:
            print(f"❌ Image analysis error: {e}")
            error_result = f"Image analysis failed: {str(e)}"
            return {
                **state,
                "context": state["context"] + f"\n{error_result}\n",
                "current_analysis": error_result,
                "step_count": state["step_count"] + 1,
            }

    def _analyze_audio(self, state: WorkflowState) -> WorkflowState:
        """Analyze audio file."""
        print(f"🎵 Step {state['step_count'] + 1} - Analyzing audio")

        try:
            from .tools import analyze_audio

            result = analyze_audio.invoke(
                {
                    "task_id": state["task_id"],
                    "filename": state.get("file_name", ""),
                    "question_context": state["question"],
                }
            )

            updated_context = state["context"] + f"\nAudio Analysis: {result}\n"

            return {
                **state,
                "context": updated_context,
                "current_analysis": result,
                "step_count": state["step_count"] + 1,
                "tools_used": state["tools_used"] + ["analyze_audio"],
                "tool_results": {**state["tool_results"], "analyze_audio": result},
            }

        except Exception as e:
            print(f"❌ Audio analysis error: {e}")
            error_result = f"Audio analysis failed: {str(e)}"
            return {
                **state,
                "context": state["context"] + f"\n{error_result}\n",
                "current_analysis": error_result,
                "step_count": state["step_count"] + 1,
            }

    def _analyze_excel(self, state: WorkflowState) -> WorkflowState:
        """Analyze Excel/CSV file."""
        print(f"📊 Step {state['step_count'] + 1} - Analyzing spreadsheet")

        try:
            from .tools import analyze_spreadsheet

            result = analyze_spreadsheet.invoke(
                {
                    "task_id": state["task_id"],
                    "filename": state.get("file_name", ""),
                    "question_context": state["question"],
                }
            )

            updated_context = state["context"] + f"\nSpreadsheet Analysis: {result}\n"

            return {
                **state,
                "context": updated_context,
                "current_analysis": result,
                "step_count": state["step_count"] + 1,
                "tools_used": state["tools_used"] + ["analyze_spreadsheet"],
                "tool_results": {
                    **state["tool_results"],
                    "analyze_spreadsheet": result,
                },
            }

        except Exception as e:
            print(f"❌ Spreadsheet analysis error: {e}")
            error_result = f"Spreadsheet analysis failed: {str(e)}"
            return {
                **state,
                "context": state["context"] + f"\n{error_result}\n",
                "current_analysis": error_result,
                "step_count": state["step_count"] + 1,
            }

    def _analyze_text(self, state: WorkflowState) -> WorkflowState:
        """Analyze text file."""
        print(f"📄 Step {state['step_count'] + 1} - Analyzing text document")

        try:
            # For text files, the fetch_file should have already extracted the content
            # We can use LLM reasoning to analyze the text content
            file_content = state["tool_results"].get("fetch_file", "")

            if file_content:
                analysis_prompt = f"""
                Analyze this text document and answer the question.
                
                Question: {state["question"]}
                
                Document Content:
                {file_content}
                
                Provide a direct answer to the question based on the document content.
                """

                messages = [SystemMessage(content=analysis_prompt)]
                response = self.llm.invoke(messages)

                if hasattr(response, "content"):
                    result = str(response.content).strip()
                else:
                    result = str(response).strip()
            else:
                result = "No text content available for analysis"

            updated_context = state["context"] + f"\nText Analysis: {result}\n"

            return {
                **state,
                "context": updated_context,
                "current_analysis": result,
                "step_count": state["step_count"] + 1,
                "tools_used": state["tools_used"] + ["analyze_text"],
                "tool_results": {**state["tool_results"], "analyze_text": result},
            }

        except Exception as e:
            print(f"❌ Text analysis error: {e}")
            error_result = f"Text analysis failed: {str(e)}"
            return {
                **state,
                "context": state["context"] + f"\n{error_result}\n",
                "current_analysis": error_result,
                "step_count": state["step_count"] + 1,
            }

    def _extract_video_transcript(self, state: WorkflowState) -> WorkflowState:
        """Extract and analyze YouTube video transcript."""
        print(f"🎥 Step {state['step_count'] + 1} - Extracting video transcript")

        try:
            from .tools import analyze_youtube_video

            # Extract YouTube URL from question
            youtube_url = self._extract_youtube_url(state["question"])

            if youtube_url:
                result = analyze_youtube_video.invoke(
                    {"url": youtube_url, "question_context": state["question"]}
                )
            else:
                result = "No YouTube URL found in question"

            updated_context = state["context"] + f"\nVideo Analysis: {result}\n"

            return {
                **state,
                "context": updated_context,
                "current_analysis": result,
                "step_count": state["step_count"] + 1,
                "tools_used": state["tools_used"] + ["analyze_youtube_video"],
                "tool_results": {
                    **state["tool_results"],
                    "analyze_youtube_video": result,
                },
            }

        except Exception as e:
            print(f"❌ Video analysis error: {e}")
            error_result = f"Video analysis failed: {str(e)}"
            return {
                **state,
                "context": state["context"] + f"\n{error_result}\n",
                "current_analysis": error_result,
                "step_count": state["step_count"] + 1,
            }

    def _simple_calculation(self, state: WorkflowState) -> WorkflowState:
        """Perform simple calculation using code execution."""
        print(f"🔢 Step {state['step_count'] + 1} - Performing calculation")

        try:
            from .tools import solve_math_problem

            result = solve_math_problem.invoke(
                {"problem": state["question"], "context": state["context"]}
            )

            updated_context = state["context"] + f"\nCalculation Result: {result}\n"

            return {
                **state,
                "context": updated_context,
                "current_analysis": result,
                "final_answer": result,  # Set final_answer for math calculations
                "step_count": state["step_count"] + 1,
                "tools_used": state["tools_used"] + ["solve_math_problem"],
                "tool_results": {**state["tool_results"], "solve_math_problem": result},
            }

        except Exception as e:
            print(f"❌ Calculation error: {e}")
            error_result = f"Calculation failed: {str(e)}"
            return {
                **state,
                "context": state["context"] + f"\n{error_result}\n",
                "current_analysis": error_result,
                "step_count": state["step_count"] + 1,
            }

    def _call_llm_reasoning(self, state: WorkflowState) -> WorkflowState:
        """Use LLM for complex reasoning tasks."""
        print(f"🧠 Step {state['step_count'] + 1} - LLM reasoning")

        try:
            reasoning_prompt = f"""
            Please analyze this question and provide a comprehensive answer using reasoning.
            
            Question: {state["question"]}
            
            Context: {state["context"]}
            
            Provide a detailed analysis and answer to the question.
            """

            messages = [SystemMessage(content=reasoning_prompt)]
            response = self.llm.invoke(messages)

            if hasattr(response, "content"):
                result = str(response.content).strip()
            else:
                result = str(response).strip()

            updated_context = state["context"] + f"\nLLM Reasoning: {result}\n"

            return {
                **state,
                "context": updated_context,
                "current_analysis": result,
                "step_count": state["step_count"] + 1,
                "tools_used": state["tools_used"] + ["llm_reasoning"],
                "tool_results": {**state["tool_results"], "llm_reasoning": result},
            }

        except Exception as e:
            print(f"❌ LLM reasoning error: {e}")
            error_result = f"LLM reasoning failed: {str(e)}"
            return {
                **state,
                "context": state["context"] + f"\n{error_result}\n",
                "current_analysis": error_result,
                "step_count": state["step_count"] + 1,
            }

    def _web_search(self, state: WorkflowState) -> WorkflowState:
        """Perform web search for information."""
        print(f"🔍 Step {state['step_count'] + 1} - Web search")

        try:
            from .tools import web_search

            result = web_search.invoke({"query": state["question"], "max_results": 5})

            updated_context = state["context"] + f"\nWeb Search: {result}\n"

            return {
                **state,
                "context": updated_context,
                "current_analysis": result,
                "step_count": state["step_count"] + 1,
                "tools_used": state["tools_used"] + ["web_search"],
                "tool_results": {**state["tool_results"], "web_search": result},
            }

        except Exception as e:
            print(f"❌ Web search error: {e}")
            error_result = f"Web search failed: {str(e)}"
            return {
                **state,
                "context": state["context"] + f"\n{error_result}\n",
                "current_analysis": error_result,
                "step_count": state["step_count"] + 1,
            }

    def _assess_answer_quality_node(self, state: WorkflowState) -> WorkflowState:
        """Assess answer quality and prepare for conditional routing."""
        print(f"✅ Step {state['step_count'] + 1} - Assessing answer quality")

        try:
            # Just increment step count and let the conditional routing handle the logic
            return {**state, "step_count": state["step_count"] + 1}

        except Exception as e:
            print(f"❌ Answer quality assessment error: {e}")
            return {**state, "step_count": state["step_count"] + 1}

    def _process_answer_to_gaia_format(self, state: WorkflowState) -> WorkflowState:
        """Process the final answer to GAIA format."""
        print(f"🎯 Step {state['step_count'] + 1} - Processing to GAIA format")

        try:
            # Generate final answer if we don't have one yet
            if not state.get("final_answer") or state["final_answer"] == "":
                synthesis_prompt = f"""
                Based on the analysis performed, provide a concise answer to the question.
                
                Question: {state["question"]}
                
                Analysis Results:
                {state.get("current_analysis", "")}
                
                Full Context:
                {state["context"]}
                
                Provide a direct, concise answer. Be as brief as possible while fully answering the question.
                """

                messages = [SystemMessage(content=synthesis_prompt)]
                response = self.llm.invoke(messages)

                if hasattr(response, "content"):
                    preliminary_answer = str(response.content).strip()
                else:
                    preliminary_answer = str(response).strip()

                # Update state with preliminary answer
                state = {**state, "final_answer": preliminary_answer}

            # Format for GAIA
            from .tools import verify_gaia_answer

            formatted_answer = verify_gaia_answer.invoke(
                {
                    "question": state["question"],
                    "current_answer": state["final_answer"],
                    "context": state["context"],
                }
            )

            return {
                **state,
                "final_answer": formatted_answer,
                "step_count": state["step_count"] + 1,
                "processing_complete": True,
            }

        except Exception as e:
            print(f"❌ GAIA formatting error: {e}")
            # Fallback to original answer with FINAL ANSWER format
            fallback_answer = f"FINAL ANSWER: {state.get('final_answer', 'Unable to determine answer')}"
            return {
                **state,
                "final_answer": fallback_answer,
                "step_count": state["step_count"] + 1,
                "processing_complete": True,
            }

    def _extract_youtube_url(self, question: str) -> str:
        """Extract YouTube URL from question."""
        youtube_patterns = [
            r"https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+",
            r"https?://youtu\.be/[\w-]+",
            r"youtube\.com/watch\?v=[\w-]+",
            r"youtu\.be/[\w-]+",
        ]

        for pattern in youtube_patterns:
            match = re.search(pattern, question)
            if match:
                url = match.group(0)
                if not url.startswith("http"):
                    url = "https://" + url
                return url

        return ""

    def process_question(self, question: str, task_id: str, file_name: str = "") -> str:
        """Process a single question through the new workflow."""
        print(f"🚀 Starting new workflow for question: {question[:100]}...")

        # Initialize state
        initial_state = WorkflowState(
            question=question,
            task_id=task_id,
            file_name=file_name,
            context="",
            tools_used=[],
            step_count=0,
            final_answer="",
            tool_results={},
            current_analysis="",
            processing_complete=False,
        )

        try:
            # Run the workflow
            result = self.graph.invoke(initial_state)

            # Extract final answer
            final_answer = result.get("final_answer", "Unable to determine answer")

            print(f"🎉 New workflow completed. Final answer: {final_answer}")
            print(f"📊 Tools used: {result.get('tools_used', [])}")
            print(f"🔄 Steps taken: {result.get('step_count', 0)}")

            return final_answer

        except Exception as e:
            print(f"❌ New workflow error: {e}")
            return "Error processing question"
