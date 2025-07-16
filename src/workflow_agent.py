"""
GAIA Agent Implementation using LangGraph StateGraph
Structured workflow to prevent infinite loops and ensure GAIA compliance
"""

import re
from typing import Dict, List, Optional, TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage
from .config import Config
from .llm_provider import get_llm
from .tools import get_all_tools, fetch_file, verify_gaia_answer

class WorkflowState(TypedDict):
    """State schema for the workflow agent"""
    question: str
    task_id: str
    file_name: Optional[str]
    context: str  # Accumulated findings
    tools_used: List[str]
    step_count: int
    confidence: float
    final_answer: str
    question_type: str  # 'file', 'math', 'youtube', 'factual', 'reasoning'
    tool_results: Dict[str, str]  # Store results by tool name
    
class WorkflowAgent:
    def __init__(self):
        self.llm = get_llm()
        self.tools = get_all_tools()
        self.graph = self._create_graph()
        
    def _create_graph(self):
        """Create the StateGraph workflow"""
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("classify_question", self._classify_question)
        workflow.add_node("handle_file", self._handle_file)
        workflow.add_node("execute_tools", self._execute_tools)
        workflow.add_node("synthesize_answer", self._synthesize_answer)
        workflow.add_node("verify_gaia_format", self._verify_gaia_format)
        
        # Define edges
        workflow.set_entry_point("classify_question")
        workflow.add_conditional_edges(
            "classify_question",
            self._should_handle_file,
            {
                "file": "handle_file",
                "no_file": "execute_tools"
            }
        )
        workflow.add_edge("handle_file", "execute_tools")
        workflow.add_conditional_edges(
            "execute_tools",
            self._should_continue_tools,
            {
                "continue": "execute_tools",
                "synthesize": "synthesize_answer"
            }
        )
        workflow.add_edge("synthesize_answer", "verify_gaia_format")
        workflow.add_edge("verify_gaia_format", END)
        
        return workflow.compile()
    
    def _classify_question(self, state: WorkflowState) -> WorkflowState:
        """Classify the question type and determine required tools"""
        print(f"🔍 Step {state['step_count'] + 1} - Classifying question")
        
        question = state["question"]
        file_name = state.get("file_name", "")
        
        # Initialize context with question
        context = f"Question: {question}\n"
        if file_name:
            context += f"File available: {file_name}\n"
        
        # Determine question type
        question_type = "factual"  # default
        
        if file_name and file_name.strip():
            question_type = "file"
        elif self._is_math_question(question):
            question_type = "math"
        elif self._is_youtube_question(question):
            question_type = "youtube"
        elif self._is_reasoning_question(question):
            question_type = "reasoning"
        
        print(f"📊 Question classified as: {question_type}")
        
        return {
            **state,
            "context": context,
            "question_type": question_type,
            "step_count": state["step_count"] + 1,
            "confidence": 0.2,  # Initial confidence
            "tool_results": {}
        }
    
    def _handle_file(self, state: WorkflowState) -> WorkflowState:
        """Handle file fetching and analysis"""
        print(f"📁 Step {state['step_count'] + 1} - Handling file: {state.get('file_name', 'N/A')}")
        
        if not state.get("file_name"):
            return {**state, "step_count": state["step_count"] + 1}
        
        try:
            # Fetch file content
            file_result = fetch_file.invoke({
                "task_id": state["task_id"],
                "filename": state["file_name"]
            })
            
            # Update context with file content
            context = state["context"] + f"\nFile Content:\n{file_result}\n"
            
            print(f"📄 File fetched successfully")
            
            return {
                **state,
                "context": context,
                "step_count": state["step_count"] + 1,
                "confidence": 0.4,  # Increased confidence after file fetch
                "tools_used": ["fetch_file"],
                "tool_results": {"fetch_file": file_result}
            }
            
        except Exception as e:
            print(f"❌ File handling error: {e}")
            context = state["context"] + f"\nFile fetch failed: {str(e)}\n"
            return {
                **state,
                "context": context,
                "step_count": state["step_count"] + 1,
                "confidence": 0.3
            }
    
    def _execute_tools(self, state: WorkflowState) -> WorkflowState:
        """Execute tools based on question type with strict limits"""
        print(f"🔧 Step {state['step_count'] + 1} - Executing tools")
        
        # Check limits
        if len(state["tools_used"]) >= 3:  # Max 3 tool calls
            print("⚠️ Tool limit reached (3 calls)")
            return {**state, "step_count": state["step_count"] + 1}
        
        if state["step_count"] >= 4:  # Max 5 nodes total
            print("⚠️ Step limit reached (5 nodes)")
            return {**state, "step_count": state["step_count"] + 1}
        
        # Select tool based on question type and current state
        tool_to_use = self._select_next_tool(state)
        
        if not tool_to_use:
            print("✅ No more tools needed")
            return {**state, "step_count": state["step_count"] + 1}
        
        print(f"🛠️ Using tool: {tool_to_use}")
        
        try:
            # Execute the selected tool
            tool_result = self._execute_single_tool(tool_to_use, state)
            
            # Update state with results
            updated_context = state["context"] + f"\n{tool_to_use} Result:\n{tool_result}\n"
            updated_tools = state["tools_used"] + [tool_to_use]
            updated_results = {**state["tool_results"], tool_to_use: tool_result}
            
            # Update confidence based on tool success
            new_confidence = min(state["confidence"] + 0.2, 1.0)
            
            return {
                **state,
                "context": updated_context,
                "tools_used": updated_tools,
                "tool_results": updated_results,
                "step_count": state["step_count"] + 1,
                "confidence": new_confidence
            }
            
        except Exception as e:
            print(f"❌ Tool execution error: {e}")
            error_context = state["context"] + f"\nTool {tool_to_use} failed: {str(e)}\n"
            return {
                **state,
                "context": error_context,
                "step_count": state["step_count"] + 1
            }
    
    def _synthesize_answer(self, state: WorkflowState) -> WorkflowState:
        """Generate final answer from accumulated context"""
        print(f"🧠 Step {state['step_count'] + 1} - Synthesizing answer")
        
        # Create synthesis prompt
        synthesis_prompt = f"""
        Based on the following information, provide a VERY CONCISE answer to the question.
        The answer must be extremely brief - typically just a number, name, or short phrase.

        Question: {state["question"]}

        Available Information:
        {state["context"]}

        Tools Used: {', '.join(state["tools_used"])}

        IMPORTANT: 
        - Give only the most direct answer possible
        - No explanations or reasoning
        - If it's a number, give just the number
        - If it's a name, give just the name
        - Maximum 10 words
        """
        
        try:
            messages = [SystemMessage(content=synthesis_prompt)]
            response = self.llm.invoke(messages)
            
            # Extract answer from response
            if isinstance(response.content, list):
                # Try to get the first string element from the list
                answer = ""
                for item in response.content:
                    if isinstance(item, str):
                        answer = item.strip()
                        break
                    elif isinstance(item, dict) and "content" in item and isinstance(item["content"], str):
                        answer = item["content"].strip()
                        break
            else:
                answer = str(response.content).strip()
            
            # Post-process to ensure conciseness
            answer = self._post_process_answer(answer)
            
            print(f"💡 Synthesized answer: {answer}")
            
            return {
                **state,
                "final_answer": answer,
                "step_count": state["step_count"] + 1,
                "confidence": min(state["confidence"] + 0.3, 1.0)
            }
            
        except Exception as e:
            print(f"❌ Synthesis error: {e}")
            return {
                **state,
                "final_answer": "Error generating answer",
                "step_count": state["step_count"] + 1
            }
    
    def _verify_gaia_format(self, state: WorkflowState) -> WorkflowState:
        """Verify and refine answer to meet GAIA specifications"""
        print(f"✅ Step {state['step_count'] + 1} - Verifying GAIA format")
        
        try:
            # Use verify_gaia_answer tool
            verified_result = verify_gaia_answer.invoke({
                "question": state["question"],
                "current_answer": state["final_answer"]
            })
            
            # Extract the verified answer
            verified_answer = self._extract_verified_answer(verified_result)
            
            print(f"🎯 Final verified answer: {verified_answer}")
            
            return {
                **state,
                "final_answer": verified_answer,
                "step_count": state["step_count"] + 1,
                "confidence": 1.0  # Maximum confidence after verification
            }
            
        except Exception as e:
            print(f"❌ Verification error: {e}")
            # Return original answer if verification fails
            return {
                **state,
                "step_count": state["step_count"] + 1,
                "confidence": 0.8
            }
    
    # Helper methods
    def _should_handle_file(self, state: WorkflowState) -> Literal["file", "no_file"]:
        """Determine if file handling is needed"""
        file_name = state.get("file_name")
        return "file" if file_name and file_name.strip() else "no_file"
    
    def _should_continue_tools(self, state: WorkflowState) -> Literal["continue", "synthesize"]:
        """Determine if more tools should be executed"""
        # Stop if limits reached
        if len(state["tools_used"]) >= 3 or state["step_count"] >= 4:
            return "synthesize"
        
        # Stop if high confidence
        if state["confidence"] >= 0.8:
            return "synthesize"
        
        # Continue if more tools might be helpful
        if self._select_next_tool(state):
            return "continue"
        
        return "synthesize"
    
    def _is_math_question(self, question: str) -> bool:
        """Detect if question requires mathematical calculation"""
        math_patterns = [
            r'\d+\s*[\+\-\*/\^]\s*\d+',  # Basic arithmetic
            r'calculate|compute|solve|equation|formula',
            r'sum|average|mean|median|percentage|%',
            r'square root|sqrt|log|exp|sin|cos|tan',
            r'derivative|integral|probability'
        ]
        
        question_lower = question.lower()
        return any(re.search(pattern, question_lower) for pattern in math_patterns)
    
    def _is_youtube_question(self, question: str) -> bool:
        """Detect if question involves YouTube content"""
        youtube_patterns = [
            r'youtube\.com|youtu\.be',
            r'video|watch|youtube'
        ]
        
        return any(re.search(pattern, question.lower()) for pattern in youtube_patterns)
    
    def _is_reasoning_question(self, question: str) -> bool:
        """Detect if question requires complex reasoning"""
        reasoning_keywords = [
            'why', 'how', 'explain', 'analyze', 'compare', 'contrast',
            'relationship', 'connection', 'implication', 'consequence'
        ]
        
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in reasoning_keywords)
    
    def _select_next_tool(self, state: WorkflowState) -> Optional[str]:
        """Select the next tool to use based on question type and current state"""
        used_tools = state["tools_used"]
        question_type = state["question_type"]
        
        # Web search limit check
        web_search_count = used_tools.count("web_search")
        code_execution_count = used_tools.count("execute_python_code")
        
        if question_type == "math":
            # For math questions, only use execute_python_code once
            if code_execution_count == 0:
                return "execute_python_code"
            else:
                # Math question already processed with code execution, no more tools needed
                return None
        elif question_type == "youtube" and "analyze_youtube_video" not in used_tools:
            return "analyze_youtube_video"
        elif question_type == "file":
            # Determine file analysis tool needed
            file_name = state.get("file_name", "")
            file_ext = file_name.lower().split('.')[-1] if file_name and '.' in file_name else ""
            if file_ext in ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp'] and "analyze_image" not in used_tools:
                return "analyze_image"
            elif file_ext in ['xlsx', 'xls', 'csv'] and "analyze_spreadsheet" not in used_tools:
                return "analyze_spreadsheet"
            elif file_ext in ['mp3', 'wav', 'flac', 'm4a'] and "analyze_audio" not in used_tools:
                return "analyze_audio"
            else:
                # File question already processed with appropriate tool, no more tools needed
                return None
        
        # Default to web search if under limit (for factual/reasoning questions)
        if web_search_count < 2:
            return "web_search"
        
        return None
    
    def _execute_single_tool(self, tool_name: str, state: WorkflowState) -> str:
        """Execute a single tool with appropriate parameters"""
        if tool_name == "web_search":
            return self._execute_web_search(state)
        elif tool_name == "execute_python_code":
            return self._execute_python_code(state)
        elif tool_name == "analyze_image":
            return self._execute_analyze_image(state)
        elif tool_name == "analyze_spreadsheet":
            return self._execute_analyze_spreadsheet(state)
        elif tool_name == "analyze_audio":
            return self._execute_analyze_audio(state)
        elif tool_name == "analyze_youtube_video":
            return self._execute_analyze_youtube(state)
        else:
            return f"Unknown tool: {tool_name}"
    
    def _execute_web_search(self, state: WorkflowState) -> str:
        """Execute web search tool"""
        from .tools import web_search
        
        # Extract search query from question
        query = state["question"]
        
        # Refine query if we have context
        if "File Content:" in state["context"]:
            query = f"{query} (context: file analysis)"
        
        return web_search.invoke({"query": query})
    
    def _execute_python_code(self, state: WorkflowState) -> str:
        """Execute Python code tool"""
        from .tools import execute_python_code
        
        # Generate code based on question
        code_prompt = f"""
        Generate Python code to solve: {state["question"]}
        
        Context: {state["context"]}
        
        Provide only the Python code needed to calculate the answer.
        """
        
        # Use LLM to generate code
        messages = [SystemMessage(content=code_prompt)]
        response = self.llm.invoke(messages)
        if hasattr(response, 'content'):
            if isinstance(response.content, list):
                # Extract first string or string-like content from the list
                code = ""
                for item in response.content:
                    if isinstance(item, str):
                        code = item.strip()
                        break
                    elif isinstance(item, dict) and "content" in item and isinstance(item["content"], str):
                        code = item["content"].strip()
                        break
            else:
                code = str(response.content).strip()
        else:
            code = str(response).strip()
        
        # Clean up code (remove markdown formatting)
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()
        
        return execute_python_code.invoke({"code": code})
    
    def _execute_analyze_image(self, state: WorkflowState) -> str:
        """Execute image analysis tool"""
        from .tools import analyze_image
        
        return analyze_image.invoke({
            "task_id": state["task_id"],
            "filename": state.get("file_name", ""),
            "question_context": state["question"]
        })
    
    def _execute_analyze_spreadsheet(self, state: WorkflowState) -> str:
        """Execute spreadsheet analysis tool"""
        from .tools import analyze_spreadsheet
        
        return analyze_spreadsheet.invoke({
            "task_id": state["task_id"],
            "filename": state.get("file_name", ""),
            "question_context": state["question"]
        })
    
    def _execute_analyze_audio(self, state: WorkflowState) -> str:
        """Execute audio analysis tool"""
        from .tools import analyze_audio
        
        return analyze_audio.invoke({
            "task_id": state["task_id"],
            "filename": state.get("file_name", ""),
            "question_context": state["question"]
        })
    
    def _execute_analyze_youtube(self, state: WorkflowState) -> str:
        """Execute YouTube analysis tool"""
        from .tools import analyze_youtube_video
        
        # Extract YouTube URL from question
        youtube_url = self._extract_youtube_url(state["question"])
        
        return analyze_youtube_video.invoke({
            "url": youtube_url,
            "question_context": state["question"]
        })
    
    def _extract_youtube_url(self, question: str) -> str:
        """Extract YouTube URL from question"""
        youtube_patterns = [
            r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
            r'https?://youtu\.be/[\w-]+',
            r'youtube\.com/watch\?v=[\w-]+',
            r'youtu\.be/[\w-]+'
        ]
        
        for pattern in youtube_patterns:
            match = re.search(pattern, question)
            if match:
                url = match.group(0)
                if not url.startswith('http'):
                    url = 'https://' + url
                return url
        
        return ""
    
    def _post_process_answer(self, answer: str) -> str:
        """Post-process answer to ensure conciseness"""
        # Remove common verbose prefixes
        prefixes_to_remove = [
            "The answer is",
            "Based on the analysis",
            "According to the information",
            "The result is",
            "The final answer is",
            "Answer:",
            "Result:",
            "Final answer:"
        ]
        
        answer = answer.strip()
        answer_lower = answer.lower()
        
        for prefix in prefixes_to_remove:
            if answer_lower.startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
                break
        
        # Remove punctuation at the end if it's just a period
        if answer.endswith('.') and not answer.endswith('..'):
            answer = answer[:-1]
        
        # Take first sentence if multiple sentences
        if '. ' in answer:
            answer = answer.split('. ')[0]
        
        return answer
    
    def _extract_verified_answer(self, verified_result: str) -> str:
        """Extract the verified answer from verification result"""
        # Look for "Refined answer:" or similar patterns
        if "Refined answer:" in verified_result:
            return verified_result.split("Refined answer:")[1].strip()
        elif "Final answer:" in verified_result:
            return verified_result.split("Final answer:")[1].strip()
        else:
            # Return the result as is if no pattern found
            return verified_result.strip()
    
    def process_question(self, question: str, task_id: str, file_name: str = "") -> str:
        """Process a single question through the workflow"""
        print(f"🚀 Starting workflow for question: {question[:100]}...")
        
        # Initialize state
        initial_state = WorkflowState(
            question=question,
            task_id=task_id,
            file_name=file_name,
            context="",
            tools_used=[],
            step_count=0,
            confidence=0.0,
            final_answer="",
            question_type="",
            tool_results={}
        )
        
        try:
            # Run the workflow
            result = self.graph.invoke(initial_state)
            
            # Extract final answer
            final_answer = result.get("final_answer", "Unable to determine answer")
            
            print(f"🎉 Workflow completed. Final answer: {final_answer}")
            print(f"📊 Tools used: {result.get('tools_used', [])}")
            print(f"🔄 Steps taken: {result.get('step_count', 0)}")
            print(f"💪 Final confidence: {result.get('confidence', 0.0)}")
            
            return final_answer
            
        except Exception as e:
            print(f"❌ Workflow error: {e}")
            return "Error processing question"