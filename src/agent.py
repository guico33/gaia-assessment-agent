"""
GAIA Question Answering Agent using LangGraph.
"""

from typing import Any, Dict, List, Optional
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from .llm_provider import get_llm
from .tools import get_all_tools


class GAIAAgent:
    """
    Advanced agent for answering GAIA (General AI Assistant) questions.
    Uses LangGraph with ReAct architecture and comprehensive toolset.
    """

    def __init__(
        self,
        model_provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        max_iterations: int = 15,
    ):
        """
        Initialize the GAIA agent.

        Args:
            model_provider: LLM provider ("openai" or "anthropic"). If None, auto-detects.
            model_name: Specific model name (optional)
            temperature: Model temperature for creativity vs consistency
            max_iterations: Maximum number of reasoning iterations
        """
        self.temperature = temperature
        self.max_iterations = max_iterations

        # Initialize LLM using centralized provider
        self.llm = get_llm(
            provider=model_provider, model_name=model_name, temperature=temperature
        )

        # Get all available tools
        self.tools = get_all_tools()

        # Create memory for conversation state
        self.memory = MemorySaver()

        # Create the ReAct agent
        self.agent = self._create_agent()

        # Determine the actual provider used for logging
        provider_name = model_provider or "auto-detected"
        print(
            f"GAIA Agent initialized with {provider_name} provider and {len(self.tools)} tools"
        )

    def _create_agent(self):
        """Create the LangGraph ReAct agent with tools and memory."""

        # Create agent with tools, memory, and proper configuration
        agent = create_react_agent(
            model=self.llm, tools=self.tools, checkpointer=self.memory
        )

        return agent

    def _post_process_answer(self, answer: str) -> str:
        """
        Post-process the agent's answer to ensure GAIA format compliance.
        
        Args:
            answer: Raw answer from the agent
            
        Returns:
            Processed answer following GAIA format requirements
        """
        if not answer or not isinstance(answer, str):
            return answer
        
        # Clean the answer - remove common prefixes/suffixes
        cleaned = answer.strip()
        
        # Remove common response prefixes
        prefixes_to_remove = [
            "the answer is",
            "the result is", 
            "answer:",
            "result:",
            "final answer:",
            "therefore,",
            "thus,",
            "so,",
            "hence,",
            "the",
            "a",
            "an"
        ]
        
        for prefix in prefixes_to_remove:
            pattern = rf"^{re.escape(prefix)}\s*"
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        
        # Remove punctuation at the end
        cleaned = re.sub(r'[.!?]+$', '', cleaned)
        
        # Check if it's a number (possibly with units)
        number_match = re.match(r'^([+-]?\d+(?:\.\d+)?)\s*([%$]|\w+)?', cleaned.strip())
        if number_match:
            number_part = number_match.group(1)
            unit_part = number_match.group(2)
            
            # For pure numbers, return just the number
            if not unit_part or unit_part.lower() in ['dollars', 'percent', 'percentage']:
                # Remove unnecessary decimal places
                try:
                    if '.' in number_part:
                        float_val = float(number_part)
                        if float_val.is_integer():
                            return str(int(float_val))
                    return number_part
                except ValueError:
                    pass
        
        # Check if it's a comma-separated list
        if ',' in cleaned and not any(char in cleaned for char in ['and', 'or', 'the']):
            parts = [part.strip() for part in cleaned.split(',')]
            processed_parts = []
            for part in parts:
                # Recursively process each part
                processed_part = self._post_process_answer(part)
                processed_parts.append(processed_part)
            return ', '.join(processed_parts)
        
        # For string answers, apply additional cleaning
        # Remove articles and common words at the beginning
        words = cleaned.split()
        if words:
            # Remove leading articles
            if words[0].lower() in ['the', 'a', 'an']:
                words = words[1:]
            
            # Join back and apply final cleanup
            cleaned = ' '.join(words)
            
            # Convert spelled-out numbers to digits if it's clearly a number word
            number_words = {
                'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
                'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
                'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
                'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
                'eighteen': '18', 'nineteen': '19', 'twenty': '20'
            }
            
            if cleaned.lower() in number_words:
                return number_words[cleaned.lower()]
        
        return cleaned.strip()

    def __call__(self, question: str, thread_id: str = "default") -> str:
        """
        Process a GAIA question and return the answer.

        Args:
            question: The question to answer
            thread_id: Thread ID for conversation memory

        Returns:
            String containing the final answer
        """
        try:
            print(f"\n🔍 Processing question: {question[:100]}...")
            print("=" * 80)

            # Create config for memory thread with debugging enabled
            config = RunnableConfig(
                configurable={"thread_id": thread_id},
                recursion_limit=self.max_iterations,  # Set recursion limit properly
            )

            # System prompt for GAIA questions
            system_prompt = """You are a highly capable AI assistant designed to answer complex, multi-step questions that may require:

1. **Research and Information Gathering**: Use web search and Wikipedia for factual information
2. **Mathematical Reasoning**: Perform calculations and analyze numerical data  
3. **File Analysis**: Process images, audio, video, spreadsheets, and documents
4. **Code Execution**: Run Python code for computational tasks
5. **Multi-modal Understanding**: Handle text, images, audio, and video content
6. **Critical Thinking**: Break down complex problems into manageable steps

**Approach for GAIA Questions:**
- Read the question carefully and identify what type of information or analysis is needed
- Break complex questions into smaller, manageable sub-questions  
- Use appropriate tools systematically to gather information
- For multi-step problems, work through each step carefully
- **STOP IMMEDIATELY** once you have sufficient information to answer the question

**CRITICAL ANSWER FORMAT REQUIREMENTS:**
- GAIA answers must be EXTREMELY CONCISE - often just one word, number, or short phrase
- If asked for a number: provide ONLY the number (no commas, no units like $ or %, no explanations)
- If asked for a string: provide ONLY the answer (no articles like "the", no abbreviations, spell out digits)
- If asked for a list: provide comma-separated values following above rules
- Examples: "42" not "The answer is 42" | "Paris" not "The city is Paris" | "Einstein" not "Albert Einstein"

**Tool Usage Guidelines:**
- Use tools efficiently - avoid redundant searches with similar queries
- For mathematical problems, use calculator/code execution for verification
- For file-based questions, fetch and analyze files thoroughly
- For research questions, limit to 2-3 search attempts maximum
- **TERMINATE** immediately once you have the specific answer needed

**When to Stop:**
1. You have found the exact number, word, or fact requested
2. You have completed the required calculation or analysis
3. You have sufficient information to provide a definitive answer
4. After 2-3 unsuccessful search attempts

<examples>
<question>What is the square root of 144?</question>
<output>12</output>
</examples>

<examples>
<question>What is the capital city of Australia?</question>
<output>Canberra</output>
</examples>

<examples>
<question>If a recipe calls for 2 cups of flour and you want to make half the recipe, how many cups of flour do you need?</question>
<output>1</output>
</examples>

<examples>
<question>List the primary colors in alphabetical order.</question>
<output>blue, red, yellow</output>
</examples>

<examples>
<question>What word is formed by reversing the letters in "star"?</question>
<output>rats</output>
</examples>

<examples>
<question>How many days are in February during a leap year?</question>
<output>29</output>
</examples>

Remember: GAIA values precision and brevity. Provide ONLY the requested information in the most concise form possible."""

            # Create messages with system prompt
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=question),
            ]

            # Run the agent with step-by-step logging
            print("🚀 Starting agent execution with step-by-step logging...\n")

            step_count = 0
            final_result = None

            # Use stream to get real-time updates of agent execution
            for chunk in self.agent.stream({"messages": messages}, config=config):
                step_count += 1

                # Log each step of the agent's execution
                for node_name, node_output in chunk.items():
                    print(f"📋 Step {step_count} - Node: {node_name}")

                    if "messages" in node_output:
                        latest_message = node_output["messages"][-1]

                        if hasattr(latest_message, "content"):
                            content = latest_message.content
                            if isinstance(content, str) and content.strip():
                                print(
                                    f"💭 Content: {content[:200]}{'...' if len(content) > 200 else ''}"
                                )

                        if (
                            hasattr(latest_message, "tool_calls")
                            and latest_message.tool_calls
                        ):
                            for tool_call in latest_message.tool_calls:
                                print(
                                    f"🔧 Tool Call: {tool_call.get('name', 'unknown')} - {tool_call.get('args', {})}"
                                )

                        if hasattr(latest_message, "type"):
                            print(f"📝 Message Type: {latest_message.type}")

                    print("-" * 40)

                # Store the final result
                final_result = chunk

                # Safety check to prevent infinite logging
                if step_count > self.max_iterations:
                    print(f"⚠️ Reached maximum step limit ({self.max_iterations})")
                    break

            print(f"\n✅ Agent execution completed in {step_count} steps")
            print("=" * 80)

            # Extract the final answer from the last result
            if final_result:
                # Get the last node's output
                last_node_output = list(final_result.values())[-1]
                if "messages" in last_node_output:
                    final_message = last_node_output["messages"][-1]
                    answer = final_message.content
                else:
                    answer = "No final message found"
            else:
                answer = "No result generated"

            print(f"🎯 Raw answer: {answer[:100]}...")
            
            # Post-process the answer for GAIA format compliance
            processed_answer = self._post_process_answer(answer)
            print(f"✨ Processed answer: {processed_answer}")
            
            return processed_answer

        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            print(error_msg)
            return error_msg

    def get_conversation_history(
        self, thread_id: str = "default"
    ) -> List[Dict[str, Any]]:
        """
        Get the conversation history for a thread.

        Args:
            thread_id: Thread ID to get history for

        Returns:
            List of message dictionaries
        """
        try:
            config = RunnableConfig({"configurable": {"thread_id": thread_id}})
            state = self.agent.get_state(config)

            messages = []
            for msg in state.values.get("messages", []):
                messages.append(
                    {
                        "type": msg.__class__.__name__,
                        "content": msg.content,
                        "timestamp": getattr(msg, "timestamp", None),
                    }
                )

            return messages
        except Exception as e:
            print(f"Error getting conversation history: {e}")
            return []

    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return [tool.name for tool in self.tools]


def create_gaia_agent(
    model_provider: Optional[str] = None, model_name: Optional[str] = None, **kwargs
) -> GAIAAgent:
    """
    Factory function to create a GAIA agent.

    Args:
        model_provider: LLM provider ("openai" or "anthropic"). If None, auto-detects.
        model_name: Specific model name (optional)
        **kwargs: Additional arguments for GAIAAgent

    Returns:
        Configured GAIAAgent instance
    """
    return GAIAAgent(model_provider=model_provider, model_name=model_name, **kwargs)


# Backward compatibility - simple agent class matching the original interface
class BasicAgent:
    """
    Simple wrapper around GAIAAgent for backward compatibility.
    Maintains the same interface as the original BasicAgent.
    """

    def __init__(self):
        """Initialize the basic agent with default settings."""
        print("BasicAgent initialized with GAIA capabilities.")

        # Use centralized provider - it will auto-detect available providers
        # and fail fast if none are available
        self.agent = create_gaia_agent()

    def __call__(self, question: str) -> str:
        """
        Answer a question using the GAIA agent.

        Args:
            question: The question to answer

        Returns:
            String containing the answer
        """
        return self.agent(question)
