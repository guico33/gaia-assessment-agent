import asyncio
import inspect
import os
import random
import time
from typing import Any, Dict, List

import gradio as gr
import pandas as pd
import requests

from src.workflow_agent import WorkflowAgent

# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# Configuration for parallel processing
MAX_CONCURRENT_QUESTIONS = int(os.getenv("MAX_PARALLEL_QUESTIONS", "1"))


async def preprocess_question_with_files(
    question_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Pre-process question to fetch any required files and enhance the question text.

    Args:
        question_data: Original question data from API

    Returns:
        Enhanced question data with file content included
    """
    # Check if this question has an attached file
    file_name = question_data.get("file_name")
    if file_name and file_name.strip():
        task_id = question_data.get("task_id", "unknown")
        print(f"📎 Detected file attachment: {file_name} for task {task_id}")

        try:
            # Import fetch_file tool to get the file
            from src.tools import fetch_file

            # Fetch the file using task_id directly
            file_result = fetch_file.invoke({"task_id": task_id, "filename": file_name})

            if "Error:" not in file_result:
                # File was fetched successfully, content is in file_result
                # Enhance the question with file content
                enhanced_question = (
                    question_data["question"]
                    + f"\n\nAttached file '{file_name}':\n{file_result}"
                )

                # Create enhanced question data
                enhanced_data = question_data.copy()
                enhanced_data["question"] = enhanced_question
                enhanced_data["_file_fetched"] = True
                enhanced_data["_file_content"] = file_result

                print(f"✅ Successfully attached file content for {file_name}")
                return enhanced_data

            else:
                print(f"⚠️ Could not fetch file {file_name}: {file_result}")

        except Exception as e:
            print(f"❌ Error fetching file {file_name}: {e}")

    # Return original question data if no file or if file fetching failed
    return question_data


async def process_with_rate_limit_retry(
    agent_func, question_text: str, task_id: str, file_name: str = "", max_retries: int = 3
) -> str:
    """
    Process a question with rate limit retry and exponential backoff.

    Args:
        agent_func: The agent function to call
        question_text: The question to process
        task_id: Task ID for logging and file handling
        file_name: File name for file handling
        max_retries: Maximum number of retry attempts

    Returns:
        Agent response string
    """
    for attempt in range(max_retries + 1):
        try:
            # Call the agent through asyncio.to_thread
            return await asyncio.to_thread(agent_func, question_text, task_id, file_name)

        except Exception as e:
            error_str = str(e)

            # Check if it's a rate limit error
            if "rate_limit_exceeded" in error_str or "Rate limit reached" in error_str:
                if attempt < max_retries:
                    # Extract wait time from error message if available
                    wait_time = 1.0  # Default wait time
                    import re

                    wait_match = re.search(
                        r"try again in (\d+(?:\.\d+)?)(?:ms|s)", error_str
                    )
                    if wait_match:
                        wait_value = float(wait_match.group(1))
                        # Convert milliseconds to seconds if needed
                        if "ms" in error_str:
                            wait_time = wait_value / 1000
                        else:
                            wait_time = wait_value
                        # Add some buffer time
                        wait_time += 0.5
                    else:
                        # Exponential backoff if no specific wait time given
                        wait_time = (2**attempt) + random.uniform(0, 1)

                    print(
                        f"⏳ Rate limit hit for {task_id}, waiting {wait_time:.1f}s (attempt {attempt + 1}/{max_retries + 1})"
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    print(f"❌ Rate limit retries exhausted for {task_id}")
                    raise e
            else:
                # Non-rate-limit error, don't retry
                raise e

    # Should never reach here, but just in case
    raise Exception(f"Max retries exceeded for {task_id}")


async def process_single_question(
    agent: WorkflowAgent,
    question_data: Dict[str, Any],
    question_num: int,
    total_questions: int,
) -> Dict[str, Any]:
    """
    Process a single question asynchronously (like JS Promise).

    Args:
        agent: The WorkflowAgent instance
        question_data: Dictionary containing task_id and question
        question_num: Current question number for progress tracking
        total_questions: Total number of questions

    Returns:
        Dictionary with processing results
    """
    task_id = question_data.get("task_id", "unknown")
    question_text = question_data.get("question")

    # Ensure question_text is a string
    if not isinstance(question_text, str):
        return {
            "task_id": task_id,
            "question": (
                str(question_text)
                if question_text is not None
                else "No question provided"
            ),
            "answer": "ERROR: Invalid question format",
            "status": "error",
            "duration": 0,
        }

    start_time = time.time()

    try:
        print(f"🚀 [{question_num}/{total_questions}] Starting: {task_id}")

        # Pre-process question to fetch any required files
        enhanced_question_data = await preprocess_question_with_files(question_data)
        enhanced_question_text = enhanced_question_data.get("question", question_text)

        # Create unique thread ID for each question to prevent memory contamination
        unique_thread_id = f"question_{task_id}_{question_num}"

        # This is like: const result = await agent.processQuestion(question)
        # Use asyncio.to_thread to make the synchronous agent call awaitable
        # Pass file_name for automatic file handling
        file_name = question_data.get("file_name", "")
        answer = await process_with_rate_limit_retry(
            agent.process_question, enhanced_question_text, task_id, file_name
        )

        duration = time.time() - start_time
        print(
            f"✅ [{question_num}/{total_questions}] Completed: {task_id} ({duration:.1f}s)"
        )

        return {
            "task_id": task_id,
            "question": question_text,
            "answer": answer,
            "status": "success",
            "duration": duration,
        }

    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ [{question_num}/{total_questions}] Failed: {task_id} - {str(e)}")

        # Enhanced error categorization for better recovery
        error_str = str(e)
        if "rate_limit_exceeded" in error_str or "Rate limit reached" in error_str:
            error_type = "RATE_LIMIT_ERROR"
        elif "recursion limit" in error_str.lower():
            error_type = "RECURSION_LIMIT_ERROR"
        elif (
            "invalid_chat_history" in error_str.lower()
            or "tool_calls" in error_str.lower()
        ):
            error_type = "MEMORY_CONTAMINATION_ERROR"
        elif "file not found" in error_str.lower():
            error_type = "FILE_ACCESS_ERROR"
        else:
            error_type = "AGENT_ERROR"

        return {
            "task_id": task_id,
            "question": question_text,
            "answer": f"{error_type}: {e}",
            "status": "error",
            "duration": duration,
        }


async def process_all_questions_async(
    agent: WorkflowAgent, questions_data: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Process all questions in parallel with progress tracking (like Promise.all() with concurrency control).

    Args:
        agent: The WorkflowAgent instance
        questions_data: List of question dictionaries

    Returns:
        List of processing results
    """
    total_questions = len(questions_data)

    # Create semaphore to limit concurrent requests (like connection pool)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_QUESTIONS)
    completed_count = 0

    async def process_with_semaphore(
        question_data: Dict[str, Any], question_num: int
    ) -> Dict[str, Any]:
        """Process question with semaphore control"""
        nonlocal completed_count

        async with semaphore:
            result = await process_single_question(
                agent, question_data, question_num, total_questions
            )

            completed_count += 1
            progress = (completed_count / total_questions) * 100
            print(
                f"📊 Overall Progress: {completed_count}/{total_questions} ({progress:.1f}%)"
            )

            return result

    print(
        f"🎯 Processing {total_questions} questions with max {MAX_CONCURRENT_QUESTIONS} concurrent requests..."
    )

    # Clear search cache to prevent state bleeding between batches
    try:
        from src.tools import clear_search_cache

        clear_search_cache()
        print("🧹 Cleared search cache for fresh start")
    except Exception as e:
        print(f"⚠️ Could not clear search cache: {e}")

    start_time = time.time()

    # Create tasks for all questions (like creating Promise array)
    tasks = [
        process_with_semaphore(question_data, idx + 1)
        for idx, question_data in enumerate(questions_data)
    ]

    # Wait for all to complete (like Promise.all())
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions and ensure we only return dictionaries
    results: List[Dict[str, Any]] = []
    for result in raw_results:
        if isinstance(result, dict):
            results.append(result)
        elif isinstance(result, Exception):
            # Convert exceptions to error dictionaries
            results.append(
                {
                    "task_id": "unknown",
                    "question": "Exception occurred during processing",
                    "answer": f"PROCESSING ERROR: {str(result)}",
                    "status": "error",
                    "duration": 0,
                }
            )
        else:
            # Handle any other unexpected types
            results.append(
                {
                    "task_id": "unknown",
                    "question": "Unexpected result type",
                    "answer": f"UNEXPECTED ERROR: {str(result)}",
                    "status": "error",
                    "duration": 0,
                }
            )

    total_duration = time.time() - start_time
    successful_results = [r for r in results if r.get("status") == "success"]
    failed_results = [r for r in results if r.get("status") == "error"]

    print(f"🎉 All questions processed in {total_duration:.1f}s")
    print(f"✅ Successful: {len(successful_results)}, ❌ Failed: {len(failed_results)}")

    return results


def run_and_submit_all(profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the WorkflowAgent on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID")  # Get the SPACE_ID for sending link to the code

    if profile:
        username = f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent ( modify this part to create your agent)
    try:
        agent = WorkflowAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    # In the case of an app running as a hugging Face space, this link points toward your codebase ( usefull for others so please keep it public)
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            print("Fetched questions list is empty.")
            return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run your Agent (Parallel Processing with AsyncIO)
    print(
        f"🚀 Running agent on {len(questions_data)} questions using parallel processing..."
    )

    # Filter out invalid questions
    valid_questions = []
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        valid_questions.append(item)

    if not valid_questions:
        print("No valid questions to process.")
        return "No valid questions to process.", None

    try:
        # Use asyncio to process all questions in parallel (like Promise.all())
        async def run_parallel_processing():
            return await process_all_questions_async(agent, valid_questions)

        # Run the async processing
        processing_results = asyncio.run(run_parallel_processing())

        # Convert results to the expected format
        results_log = []
        answers_payload = []

        for result in processing_results:
            if isinstance(result, dict):  # Ensure it's a valid result, not an exception
                results_log.append(
                    {
                        "Task ID": result["task_id"],
                        "Question": result["question"],
                        "Submitted Answer": result["answer"],
                        "Duration (s)": f"{result.get('duration', 0):.1f}",
                        "Status": result["status"],
                    }
                )

                answers_payload.append(
                    {"task_id": result["task_id"], "submitted_answer": result["answer"]}
                )
            else:
                # Handle any unexpected exceptions
                print(f"Unexpected result type: {type(result)} - {result}")

        print(
            f"✅ Parallel processing completed! Processed {len(answers_payload)} questions."
        )

    except Exception as e:
        print(f"❌ Error during parallel processing: {e}")
        return f"Error during parallel processing: {e}", None

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission
    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload,
    }
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 GAIA Agent Evaluation Runner")
    gr.Markdown(
        """
        **Enhanced Agent with LangGraph & Multi-Modal Capabilities + Parallel Processing**

        This agent is designed to handle complex GAIA (General AI Assistant) questions that require:
        - 🔍 **Web search** and research capabilities  
        - 📊 **Mathematical reasoning** and calculations
        - 🖼️ **Image analysis** (chess positions, charts, etc.)
        - 🎵 **Audio processing** (MP3 transcription)
        - 📁 **File analysis** (Excel, PDF, code files)
        - 🎥 **Video analysis** (YouTube transcripts)
        - 🧠 **Multi-step reasoning** with tool orchestration
        - ⚡ **Parallel processing** for faster evaluation (3-5x speedup)

        **Performance Features:**
        - **Async Processing**: Questions processed concurrently using AsyncIO
        - **Real-time Progress**: Live updates showing completion status
        - **Configurable Concurrency**: Set `MAX_PARALLEL_QUESTIONS` env var (default: 3)
        - **Error Resilience**: Individual failures don't stop the entire evaluation

        **Instructions:**
        1. **Set up API keys**: Add `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` to environment variables
        2. **Optional**: Add `TAVILY_API_KEY` for enhanced web search capabilities
        3. **Optional**: Set `MAX_PARALLEL_QUESTIONS=5` for higher concurrency (if your API allows)
        4. Log in to your Hugging Face account using the button below
        5. Click 'Run Evaluation & Submit All Answers' to test the agent

        ---
        **Performance**: Evaluation time reduced from ~10 minutes to ~2-3 minutes for 20 questions!
        The agent processes multiple questions simultaneously while maintaining accuracy.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(
        label="Run Status / Submission Result", lines=5, interactive=False
    )
    # Removed max_rows=10 from DataFrame constructor
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(fn=run_and_submit_all, outputs=[status_output, results_table])

if __name__ == "__main__":
    print("\n" + "-" * 30 + " App Starting " + "-" * 30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")  # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup:  # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(
            f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main"
        )
    else:
        print(
            "ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined."
        )

    print("-" * (60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)
