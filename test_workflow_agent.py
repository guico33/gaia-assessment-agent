#!/usr/bin/env python3
"""
Simple test script for WorkflowAgent integration
"""

import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.workflow_agent import WorkflowAgent

task_ids_to_test = [
    # "a1e91b78-d3d8-4675-bb8d-62741b4b68a6",
    # "6f37996b-2ac7-44b0-8e68-6d28256631b4",
    # "9d191bce-651d-4746-be2d-7ef8ecadb9c2",
    # "cabe07ed-9eca-40ea-8ead-410ef5e83f91",
    # "f918266a-b3e0-4914-865d-4faa564f1aef",
    "bda648d7-d618-4883-88f4-3466eabd860e",
    # "cf106601-ab4f-4af9-b045-5295fe67b37d",
    # "5a0c1adf-205e-4841-a666-7c3ef95def9d"
]


def test_workflow_agent():
    """Test basic WorkflowAgent functionality"""
    print("🧪 Testing WorkflowAgent Integration")
    print("=" * 50)

    try:
        # Initialize the agent
        print("Initializing WorkflowAgent...")
        agent = WorkflowAgent()
        print("✅ WorkflowAgent initialized successfully")

        # load question from gaia_questions.json
        with open("gaia_questions.json", "r") as f:
            questions = json.load(f)

        for q in questions:
            if q["task_id"] not in task_ids_to_test:
                continue
            print(f"🔍 Processing question: {q['question']}")
            result = agent.process_question(
                q["question"], q["task_id"], q.get("file_name")
            )
            print(f"✅ Question result: {result}")

        print("\nAll questions have been processed")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = test_workflow_agent()
    sys.exit(0 if success else 1)
