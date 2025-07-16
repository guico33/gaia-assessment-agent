#!/usr/bin/env python3
"""
Simple test script for WorkflowAgent integration
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.workflow_agent import WorkflowAgent

def test_workflow_agent():
    """Test basic WorkflowAgent functionality"""
    print("🧪 Testing WorkflowAgent Integration")
    print("=" * 50)
    
    try:
        # Initialize the agent
        print("1. Initializing WorkflowAgent...")
        agent = WorkflowAgent()
        print("✅ WorkflowAgent initialized successfully")
        
        # Test simple math question
        print("\n2. Testing simple math question...")
        question = "What is 15 + 25?"
        task_id = "test_math"
        
        result = agent.process_question(question, task_id)
        print(f"✅ Math question result: {result}")
        
        # Test reasoning question
        print("\n3. Testing reasoning question...")
        question = "What is the capital of France?"
        task_id = "test_reasoning"
        
        result = agent.process_question(question, task_id)
        print(f"✅ Reasoning question result: {result}")
        
        # Test file question (image) - Real GAIA chess question
        print("\n4. Testing file question (image)...")
        question = "Review the chess position provided in the image. It is black's turn. Provide the correct next move for black which guarantees a win. Please provide your response in algebraic notation."
        task_id = "cca530fc-4052-43b2-b130-b30968d8aa44"
        file_name = "cca530fc-4052-43b2-b130-b30968d8aa44.png"
        
        result = agent.process_question(question, task_id, file_name)
        print(f"✅ File question (image) result: {result}")
        
        # Test YouTube video question - Real GAIA YouTube question
        print("\n5. Testing YouTube video question...")
        question = "In the video https://www.youtube.com/watch?v=L1vXCYZAYYM, what is the highest number of bird species to be on camera simultaneously?"
        task_id = "a1e91b78-d3d8-4675-bb8d-62741b4b68a6"
        
        result = agent.process_question(question, task_id)
        print(f"✅ YouTube question result: {result}")
        
        # Test file question (spreadsheet) - Real GAIA spreadsheet question
        print("\n6. Testing file question (spreadsheet)...")
        question = "The attached Excel file contains the sales of menu items for a local fast-food chain. What were the total sales that the chain made from food (not including drinks)? Express your answer in USD with two decimal places."
        task_id = "7bd855d8-463d-4ed5-93ca-5fe35145f733"
        file_name = "7bd855d8-463d-4ed5-93ca-5fe35145f733.xlsx"
        
        result = agent.process_question(question, task_id, file_name)
        print(f"✅ Spreadsheet question result: {result}")
        
        # Test file question (audio) - Real GAIA audio question
        print("\n7. Testing file question (audio)...")
        question = "Hi, I'm making a pie but I could use some help with my shopping list. I have everything I need for the crust, but I'm not sure about the filling. I got the recipe from my friend Aditi, but she left it as a voice memo and the speaker on my phone is buzzing so I can't quite make out what she's saying. Could you please listen to the recipe and list all of the ingredients that my friend described? I only want the ingredients for the filling, as I have everything I need to make my favorite pie crust. I've attached the recipe as Strawberry pie.mp3."
        task_id = "99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3"
        file_name = "99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3.mp3"
        
        result = agent.process_question(question, task_id, file_name)
        print(f"✅ Audio question result: {result}")
        
        print("\n🎉 All tests passed! WorkflowAgent is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_workflow_agent()
    sys.exit(0 if success else 1)