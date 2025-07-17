#!/usr/bin/env python3
"""
WorkflowAgent StateGraph Visualization using LangGraph's built-in tools
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.workflow_agent import WorkflowAgent


def visualize_workflow():
    """Use LangGraph's built-in visualization capabilities"""
    print("🎨 WorkflowAgent StateGraph Visualization")
    print("=" * 50)

    try:
        # Initialize the agent
        print("1. Initializing WorkflowAgent...")
        agent = WorkflowAgent()
        graph = agent.graph

        # Method 1: Print graph structure
        print("\n2. Graph Structure:")
        print("=" * 30)
        print(f"   Nodes: {list(graph.nodes.keys())}")

        # Method 2: Generate Mermaid diagram using LangGraph
        print("\n3. Mermaid Diagram (LangGraph Built-in):")
        print("=" * 30)
        try:
            drawable_graph = graph.get_graph()
            mermaid_code = drawable_graph.draw_mermaid()
            print(mermaid_code)

            # Save mermaid code to file for easy access
            with open("workflow_graph.mmd", "w") as f:
                f.write(mermaid_code)
            print(f"\n   ✅ Mermaid code saved to 'workflow_graph.mmd'")
            print(f"   💡 Copy the code above and paste into https://mermaid.live/")

        except Exception as e:
            print(f"   Mermaid generation error: {e}")

        # Method 3: Try to save as image (if graphviz is available)
        print("\n4. Image Generation:")
        print("=" * 30)
        try:
            # Check if pygraphviz is available
            try:
                import pygraphviz

                print("   ✅ pygraphviz is available")
            except ImportError:
                print("   ❌ pygraphviz not installed: uv add pygraphviz")
                return

            drawable_graph = graph.get_graph()
            print(f"   Graph type: {type(drawable_graph)}")

            # Try different methods to generate PNG
            try:
                # Method 1: LangGraph's draw_png method
                if hasattr(drawable_graph, "draw_png"):
                    print("   📄 Using LangGraph's draw_png() method...")
                    png_data = drawable_graph.draw_png()
                    with open("workflow_graph.png", "wb") as f:
                        f.write(png_data)
                    print("   ✅ PNG saved as 'workflow_graph.png'")

                # Method 2: LangGraph's draw_mermaid_png method
                elif hasattr(drawable_graph, "draw_mermaid_png"):
                    print("   📄 Using LangGraph's draw_mermaid_png() method...")
                    png_data = drawable_graph.draw_mermaid_png()
                    with open("workflow_graph_mermaid.png", "wb") as f:
                        f.write(png_data)
                    print("   ✅ Mermaid PNG saved as 'workflow_graph_mermaid.png'")

                # Method 3: Direct draw method (fallback)
                elif hasattr(drawable_graph, "draw"):
                    print("   📄 Using draw() method...")
                    png_data = drawable_graph.draw(format="png")
                    with open("workflow_graph.png", "wb") as f:
                        f.write(png_data)
                    print("   ✅ PNG saved as 'workflow_graph.png'")

                else:
                    print(
                        f"   ❌ No PNG generation methods available on {type(drawable_graph)}"
                    )
                    print(
                        f"   Available methods: {[attr for attr in dir(drawable_graph) if not attr.startswith('_')]}"
                    )

            except Exception as e:
                print(f"   ❌ PNG generation failed: {str(e)}")
                print(f"   Exception type: {type(e)}")
                import traceback

                traceback.print_exc()

        except Exception as e:
            print(f"   ❌ Image generation error: {e}")

        # Method 4: Show workflow details
        print("\n5. Workflow Details:")
        print("=" * 30)
        show_workflow_details()

        print("\n✅ Visualization complete!")
        print("\n📋 Files generated:")
        print("   - workflow_graph.mmd (mermaid code)")
        if os.path.exists("workflow_graph.png"):
            print("   - workflow_graph.png (image)")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


def show_workflow_details():
    """Show detailed workflow information"""
    details = """
    Node Details:
    ├── classify_question: Analyze question type (file/math/youtube/factual/reasoning)
    ├── handle_file: Fetch and process files if file_name exists
    ├── execute_tools: Run selected tools with limits (max 3 calls)
    ├── synthesize_answer: Generate concise answer from context
    └── verify_gaia_format: Ensure GAIA compliance
    
    Decision Points:
    ├── file_name exists? → handle_file OR execute_tools
    └── should_continue? → execute_tools OR synthesize_answer
    
    Termination Conditions:
    ├── Max 3 total tool calls
    ├── Max 5 nodes traversed
    └── No more useful tools available
    
    Tool Selection Logic:
    ├── Math questions → execute_python_code
    ├── YouTube questions → analyze_youtube_video
    ├── File questions → analyze_image/spreadsheet/audio
    └── Default → web_search (max 2 calls)
    """
    print(details)


if __name__ == "__main__":
    visualize_workflow()
