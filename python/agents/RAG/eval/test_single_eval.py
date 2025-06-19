#!/usr/bin/env python3
"""
Quick test to validate Vertex AI evaluation API integration.
Run this to test a single evaluation before running the full experiment.
"""

import os
import sys
import json
import pandas as pd
from dotenv import load_dotenv

# Add the parent directory to the Python path so we can import from rag
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import evaluation functions (now from same directory)
from test_eval_arize import (
    create_reference_trajectory, 
    evaluate_with_vertex_ai_single_metric,
    trajectory_exact_match_evaluator,
    trajectory_precision_evaluator,
    tool_name_match_evaluator
)

load_dotenv()

def test_vertex_ai_evaluation():
    """Test the Vertex AI evaluation API with sample data."""
    print("🔬 Testing Vertex AI Evaluation API Integration...")
    
    # Sample data mimicking the conversation test format
    sample_expected_tool_use = [
        {
            "tool_name": "retrieve_rag_documentation",
            "tool_input": {
                "query": "How does the growth of non-advertising revenue affect Alphabet's margins?"
            }
        }
    ]
    
    sample_actual_tool_calls = [
        {
            "tool_name": "retrieve_rag_documentation",
            "tool_input": "According to the report, revenues from cloud, consumer subscriptions..."
        }
    ]
    
    sample_response = "The report states that revenues from cloud, consumer subscriptions, platforms, and devices are increasing. It notes that the margins on these revenues vary significantly and are generally lower than advertising margins."
    
    sample_reference = "The report states that revenues from cloud, consumer subscriptions, platforms, and devices are increasing. It notes that the margins on these revenues vary significantly and are generally lower than advertising margins. Specifically, device sales adversely affect consolidated margins due to pricing pressure and higher cost of sales."
    
    print("📊 Testing trajectory conversion...")
    reference_trajectory = create_reference_trajectory(sample_expected_tool_use)
    print(f"✅ Reference trajectory: {reference_trajectory}")
    
    print("📊 Testing Vertex AI evaluation (exact match)...")
    results_exact = evaluate_with_vertex_ai_single_metric(
        predicted_trajectory=sample_actual_tool_calls,
        reference_trajectory=reference_trajectory,
        metric="trajectory_exact_match"
    )
    print(f"✅ Exact match results: {results_exact}")
    
    print("📊 Testing Vertex AI evaluation (precision)...")
    results_precision = evaluate_with_vertex_ai_single_metric(
        predicted_trajectory=sample_actual_tool_calls,
        reference_trajectory=reference_trajectory,
        metric="trajectory_precision"
    )
    print(f"✅ Precision results: {results_precision}")
    
    # Test the full evaluator functions
    print("📊 Testing individual evaluator functions...")
    
    # Create sample output format that matches what the task function returns
    sample_output = json.dumps({
        "agent_response": sample_response,
        "tool_calls": sample_actual_tool_calls,
        "expected_tool_use": sample_expected_tool_use,
        "reference": sample_reference
    })
    
    sample_dataset_row = {
        "input": {
            "query": "According to the MD&A, how might the increasing proportion of revenues derived from non-advertising sources like Google Cloud and devices potentially impact Alphabet's overall operating margin, and why?",
            "expected_tool_use": sample_expected_tool_use,
            "reference": sample_reference
        },
        "output": sample_reference
    }
    
    exact_match_result = trajectory_exact_match_evaluator(sample_output, sample_dataset_row)
    print(f"✅ Exact match evaluator result: {exact_match_result}")
    
    precision_result = trajectory_precision_evaluator(sample_output, sample_dataset_row)
    print(f"✅ Precision evaluator result: {precision_result}")
    
    tool_name_result = tool_name_match_evaluator(sample_output, sample_dataset_row)
    print(f"✅ Tool name match evaluator result: {tool_name_result}")
    
    return True

def main():
    """Run the test."""
    print("🚀 Testing Vertex AI Evaluation Integration...")
    print("=" * 50)
    
    try:
        success = test_vertex_ai_evaluation()
        if success:
            print("=" * 50)
            print("🎉 Vertex AI evaluation integration test passed!")
            print("Ready to run the full evaluation experiment.")
        else:
            print("❌ Test failed")
            return 1
            
    except Exception as e:
        print(f"❌ Test crashed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 