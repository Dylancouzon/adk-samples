# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility for retrieving prompts from Phoenix API"""

import os
import dotenv
from phoenix.client import Client

# Load environment variables
dotenv.load_dotenv()
os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={os.getenv('PHOENIX_API_KEY')}"

# Initialize Phoenix client
client = Client()

def get_prompt(prompt_name: str) -> str:
    """
    Retrieve a prompt from Phoenix API by name
    
    Args:
        prompt_name: Name of the prompt to retrieve
        
    Returns:
        The prompt content as a string
    """
    prompt = client.prompts.get(prompt_identifier=prompt_name, tag="production")
    content = prompt._template['messages'][0]['content']
    
    # Handle different prompt formats
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Handle format like [{'type': 'text', 'text': 'content'}]
        text_parts = []
        for item in content:
            if isinstance(item, dict) and 'text' in item:
                text_parts.append(item['text'])
            elif isinstance(item, dict) and 'content' in item:
                text_parts.append(item['content'])
        return '\n'.join(text_parts) if text_parts else str(content)
    else:
        # Fallback to string conversion
        return str(content)


def verify_prompts():
    """
    Verification script to test all financial advisor prompts
    """
    prompt_names = [
        "financial-advisor-financial-coordinator",
        "financial-advisor-data-analyst", 
        "financial-advisor-trading-analyst",
        "financial-advisor-execution-analyst",
        "financial-advisor-risk-analyst"
    ]
    
    print("=== Phoenix Prompt Verification ===")
    for prompt_name in prompt_names:
        try:
            content = get_prompt(prompt_name)
            content_type = type(content).__name__
            content_length = len(content) if content else 0
            preview = (content[:100] + "...") if len(content) > 100 else content
            
            print(f"\n✓ {prompt_name}:")
            print(f"  Type: {content_type}")
            print(f"  Length: {content_length} characters") 
            print(f"  Preview: {repr(preview)}")
            
        except Exception as e:
            print(f"\n✗ {prompt_name}: ERROR - {str(e)}")
    
    print("\n=== Verification Complete ===")


if __name__ == "__main__":
    verify_prompts()
