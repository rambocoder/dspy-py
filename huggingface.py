import os
import json

from dotenv import load_dotenv
load_dotenv()
from huggingface_hub import InferenceClient

# client = InferenceClient("meta-llama/Llama-3.1-8B-Instruct") # or "HuggingFaceH4/zephyr-7b-beta"
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta", token=os.environ["HF_KEY"])

messages = [
    {
        "role": "system",
        "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.",
    },
    {
        "role": "user",
        "content": "What's the weather like in Kyiv in Celsius?",
    },
]
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                },
                "required": ["location", "format"],
            },
        },
    }
]

# output = client.chat_completion(messages=messages, tools=tools, max_tokens=500, temperature=0.3)
# print(output)
tool_call = {"name": "get_current_weather", "arguments": "{\"location\": \"Kyiv\", \"format\": \"celsius\"}"}
messages.append({"role": "assistant", "tool_calls": [{"type": "function", "function": tool_call}], "content": "22.0"})
messages.append({"role": "tool", "name": "get_current_weather", "content": "22.0"})

output = client.chat_completion(messages=messages, tools=tools, max_tokens=500, temperature=0.3)
print(output)
