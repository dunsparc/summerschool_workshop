from data.milvus.indexing import MilvusIndexer
import os
from llm.base import AgentClient
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider

from data.cache.memory_handler import MessageMemoryHandler

import chainlit as cl

from utils.basetools import *
from bloom_tool import agentic_post_tool
provider = GoogleGLAProvider(api_key=os.getenv("GEMINI_API_KEY"))
model = GeminiModel('gemini-2.0-flash', provider=provider)

tools = [
    {
        "name": "agentic_post_tool",
        "description": "Detects trigger phrases and sends a POST request to a fixed API if the trigger is present.",
        "parameters": UserInput.model_json_schema(),
        "function": agentic_post_tool,
    }
]

agent = AgentClient(
   model=model,
   system_prompt=(
        "You are a friendly virtual assistant. "
        "When users mention starting workflows, submitting reports, or sending notifications, "
        "use the agentic_post_tool."
    ),
    tools=tools
).create_agent()

@cl.on_message
async def main(message: cl.Message):   
   # YOUR LOGIC HERE
   response = await agent.run((message.content))
   await cl.Message(content=str(response.output)).send()