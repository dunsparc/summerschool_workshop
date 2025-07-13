from data.milvus.indexing import MilvusIndexer
import os
from llm.base import AgentClient
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider

from data.cache.memory_handler import MessageMemoryHandler

import chainlit as cl

from utils.basetools import *

provider = GoogleGLAProvider(api_key=os.getenv("GEMINI_API_KEY"))
model = GeminiModel('gemini-2.0-flash', provider=provider)
faq2=create_faq2_tool(collection_name="company1")
agent = AgentClient(
   model=model,
   system_prompt="You are a friendly virtual assistant.Your task is to greet users in a warm, polite, and welcoming way. ",  # Replace with your system prompt
   tools = [faq2]
).create_agent()

@cl.on_message
async def main(message: cl.Message):   
   # YOUR LOGIC HERE
   response = await agent.run((message.content))
   await cl.Message(content=str(response.output)).send()