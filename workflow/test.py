from data.milvus.indexing import MilvusIndexer
import os
from llm.base import AgentClient
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from pydantic import BaseModel, Field
from data.cache.memory_handler import MessageMemoryHandler

import chainlit as cl
from utils.basetools import *

import requests
provider = GoogleGLAProvider(api_key=os.getenv("GEMINI_API_KEY"))
model = GeminiModel('gemini-2.0-flash', provider=provider)
# This is the function (tool) that makes the POST request

def bloom_tool(user_input: str) -> str:
    """
    Calls an external API when an essay is mentioned.
    """
    api_url = "https://bloom-bert-api-dmkyqqzsta-as.a.run.app/predict"  # ✅ Replace this with your actual working URL

    headers = {
        "Content-Type": "application/json",
        # "Authorization": "Bearer YOUR_API_KEY"  # Uncomment and update if your API needs it
    }

    payload = {
        "text": user_input  # ✅ Match this key to what your API expects
    }

    print("\n📡 Sending request to:", api_url)
    print("📝 Payload:", payload)
    print("📨 Headers:", headers)

    try:
        response = requests.post(api_url, json=payload, headers=headers)
        print("📬 Response status code:", response.status_code)
        print("🧾 Response text:", response.text)

        response.raise_for_status()  # Raises HTTPError for bad status
        try:
            json_data = response.json()
            print("✅ Parsed JSON:", json_data)
            return f"API Response: {json_data}"
        except Exception as e:
            print("❌ Failed to parse JSON:", e)
            return f"Error: Failed to parse API response: {e}"

    except requests.exceptions.RequestException as e:
        print("❌ Network/API error:", e)
        return f"Error: Failed to reach API: {e}"


def web_search_tool(user_input: str) -> str:
    """
    Performs a web search using DuckDuckGo and returns a list of results.
    """
    try:
        search_input = SearchInput(query=user_input, max_results=3)
        search_output = search_web(search_input)

        if not search_output.results:
            return "No search results found."

        # Format output for user
        formatted_results = "\n".join(
            [f"- {item['title']} ({item['link']})" for item in search_output.results]
        )
        return f"🔎 Here are the top results for your query:\n{formatted_results}"

    except Exception as e:
        return f"Error during search: {e}"

class BloomSearchInput(BaseModel):
    query: str = Field(..., description="The academic question or topic to analyze and search.")

def bloom_question_search(input: BloomSearchInput) -> str:
    user_input = input.query

    # 1. Call bloom_tool
    bloom_result_json = bloom_tool(user_input)
    import json
    bloom_data = json.loads(bloom_result_json) if isinstance(bloom_result_json, str) else bloom_result_json
    bloom_level = bloom_data.get("level", "Unknown")

    # 2. Question generator (optional)
    #question = question_tool(bloom_level) if callable(question_tool) else "No sample question."

    # 3. Web search
    search_results = web_search_tool(user_input)

    # 4. Compose final response
    return (
        f"🌱 Bloom's Taxonomy Level: **{bloom_level}**\n\n"
        f"💡 Example Question: {question}\n\n"
        f"{search_results}"
    )



# Create the agent
agent = AgentClient(
    model=model,  # or whichever model you prefer
    system_prompt="""
You are a helpful assistant.
Use the 'bloom_question_search' to find relevant sources to the question from the user
""",
    tools=[bloom_question_search]
).create_agent()

@cl.on_chat_start
async def start():
    """Initialize chat session"""
    cl.user_session.set("message_count", 0)
    await cl.Message(content="welcome to the bloom rank").send()


@cl.on_message
async def main(message: cl.Message):   
   # YOUR LOGIC HERE
   response = await agent.run((message.content))
   await cl.Message(content=str(response.output)).send()