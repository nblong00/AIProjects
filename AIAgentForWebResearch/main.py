from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from tools import search_tool, wiki_tool, save_tool
from utils import clean_api_text

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    output: str
    sources: list[str]
    tools_used: list[str]

active_tools = [search_tool, wiki_tool, save_tool]
model = init_chat_model('anthropic:claude-sonnet-4-5-20250929')


agent = create_agent(
    model=model,
    tools=active_tools,
    system_prompt="""
        You are a research assistant that will help generate a research paper.
        Answer the user query and use necessary tools.
        Always save output to a log file.
        Always include sources and tools used in saved output and what the sources and tools were used for.
        Wrap your final answer in this JSON format exactly as shown (without ```json at the beginning or ``` at the end):
        {
          "topic": "<topic>",
          "output": "<full formatted research output>",
          "sources": ["<source1>", ...],
          "tools_used": ["<tool1>", ...]
        }
        Do not print your final answer to the user. I will handle it based on the formatting of your final answer.
    """,
)

query = input('What can I help you research? ')
response = agent.invoke({"messages": [("human", query)]})
# print(response)

raw_output = response["messages"][-1].content
try:
    structured_response = ResearchResponse.model_validate_json(raw_output)
except Exception as e:
    print("Error parsing response:", e)
    print("Raw response:", raw_output)

print(f'\n{structured_response.topic}\n')
print(clean_api_text(structured_response.output))
print('\nTools used:', structured_response.tools_used)
print('\nSources:', structured_response.sources)
