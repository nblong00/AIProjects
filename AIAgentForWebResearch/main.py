from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    output: str
    sources: list[str]
    tools_used: list[str]

llm = ChatAnthropic(model='claude-sonnet-4-5')
parser = PydanticOutputParser(pydantic_object=ResearchResponse)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text.
            Always save output to file.
            Always include sources and tools used in saved output.\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input('What can I help you research? ')
raw_response = agent_executor.invoke({'query': query})

try:
    structured_response = parser.parse(raw_response.get('output')[0]['text'])
except Exception as e:
    print('Error parsing response', e, 'Raw Respense - ', raw_response)

print(structured_response.output)
# Move the below into utils.py to include in output response.
print(structured_response.tools_used)
print(structured_response.sources)