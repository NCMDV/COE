from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import PostgresChatMessageHistory
# from langchain_community.chat_message_histories import RedisChatMessageHistory, PostgresChatMessageHistory
from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
from langchain_openai import OpenAI

import constants
import constants_allen
import os
import psycopg2

os.environ['OPENAI_API_KEY'] = constants.OPENAI_API_KEY
os.environ['GOOGLE_API_KEY'] = constants.GOOGLE_API_KEY
os.environ['GOOGLE_CSE_ID'] = constants.GOOGLE_CSE_ID

# Set up search wrapper
search = GoogleSearchAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    )
]

prefix = """Have a conversation with a human, answering the following questions as best as you can. You have access to the following tools:"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpag}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)

message_history = PostgresChatMessageHistory(
    connection_string=f"postgresql://{constants_allen.DB_CONFIG['user']}:{constants_allen.DB_CONFIG['password']}@{constants_allen.DB_CONFIG['host']}:{constants_allen.DB_CONFIG['port']}/{constants_allen.DB_CONFIG['dbname']}",
    session_id="my-session"
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    chat_memory=message_history
)

# Construct LLMChain with Memory object and then create the agent
llm_chain = LLMChain(llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=constants.OPENAI_API_KEY))
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory
)

def get_prompt():
    print("Type 'exit' to quit")

    while True:
        prompt = input("\nEnter a prompt: ")

        if prompt.lower() == 'exit':
            print("Exiting...")
            break
        else:
            try:
                # question = QUERY.format(question=prompt)
                print(agent_chain.run(prompt))
            except Exception as e:
                print(e)

get_prompt()