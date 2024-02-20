from langchain import OpenAI, SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import (AgentType,
                              AgentExecutor,
                              create_react_agent, 
                              create_openai_functions_agent,
                              create_sql_agent)
from langchain.agents.initialize import initialize_agent
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
# from langchain.retrievers import RetrievalModel
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain_community.document_loaders.text import TextLoader
from langchain_community.agent_toolkits import SQLDatabaseToolkit

# Document Loaders for PDFs and Docx
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.document_loaders.word_document import Docx2txtLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chat_models.openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter

import langchain
import psycopg2
import os
import constants
import constants_allen
os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY

import warnings
warnings.filterwarnings('ignore')


# Setup database connection
# Setup database
db = SQLDatabase.from_uri(
    f"postgresql+psycopg2://postgres:{constants.DBPASS}@localhost:5433/{constants.DB}",
)

# Setup conversation memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Setup llm
llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")

# Create a list of documents from all our files in the ./data/docs folder
documents = []
for file in os.listdir("./data/docs"):
    if file.endswith(".pdf"):
        pdf_path = "./data/docs/" + file
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    elif file.endswith("./docx") or file.endswith(".doc"):
        doc_path = "./data/docs/" + file
        loader = Docx2txtLoader(doc_path)
        documents.extend(loader.load())
    elif file.endswith(".txt"):
        text_path = "./data/docs/" + file
        loader = TextLoader(text_path)
        documents.extend(loader.load())

# Split the documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
documents = text_splitter.split_documents(documents)

# Convert the document chunks to embedding and save them to the vector store
vectordb = Chroma.from_documents(documents, embedding=OpenAIEmbeddings(), persist_directory="./data/docs")
vectordb.persist()

# Setup prompt template
_DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer. Unless the user specifies in his question a specific number of examples he wishes to obtain, always limit your query to at most {top_k} results. You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.
Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

Only use the following tables:

{table_info}

Question: {input}
"""

# TEMPLATE = '''"Given your input question, provide the answer from the most relevant source. You can choose between two sources:

# 1. Text File Source: This source contains Q&A data on data structures and algorithms.

# 2. Database Source: This source includes information from three tables:
#     - 'developers' table: Contains details about developers, such as full_name, email, phone, position, and department.
#     - 'tasks' table: Holds information about developers' tasks, including the task, completion status, due date, completion date, and priority.
#     - 'insurance_data' table: Contains information about PRU Life Insurance and USA Insurance, including questions and answers.

# Only query the content of the sources, not the metadata.
# You have access to the following tools:

# {tools}

# Use the following format:

# Question: the input question you must answer
# Thought: you should always think about what to do
# Action: the action to take, should be one of [{tool_names}]
# Action Input: the input to the action
# Observation: the result of the action
# ... (this Thought/Action/Action Input/Observation can repeat N times)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question

# Begin!

# Question: {input}
# Thought:{agent_scratchpad}'''

PROMPT = PromptTemplate.from_template(_DEFAULT_TEMPLATE)

# Setup db chain
db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)
toolkit = SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0, verbose=True, model_name="gpt-3.5-turbo"))

# Create Document Q&A chain
pdf_qa = ConversationalRetrievalChain.from_llm(
    OpenAI(temperate=0.7, model_name="gpt-3.5-turbo"),
    retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
    return_source_documents=True,
    verbose=False
)

# Setup text data source
directory = "./data/text/final1.txt"
loader = TextLoader(directory)
index_creator = VectorstoreIndexCreator()
index = index_creator.from_loaders([loader])
retriever_text = index.vectorstore.as_retriever()

# Define the tools needed
tools = [
    Tool(name="MyDocs", func=pdf_qa, description="Retrieve information from documents."),
    Tool(name="MyDB", func=db_chain.run, description="Query the PostgreSQL database",),
    Tool(name="MyRetriever", func=retriever_text.get_relevant_documents, description="Retrieve documents from text file"),
]

agent = initialize_agent(
    tools, llm, memory=memory,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True, 
    output_key="result",
    early_stopping_method="generate",
    max_iterations=3,
)

# agent = create_sql_agent(
#     llm=OpenAI(temperature=0),
#     toolkit=toolkit,
#     verbose=True,
#     agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
# )

# agent = create_react_agent(
#     llm=llm,
#     tools=tools,
#     prompt=PROMPT
# )

# agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory,
#                                verbose=True, handle_parsing_errors=True)

yellow = "\033[0;33m"
green = "\033[0;32m"
white = "\033[0;39m"

def chat():
    while True:
        user_input = input("You: ") # Get user input

        # Use the agent to generate a response
        try:
            bot_response = agent.run(user_input)
            # bot_response = agent_executor.invoke({"input": user_input})
        except ValueError as e:
            bot_response = str(e)
            if not bot_response.startswith("Could not parse LLM output: `"):
                raise e
            bot_response = bot_response.removeprefix("Could not parse LLM output: `").removesuffix("`")

        print("Chatbot: " + bot_response)

        if user_input.lower() == 'exit':
            break

if __name__ == "__main__":
    chat()