from langchain import SQLDatabase
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
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain_community.document_loaders.text import TextLoader
from langchain_community.agent_toolkits import SQLDatabaseToolkit


#for local data
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain_community.callbacks import get_openai_callback
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain_community.document_loaders import (
                                        Docx2txtLoader, 
                                        PyPDFLoader, 
                                        TextLoader,
                                        S3FileLoader,
                                        S3DirectoryLoader)


from io import BytesIO
from PyPDF2 import PdfReader

import os
import json
import constants

os.environ['OPENAI_API_KEY'] = constants.OPENAI_API_KEY
os.environ['AWS_ACCESS_KEY_ID'] = constants.AWS_ACCESS_KEY
os.environ['AWS_SECRET_ACCESS_KEY'] = constants.AWS_SECRET_ACCESS_KEY

import boto3

s3 = boto3.resource('s3')
s3_client = boto3.client('s3')
bucket_name = "77gsi-insurance-data"

# Setup conversation memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Setup llm
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-1106")

# Define prompt template
PROMPT = PromptTemplate(
        input_variables=[
            "chat_history",
            # "table_info",
            "question",
        ],
        template=("""
                    You are a helpful AI assistant specializing in Insurance.
                    You will determine the best possible answer based on the context of the user's question.
                    You will answer on a friendly tone.
                    If there are questions not related to insurance, reply appropriately.
                                    
                    Here is the chat history so far, you may use it to answer subsequent questions:

                    {chat_history}

                    Question: {question}
                    """)
    )


# Helper function for getting all file names from S3 Bucket
def get_all_files(bucket_name):
    bucket = s3.Bucket(bucket_name)

    files = []
    for s3_obj in bucket.objects.all():
        files.append(s3_obj.key)
    
    return files

# Function for loading files from S3 Bucket
def load_files_from_s3():
    """Load documents from S3"""
    documents = []
    bucket_name = "77gsi-insurance-data"
    file_names = get_all_files("77gsi-insurance-data")

    for file in file_names:
        if file.endswith(".pdf"):
            # obj = s3.Object(bucket_name, file)
            # pdf = obj.get()['Body'].read()
            # loader = PyPDFLoader(BytesIO(pdf))
            loader = S3FileLoader(bucket_name, file)
            documents.extend(loader.load())
            # print("".join([s for s in documents if isinstance(s, list) f]))
        elif file.endswith(".docx") or file.endswith(".doc"):
            # obj = s3.Object(bucket_name, file)
            # doc = obj.get()["Body"].read()
            # loader = Docx2txtLoader(BytesIO(doc))
            loader = S3FileLoader(bucket_name, file)
            documents.extend(loader.load())
        elif file.endswith(".txt"):
            # obj = s3.Object(bucket_name, file)
            # txt = obj.get()["Body"].read()
            # loader = TextLoader(BytesIO(txt))
            loader = S3FileLoader(bucket_name, file)
            documents.extend(loader.load())
        
    return documents

documents = load_files_from_s3()
print(documents)

# Split the documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=50)
documents = text_splitter.split_documents(documents)

# # Convert the document chunks to embedding and save them to the vector store
# vectordb = Chroma.from_documents(documents, embedding=OpenAIEmbeddings(), persist_directory="./data/docs")
# vectordb.persist()

# For embeddings
embeddings = OpenAIEmbeddings()
# For text searching
vector = FAISS.from_documents(documents, embeddings)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
# chain = load_qa_chain(llm, chain_type="stuff")
print(vector.as_retriever(ssearch_type="similarity_score_threshold", 
                                 search_kwargs={"score_threshold": .5, 
                                                "k": 6}))
chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo-1106"),
    # retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
    # retriever=vector.as_retriever(search_type="similarity_score_threshold", 
    #                              search_kwargs={"score_threshold": .5, 
    #                                             "k": 6}),
    retriever=vector.as_retriever(search_kwargs={"k": 6}),
    return_source_documents=True,
    verbose=False

)


def model_response(user_prompt, chat_history):
        docs = vector.similarity_search(user_prompt)
        response = chain.invoke({'input_documents': docs, 'question': user_prompt, 'chat_history': chat_history}, return_only_outputs=True)
        
        return response['output_text']

def save_conversation(chat_history):
    convo = []

    if os.path.exists("chat_history.json"):
        with open('chat_history.json', 'r+') as f:
            try:
                data = json.load(f)
                convo.extend(data)

                for i in range(len(chat_history)):
                    convo.append({"user": chat_history[i][0], "bot": chat_history[i][1]})
            except Exception as e:
                return str(e)
        
        with open("chat_history.json", "w") as js:
            json.dump(convo, js)
    else:
        for i in range(len(chat_history)):
            convo.append({"user": chat_history[i][0], "bot": chat_history[i][1]})
        
        with open("chat_history.json", "w") as js:
            json.dump(convo, js)

def load_conversation():
    chat_history = []
    if os.path.exists("chat_history.json"):
        saved_history = json.load(open("chat_history.json"))
        
        for i in range(len(saved_history)):
            chat_history.append((saved_history[i]['user'], saved_history[i]['bot']))
    else:
        chat_history = []

    return chat_history


def chat():
    
    chat_history = load_conversation()
    print(chat_history)
    while True:
        user_input = input("You (`bye` to end): ") # Get user input
        if user_input.lower() == 'bye':
            save_conversation(chat_history)
            break

        # Use the agent to generate a response
        try:
            # bot_response = agent.run(user_input)
            # bot_response = agent_executor.invoke({"input": user_input})
            bot_response = chain({'question': user_input, 'chat_history': chat_history})
            bot_response = bot_response['answer']
            # bot_response = model_response(user_input, chat_history)
        except ValueError as e:
            bot_response = str(e)
            if not bot_response.startswith("Could not parse LLM output: `"):
                raise e
            bot_response = bot_response.removeprefix("Could not parse LLM output: `").removesuffix("`")

        print("Chatbot: " + bot_response)

        chat_history.append((user_input, bot_response))

    

chat()