import re
import psycopg2
import json
import spacy
import random
import time
from django.shortcuts import render, redirect
from django.conf import settings
from django.http import JsonResponse, HttpResponse
from langchain .agents import create_sql_agent
from langchain_community.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_experimental.sql import SQLDatabaseChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import get_openai_callback
from spacy.cli.download import download
from langchain.prompts.chat import ChatPromptTemplate
import requests
from dashboard_func.intent_classification import predict_intent
from django.utils import timezone
from .forms import DateFilterForm

import subprocess as st
import sys
import os
from dotenv import load_dotenv

# print("Before subprocess.run")
# gdrive = 'download_from_drive.py'
#sharepoint = 'sharepoint_dl.py'
#python_executable = sys.executable  # Get the path to the current Python interpreter
# print(python_executable)
#st.run([python_executable, gdrive])
#st.run([python_executable, sharepoint])



#for api directory
import sys
api_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
api_directory = os.path.join(api_parent, 'api')
sys.path.append(api_directory)
import auth
from datetime import timedelta, datetime as dt


#for online offline status
from django.views.decorators.http import require_GET
from django.views.decorators.csrf import csrf_exempt

#for local data
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader, CSVLoader

#for gettting data to cloud
import subprocess
#script_path = './path.py'
#subprocess.run(['python', script_path])

#for aiassistant
# from assistant import AIAssistant
# from sql_assistant import GetDBSchema, RunSQLQuery

# assistant = AIAssistant(
#     instruction="""
# You are a SQL expert. User asks you questions about the database.
# First obtain the schema of the database to check the tables and columns, then generate SQL queries to answer the questions.
# """,
# model="gpt-3.5-turbo-0613",
# functions=[GetDBSchema(), RunSQLQuery()],
# use_code_interpreter=True,
# )

load_dotenv()

#OPENAI API Connection
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY

def get_chatbox_status(request):
    chatbox_status = True 
    return JsonResponse({'chatbox_status': chatbox_status})


# library realtime currency exchange rate
from forex_python.converter import CurrencyRates

#imports for chatlogs
import datetime
import uuid
from .models import ChatlogsTest
import pandas as pd

# chatlog variables that change in different views
chatlog_variables = {"last_logged_time":"", "id":"", "intent":""}

#PostgreSQL Database Connection
#gagamitin to sa psycopg2
#kapag public schema ang gagamitin, eto yung connection na gamitin nyo
connection_uri = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"

# PostgreSQL Database Connection with specified schema
#gagamitin to para sa create_sql_agent or SQLDatabaseChain
#kapag gpt_data schema ang gagamitin, eto yung connection na gamitin nyo
#connection = f"postgresql://{constants.DB_CONFIG['user']}:{constants.DB_CONFIG['password']}@{constants.DB_CONFIG['host']}:{constants.DB_CONFIG['port']}/{constants.DB_CONFIG['dbname']}?options=-csearch_path=gpt_data"

db = SQLDatabase.from_uri(connection_uri,
                          schema='gpt_data',
                          include_tables=['qna','it_qas_data','test_table'])

#LLM Setup
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, verbose=True)

#Load the spaCy English language model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

#Create DB Chain

QUERY = """
Given an input question, first create a syntactically correct postgresql query to run, then look at the results of the query and return the answer.
Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: 

{question}
"""


#Define a prompt
prompt = PromptTemplate(
    input_variables=["user_message", "db_response"],
    template=QUERY
)



#Test Prompt
test_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """
         Your name is ALiCIA, short for Automated Live Chatbot Insurance Agent built by 77Global.
         You are a helpful chatbot
         You will determine the best possible answer based on the context of the user's question.
         You will answer on a friendly tone.
         """),
        ("user", "{question}\n ai: ")
    ]
)

#Create a memory object to store conversation history
memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=True,
)

#Setup DB Chain
db_chain = SQLDatabaseChain.from_llm(llm, 
                            db,
                            memory=memory,
                            verbose=True)

#Frontend Views
def chat(request):
    return render(request, 'chat.html')

def we_do_page(request):
    return render(request, 'login.html')

def products_page(request):
    return render(request, 'products.html')

def claims_services_page(request):
    return render(request, 'claims.html')

def work_with_us_page(request):
    return render(request, 'work.html')

def about_us_page(request):
    return render(request, 'about.html')


"""Start of code for using local data"""
documents = []

def process_file(file_path):
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif file_path.endswith('.docx') or file_path.endswith('.doc'):
            loader = Docx2txtLoader(file_path)
            documents.extend(loader.load())
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
        elif file_path.endswith('.csv'):
            loader = CSVLoader(file_path, csv_args={"delimiter": ","})
            documents.extend(loader.load())
            
# Specify the root folder
root_folder = './documents'     
# root_folder = './web_application/documents' #--> if wala sa loob ng web_application yung current directory
#chunk_folder = os.path.join(root_folder, 'chunk_files')

# Create the 'chunk_files' folder if it doesn't exist
#os.makedirs(chunk_folder, exist_ok=True)

# Walk through the directory tree
for folder, _, files in os.walk(root_folder):
    for file in files:
        file_path = os.path.join(folder, file)
        process_file(file_path)

# To split and chunks the loaded documents into smaller token
text_splitter = CharacterTextSplitter(separator = "\n",chunk_size=400, chunk_overlap=50)
docs =  text_splitter.split_documents(documents)

# # Save each chunk as a separate file in the 'chunk_files' folder
# for i, chunk in enumerate(docs):
#     chunk_str = str(chunk)  # Convert the chunk to a string
#     chunk_filename = f"chunk_{i + 1}.txt"
#     chunk_filepath = os.path.join(chunk_folder, chunk_filename)
#     with open(chunk_filepath, 'w', encoding='utf-8') as chunk_file:
#         chunk_file.write(chunk_str)


# For embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
# For text searching
vector = FAISS.from_documents(docs, embeddings)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
#chain = load_qa_chain(llm, chain_type="stuff")

chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo-1106"),
    # retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
    # retriever=vector.as_retriever(search_type="similarity_score_threshold", 
    #                              search_kwargs={"score_threshold": .5, 
    #                                             "k": 6}),
    retriever=vector.as_retriever(search_kwargs={"k": 6}),
    return_source_documents=True,
    verbose=False,

)


def model_response(user_prompt, chat_history):
        docs = vector.similarity_search(user_prompt)
        #response = chain.invoke({'input_documents': docs, 'question': user_prompt}, return_only_outputs=True)
        response = chain.invoke({'input_documents': docs, 'question': user_prompt, 'chat_history': chat_history}, return_only_outputs=True)
        
        return response['output_text']

"""End"""

# Helper function for saving conversation history
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

# Load chat history from json file
def chunk_chat_history(chat_history):
    chunked_chat_history = chat_history
    if len(chat_history) >= 100:
        in_length = len(chat_history) - 50
        chunked_chat_history = chat_history[in_length:len(chat_history)]
        
    return chunked_chat_history

def load_conversation():
    chat_history = []
    if os.path.exists("chat_history.json"):
        saved_history = json.load(open("chat_history.json", encoding='utf-8'))
        
        for i in range(len(saved_history)):
            chat_history.append((saved_history[i]['user'], saved_history[i]['bot']))

        chat_history = chunk_chat_history(chat_history)
    else:
        chat_history = []

    return chat_history


#Query all tables in DB public schema
def get_all_tables():
    with psycopg2.connect(connection_uri) as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'gpt_data';")
            tables = cursor.fetchall()
    return [table[0] for table in tables]


#Setup SQL Agent
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent_executor = create_sql_agent(
            llm=ChatOpenAI(temperature=0),
            toolkit=toolkit,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            all_tables_method=get_all_tables,
            prompt=prompt,
            memory=memory,
            handle_parsing_errors=True,
            top_k=100, #set limit on how many data rows will be queried on the table
            )


#User Input Processing
def transform_sentence(input_sentence):
    patterns = {
        r'\bpru\s*term\s*15\b': 'Pruterm 15',
        r'\bprulife\s*your\s*term\b': 'PRULife Your Term',
        r'\bprushield\b': 'PRUShield',
        r'\bprulove\s*for\s*life\b': 'PRULove for Life'
    }

    for pattern, replacement in patterns.items():
        input_sentence = re.sub(re.compile(pattern, re.IGNORECASE), replacement, input_sentence)

    return input_sentence

# -----------------included in intent classification: PWEDE NA TO TANGGALIN (??) ---------------------
# def is_greeting(message):
#     doc = nlp(message)
#     for token in doc:
#         print(token.text, token.dep_, token.head.text, token.head.pos_, [child for child in token.children])
#         if token.text.lower() in ["hi", "hello", "hey", "greetings", "hey there"]:
#             return True
#     return False


# def is_concern(message):
#     doc = nlp(message)
#     for token in doc:
#         print(token.text, token.dep_, token.head.text, token.head.pos_, [child for child in token.children])
#         if token.text.lower() in ["help", "assist", "assistance", "aid", "have a problem", "problem", "issue", "concern"]:
#             return True
#     return False

# ----------------------------------------------------------------------------------------------------
def refresh_request_coe_variables(request):
    request.session['coe_questions'] = []
    request.session['coe_question_index'] = 0
    request.session['ai_response_index'] = 1
    request.session['coe_purpose'] = ""
    request.session['coe_filing_data'] = {}
    request.session['filing_data_header'] = []
    request.session['filing_data_content'] = []

    #sir Eugene's variables

    request.session['json_question_index']  = 0
    request.session['json_ai_response_index'] = 1
    request.session['json_question']= []
    request.session['answer'] = []
    request.session['rdrct'] = False
    request.session['target_index'] = None


def get_form_element (item_index):
    f = open(os.path.join('./json_files','coe_redirect.json'))
    file = f.read()
    data = json.loads(file)
    elements = data["items"]
    element = elements[item_index]
    return element


def final_ai_response(request):

    question_index = request.session['json_question_index']
    ai_response_index = request.session['json_ai_response_index']
    question = request.session['json_question']
    answer = request.session['answer']
    rdrct = request.session['rdrct']
    target_index = request.session['target_index']


    # #processes for chat logs
    # id = chatlog_variables['id']
    # datetime_now = datetime.datetime.now(tz=timezone.get_current_timezone())
    # # getting warnings about excluding timezones in datetime
    # # tz = timezone.get_current_timezone()
    # # datetime_now = timezone.make_aware(datetime_now, tz, True)
    # lapsed_time = datetime_now - chatlog_variables["last_logged_time"]
    # chatlog_variables["last_logged_time"] = datetime_now

    # log = ChatlogsTest.objects.get(id=id)
    # log.duration += lapsed_time.total_seconds()
    # log.user_typing_time += lapsed_time.total_seconds()
    # log.interaction_count += 1
    
    # chat_history = load_conversation()

    try:
        data = json.loads(request.body)
        user_message = data.get('userMessage', '')
        start_time = time.time()
        
    except json.JSONDecodeError:
        response = {'reply': 'Invalid JSON data'}
        return JsonResponse(response, status=400)
    
    # SPECIFY A FILE
    json_files = [pos_json for pos_json in os.listdir('./json_files/') if pos_json.endswith('.json')]

    # CONVERTING JSON TO QUESTION REQUIRING USER INPUT
    f = open(os.path.join('./json_files','coe_redirect.json'))
    file = f.read()
    data = json.loads(file)

    # Print title of form
    form_title = data["metadata"]["title"]
    
    elements = data["items"]

    if ai_response_index == 1:


        while True:

            #done iterating on the questions
            if len(data["items"]) == (question_index):
                ai_response_index +=1
                break
            
            item_index = get_form_element(question_index)["index"]   #0 1 2 3 4 5 6

            if rdrct == True:
                if item_index == target_index:
                    rdrct = False
                else:
                    question_index +=1
                    continue

            q_type = get_form_element(question_index)["type"]
            title = get_form_element(question_index)["title"]
            
            # LIST is a dropdown
            if q_type == "LIST" or q_type =="MULTIPLE_CHOICE":

                choices = [x["choice"] for x in get_form_element(question_index)["choices"]]           #travel, Credit Card Application
                nav_type = [x["navType"] for x in get_form_element(question_index)["choices"]]         #GO_TO_PAGE, GO_TO_PAGE
                target_page = [x["targetIndex"] for x in get_form_element(question_index)["choices"]]  #1, 5
                choices_lower = [x.lower() for x in choices]

                # choice = None
                # while choice is None:

                if title not in question:
                    response = f"***{title}*** \n Choices:{choices}"
                    question.append(title)
                    break
                    
                else:
                    #nasa choices ung sagot, next question
                    if user_message.lower() in choices_lower:
                        choice = user_message.lower()
                        choice_index = choices_lower.index(user_message.lower())

                        if nav_type[choice_index] == "GO_TO_PAGE":
                            target_index = target_page[choice_index]            #1
                            rdrct = True


                        # choice = user_message.lower()
                        answer.append(choice)
                        question_index += 1
                        continue
                            
                    else:
                        response = f"{user_message} is not in the given choices."
                        break
                

            #checkbox
            elif q_type == "CHECKBOX":
                choices = elements[question_index]["choices"]    
                choices_lower = [x.lower() for x in choices]
                choice_list = []

                if title not in question:
                    response = f"***{title}*** \n\n Choices:{choices} \n\n Please type your desired choice/s separated by comma."
                    question.append(title)
                    break
                
                else:
                    user_answer = user_message
                    answer_list = user_answer.split(",")
                    answer_list_lower = [x.lower().strip() for x in answer_list]
                    diff = set(answer_list_lower).difference(set(choices_lower))

                    if user_message.lower() == 'exit':
                        response = "I should now be back to intent classification."
                        break

                    #nasa choices ung sagot, next question
                    elif len(diff) == 0:
                        answer.append(user_message)
                        question_index += 1
                        continue
                    else:
                        response = f"Apologies, I could not understand that. \n\n This is the question: {title} \n\n Choices:{choices} \n\n Please type your desired choice/s separated by comma."
                        break
            

            elif q_type == "DATE":
                format = "%d-%m-%Y"
                if title not in question:
                    # Check if the variable 'response' is not yet declared
                    if 'response' not in locals() and 'response' not in globals():
                        # If not declared, do something
                        response = f"***{title}*** \n Please type date in 'dd-mm-yyyy' format: "
                    
                    #response is already declared by PAGE-BREAK, hence add the next question
                    else:
                        response += f"\n\n***{title}*** \n Please type date in 'dd-mm-yyyy' format: "

                    question.append(title)
                    break

                else:

                    user_answer = user_message.strip()
                    if user_message.lower() == 'exit':
                        response = "I should now be back to intent classification."
                        break
                    #tama ang sagot sa date
                    try:
                        if bool(dt.strptime(user_answer, format)):
                            answer.append(user_answer)
                            question_index += 1
                            continue
                    except:
                       response = f"{user_message} is not in the valid format. \n\n Please type date in 'dd-mm-yyyy' format for '{title}'"
                       break

            elif q_type == "PAGE_BREAK":
                helptext = get_form_element(question_index)["helpText"]
                response = f"{title} \n {helptext}"
                question_index += 1
                # break

            elif q_type == "TEXT":
                if title not in question:

                    if 'response' not in locals() and 'response' not in globals():
                        # If not declared, do something
                        response = f"***{title}***"
                    
                    #response is already declared by PAGE-BREAK, hence add the next question
                    else:
                        response += f"\n\n***{title}***"
                    question.append(title)
                    break

                #tama na ung sagot
                else:
                    answer.append(user_message.strip())
                    question_index += 1
                    continue


    
    if ai_response_index == 2:
        summary = "Here is the summary of your COE filing request:\n*************************\n"
        coe_final_data = dict(zip(question, answer))
        for key, value in coe_final_data.items():
            summary += f"{key.capitalize()}: {value}\n"
        summary += "*************************\n Do you want to proceed with your COE request?"
        response = summary

    request.session['json_question_index'] = question_index 
    request.session['json_ai_response_index'] = ai_response_index 
    request.session['json_question'] = question 
    request.session['answer'] = answer 
    request.session['rdrct'] = rdrct 
    request.session['target_index'] = target_index 


    return HttpResponse(response.replace('\n', '<br>'))

    # log.transcript += "\n\nUser: {}".format(user_message)
    # log.start_datetime = timezone.make_aware(log.start_datetime, timezone.get_current_timezone(), True)
    # log.save()

    # with get_openai_callback() as cb:

    #     if user_message.lower() == 'exit':
    #         response = "Thank you for using AI Chat!"
        
    #     else:
    #         intent = predict_intent(user_message) 

    #         chatlog_variables['intent'] = intent

    #         if intent == "greeting":
    #             responses = ["Hi! How may I help you?",
    #                             "Hello! Please let me know how I can assist you.",
    #                             "Greetings! Is there anything I can help you with? Please send a message below."]
    #             response = random.choice(responses)

    #         elif intent =="coe":
    #             response = """@ember @chris :), pagawa pls ng buttons sa chatbot at this part ty: 
                
    #             Do you want to request for a Certification of Employment? 
    #             [yes - run coe function; succeeding submit message button clicks point to coe function at backend] 
    #             [no - runs q_and_a function; succeeding submit message button clicks point to final_ai_response function at backend]"""

    #         elif intent == "live_agent":
    #             response = """@ember @chris :), pagawa pls ng buttons sa chatbot at this part ty: 
                
    #             Do you want to talk with a live agent? 
    #             [yes - run live_agent function; succeeding submit message button clicks point to live_agent function at backend] 
    #             [no - runs q_and_a function; succeeding submit message button clicks point to final_ai_response function at backend]"""


    #         # # no need since part of greeting intent
    #         # elif is_concern(user_message):
    #         #     responses = ["Sure, how I can help you?",
    #         #                     "What seems to be the problem? Please tell me how I can assist you.",
    #         #                     "Sure, you may tell me what's on your mind so I can assist you."]
    #         #     response = random.choice(responses)
            
    #         else:
    #     #         # [START] COMMENTED OUT FOR METRICS TESTING -------------------------------
    #     #         user_message = transform_sentence(user_message)
    #     #         # Retrieve or initialize conversation history from the session
    #     #         conversation_history = request.session.get('conversation_history', [])
    #     #         # Add the user message to the conversation history
    #     #         conversation_history.append({"role": "user", "content": user_message})

    #     #         #AI response generation
    #             # try:
    #             #     #response = model_response(user_message)
    #             #     response = chain({'question': user_message, 'chat_history': chat_history})
    #             #     response = response['answer']
    #             #     print("AI Response:", response)
                    
    #             #     #Chat History
    #             #     chat_history.append((user_message, response))
    #             #     save_conversation(chat_history)
    #             # except Exception as e:
    #             #     print("Error generating response: ", str(e))
                                
    #     #         # Add the assistant's reply to the conversation history
    #     #         conversation_history.append({"role": "assistant", "content": response})
                
    #     #         # Update the conversation history in the session
    #     #         request.session['conversation_history'] = conversation_history
                
    #     #         #Print Tokens consumed and Cost
    #     #         print('='*100)
    #     #         print("OpenAI Consumption")
    #     #         print(f"Total Tokens: {cb.total_tokens}")
    #     #         print(f"Prompt Tokens: {cb.prompt_tokens}")               
    #     #         print(f"Completion Tokens: {cb.completion_tokens}")
    #     #         print(f"Total Cost (USD): ${cb.total_cost}")
    #     #         print(f"Total Cost (PHP): PHP{cb.total_cost * round(CurrencyRates().get_rate('USD', 'PHP'), 2)}")
    #     #         print('='*100)
            
    #     # # else:
    #     # #     user_message = transform_sentence(user_message)
    #     # #     #user_message = QUERY.format(question=user_message)
    #     # #     #response = agent_executor.run(user_message) #using create_sql_agent and prompt
    #     # #     response = agent_executor.run(test_prompt.format(question=user_message)) #using create_sql_agent and test_prompt
    #     # #     #response = db_chain.run(user_message) #using SQLDatabaseChain
    #     # #     print('='*100)
    #     # #     print("OpenAI Consumption")
    #     # #     print(f"Total Tokens: {cb.total_tokens}")
    #     # #     print(f"Prompt Tokens: {cb.prompt_tokens}")
    #     # #     print(f"Completion Tokens: {cb.completion_tokens}")
    #     # #     print(f"Total Cost (USD): ${cb.total_cost}")
    #     # #     print(f"Total Cost (PHP): PHP{cb.total_cost * round(CurrencyRates().get_rate('USD', 'PHP'), 2)}")
    #     # #     print('='*100)
            
    #     #     insert_to_database(user_message, response, start_time, connection_uri, cb)
    #     #     # [END] COMMENTED OUT FOR METRICS TESTING -------------------------------
    
    #             time.sleep(1)
    #             response = "This is a dummy response for metrics generation."

    # log = ChatlogsTest.objects.get(id=id)
    # log.transcript += "\n\nChatbot: {}".format(response)

    # datetime_now = datetime.datetime.now(tz=timezone.get_current_timezone())
    # # tz = timezone.get_current_timezone()
    # # datetime_now = timezone.make_aware(datetime_now, tz, True)
    
    # lapsed_time = datetime_now - chatlog_variables["last_logged_time"]
    # chatlog_variables["last_logged_time"] = datetime_now

    # log.duration += lapsed_time.total_seconds()
    # log.chatbot_inference_time += lapsed_time.total_seconds()
    # log.start_datetime = timezone.make_aware(log.start_datetime, timezone.get_current_timezone(), True)
    # log.save()
    # return HttpResponse(response.replace('\n', '<br>'))




# Function to insert data into the database
def insert_to_database(user_message, response, start_time, connection_uri, cb):
    end_time = time.time()
    inference_time = end_time - start_time
    
    total_tokens = cb.total_tokens if cb.total_tokens is not None else 0
    prompt_tokens = cb.prompt_tokens if cb.prompt_tokens is not None else 0
    total_cost_usd = cb.total_cost if cb.total_cost is not None else 0
    
    sql_query = "INSERT INTO public.chatlogs (user_message, ai_response, inference_time, total_tokens, prompt_tokens, total_cost_usd) VALUES (%s, %s, %s, %s, %s, %s);"
    sql_params = [user_message, response, inference_time, total_tokens, prompt_tokens, total_cost_usd]
    
    with psycopg2.connect(connection_uri) as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql_query, sql_params)

        
#Function to create log (if chatbox was opened: create log)
def chatbot_display(request):

    if request.method == 'POST':
        # global eu_question_index
        # global eu_ai_response_index
        # global question
        # global answer
        # eu_question_index = 0
        # eu_ai_response_index = 1
        # question = []
        # answer = []

        refresh_request_coe_variables(request)
        data = json.loads(request.body)
        chatbotDisplay = data.get('chatbotDisplay', '')

        user_id = request.user

        if chatbotDisplay == 1:

            while True:
                id = uuid.uuid4()
                chatlog_variables['id'] = id

                try:
                    chatlog_duplicate = ChatlogsTest.objects.get(id=id)
                    continue

                except:
                    start_datetime = datetime.datetime.now(tz=timezone.get_current_timezone())
                    # tz = timezone.get_current_timezone()
                    # start_datetime = timezone.make_aware(start_datetime, tz, True)
                    transcript= "Start datetime: {}\n".format(start_datetime.strftime("%d %b %Y %H:%M:%S %Z"))
                    
                    log = ChatlogsTest(id = id, user_id=user_id, transcript=transcript, start_datetime=start_datetime)
                    log.save()
                    print("done saving initial row with id", id)

                    chatlog_variables["last_logged_time"] = start_datetime
                    break
        else:
            id = chatlog_variables['id']
            datetime_now = datetime.datetime.now(tz=timezone.get_current_timezone())
            # tz = timezone.get_current_timezone()
            # datetime_now = timezone.make_aware(datetime_now, tz, True)
            lapsed_time = datetime_now - chatlog_variables["last_logged_time"]

            log = ChatlogsTest.objects.get(id=id)
            log.duration += lapsed_time.total_seconds()
            log.transcript += "\n\nEND OF CONVERSATION\n\n\nEnd datetime: {}".format(datetime_now.strftime("%d %b %Y, %H:%M:%S %Z"))
            log.start_datetime = timezone.make_aware(log.start_datetime, timezone.get_current_timezone(), True)
            log.save()

            return render(request, "chat.html")

        return HttpResponse('success')

    return HttpResponse('Fail')


#Function to create log (if chatbox was opened: create log)
def idle_response(request):

    if request.method == 'POST':
        data = json.loads(request.body)
        idle_response = data.get('idle_response', '')
        print(idle_response)
        
        id = chatlog_variables['id']
        print("querying for this id:", id)

        log = ChatlogsTest.objects.get(id=id)
        log.transcript += "\n\nChatbot: {}".format(idle_response)
        log.start_datetime = timezone.make_aware(log.start_datetime, timezone.get_current_timezone(), True)
        log.save()

        return HttpResponse('success')

    return HttpResponse('Fail')

def process_feedback(request):

    if request.method == "POST":
        data = json.loads(request.body)
        user_feedback = data.get('feedbackInputs', '')

        id = chatlog_variables['id']
        log = ChatlogsTest.objects.get(id=id)
        log.user_feedback = user_feedback
        log.start_datetime = timezone.make_aware(log.start_datetime, timezone.get_current_timezone(), True)
        log.save()

        return HttpResponse('success')

    return HttpResponse('Fail')

def dashboard(request):
    
    # if this is a POST request we need to process the form data
    if request.method == "POST":
        # create a form instance and populate it with data from the request:
        form = DateFilterForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            start_date = form.cleaned_data["start_date"]
            end_date = form.cleaned_data["end_date"]

            
            # redirect to a new URL:
            # return HttpResponseRedirect("/dashboard/")
        else:
            start_date = ChatlogsTest.objects.earliest("start_datetime")
            end_date = ChatlogsTest.objects.latest("start_datetime")
        
        # tz = timezone.get_current_timezone()
        # start_date = timezone.make_aware(start_date, tz, True)
        # end_date = timezone.make_aware(end_date, tz, True)
        df = pd.DataFrame.from_records(ChatlogsTest.objects.filter(start_datetime__gte=start_date, start_datetime__lte=end_date + timedelta(days=1)).values())
    # if a GET (or any other method) we'll create a blank form
    else:
        df = pd.DataFrame.from_records(ChatlogsTest.objects.all().values())
    
    form = DateFilterForm()
    labels_h = list(range(24))
    labels_d = ["Sunday","Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    if df.empty:
        htr=0
        ur=0
        ir=0
        data_h = [0] * 24
        data_d = [0] * 7
        uf = 0
    else:
        df['hour'] = df['start_datetime'].apply(lambda x:x.hour)

        #human takeover rate
        htr = round((df[df['escalation'] == 1]['escalation'].count() / df.shape[0] * 100), 2)

        #user retention
        df_duplicates = df[df['user_id'].duplicated()]
        ur = len(df_duplicates['user_id'].unique())

        #interaction rate
        ir = round(df['interaction_count'].mean(),1)

        #hourly conversations data
        data_h = [df[df['hour'] == z].shape[0] for z in labels_h]

        #daily conversation data
        df['day'] = df['start_datetime'].apply(lambda x:x.strftime('%A'))
        # labels_d = ["Sunday","Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        data_d = [df[df['day'] == z].shape[0] for z in labels_d]

        #average user feedback
        uf = round(df[df['user_feedback'] != 0]['user_feedback'].mean(),1)

    return render(request, "dashboard.html", {'labels_h':labels_h, "data_h":data_h, "htr":htr, "ur":ur, "ir":ir, "labels_d":labels_d, "data_d":data_d, "uf":uf, "form": form})


"""for logging in"""

def login(request):
    login_messages=[]

    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        #checks if token is accessed using request_token function, which means user exists
        token_value = request_token(username, password)
        # print("Token Value", token_value)

        if token_value:
            
            if "access_token" in token_value:   #Successfully Logs in if access_token is available
                # place access token after Bearer
                access_token = token_value['access_token']
                refresh_token = token_value['refresh_token']
                refresh_exp = token_value['refresh_exp']

                headers = {
                        'accept': 'application/json',
                        'Authorization': 'Bearer '+ access_token,}

                # access and department type of particular user
                # user = requests.get('http://localhost:8001/user', headers=headers)     
                # response3 = requests.get('http://localhost:8001/dep_one', headers=headers)  #checks if user belongs to department one
                
                #saves what type of user it is
                # request.session['user_info'] = user.json()
                request.session['access_token'] = access_token
                request.session['refresh_token'] = refresh_token
                request.session['refresh_exp'] = int(refresh_exp)
                request.session['user_info'] = auth.get_current_user(f'{access_token}')


                login_messages.append("Logging in Successful")


                # Redirect to the 'chat' path
                return render(request, 'main.html')
        
            elif "detail" in token_value:       
                login_messages.append("Incorrect Credentials.")


        else:
            # print("Failed to obtain tokens.")
            login_messages.append("Error log in. Please try again.")

    # print(login_messages)
    return render(request, 'login.html', {'login_messages': login_messages})




## REQUESTING FOR TOKENS UPON LOGGED IN
def request_token(username, password):
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/x-www-form-urlencoded',
    }
    data = {
    'grant_type': '',
    'username': username,
    'password': password,
    'scope': '',
    'client_id': '',
    'client_secret': ''
    }

    try:
        response = requests.post('http://localhost:8001/auth/token', headers=headers, data=data)
        response_value = response.json()
        return response_value       #returns access & refresh tokens if existing, else "detail"
    except requests.exceptions.RequestException as e:
        # Handle exceptions, print an error message, or raise a custom exception if needed
        print(f"Error occurred during token request: {e}")
        return None

## Place the current token to blacklist
def logout(request):
    access_token = request.session["access_token"]
    headers = {
    'accept': 'application/json',
    'Authorization': 'Bearer '+access_token,}

    #add the active access token to the blacklist
    response = requests.post('http://localhost:8001/auth/logout', headers=headers)
    return render(request, 'login.html')




## Refreshing the tokens, get new access token
def refresh_access_token(request):
    access_token = request.session.get('access_token')
    refresh_token = request.session.get('refresh_token')
    refresh_exp = request.session.get('refresh_exp')

    #refresh_token is not yet expired
    if refresh_token and (dt.utcfromtimestamp(refresh_exp) > dt.utcnow()):
        print("*"*100)
        print("expiry of refresh token: ", dt.utcfromtimestamp(refresh_exp))
        print("time today: ", dt.utcnow())
        print("*"*100)
        headers = {'accept': 'application/json',}

        # replace tokens
        params = (('access_token', access_token),('refresh_token', refresh_token),)
        
        # Call your token refresh endpoint with the refresh token
        #if access token is active, add the access token to blacklist
        refresh_response = requests.post('http://localhost:8001/auth/refresh', headers=headers, params=params)
        
        #valid pa si access at refresh token
        if refresh_response.status_code == 200:
            # Update the session with the new access token
            new_access_token = refresh_response.json().get('access_token')
            # print("I entered on line 583", new_access_token)
            request.session['access_token'] = new_access_token
            # print("I entered on line 585", request.session['access_token'])
            return new_access_token

        #access token expired na but refresh token active pa, get new access_token
        else:
            new_access_token = auth.create_access_token(request.session['user_info']['username'], 
                                                        request.session['user_info']['id'], 
                                                        request.session['user_info']['access'], 
                                                        request.session['user_info']['dept'], 
                                                        timedelta(minutes=auth.TOKEN_EXPIRE_MINUTES))
            return new_access_token


    #active and refresh token are both expired
    else:
        print("Expired na both")
        return None
   

#function for handling live agent interactions
def live_agent(request):
    id = chatlog_variables['id']
    log = ChatlogsTest.objects.get(id=id)
    log.escalation = 1
    log.start_datetime = timezone.make_aware(log.start_datetime, timezone.get_current_timezone(), True)
    log.save()
    response = "- - - PROCESS FOR LIVE AGENT CONNECTION - - -"
    return HttpResponse(response.replace('\n', '<br>'))


#function for handling live agent interactions
def q_and_a(request):
    chat_history = load_conversation()
    
    try:
        data = json.loads(request.body)
        user_message = data.get('userMessage', '')
        start_time = time.time()
        
    except json.JSONDecodeError:
        response = {'reply': 'Invalid JSON data'}
        return JsonResponse(response, status=400)

    with get_openai_callback() as cb:

        if user_message.lower() == 'exit':
            response = "Thank you for using AI Chat!"
        
        else:
            user_message = transform_sentence(user_message)
            # Retrieve or initialize conversation history from the session
            conversation_history = request.session.get('conversation_history', [])
            # Add the user message to the conversation history
            conversation_history.append({"role": "user", "content": user_message})

            #AI response generation
            #response = model_response(user_message)
            response = chain({'question': user_message, 'chat_history': chat_history})
            response = response['answer']
            print("AI Response:", response)
            
            #Chat History
            chat_history.append((user_message, response))
            save_conversation(chat_history)
                            
            # Add the assistant's reply to the conversation history
            conversation_history.append({"role": "assistant", "content": response})
            
            # Update the conversation history in the session
            request.session['conversation_history'] = conversation_history
            
    #         #Print Tokens consumed and Cost
    #         print('='*100)
    #         print("OpenAI Consumption")
    #         print(f"Total Tokens: {cb.total_tokens}")
    #         print(f"Prompt Tokens: {cb.prompt_tokens}")               
    #         print(f"Completion Tokens: {cb.completion_tokens}")
    #         print(f"Total Cost (USD): ${cb.total_cost}")
    #         print(f"Total Cost (PHP): PHP{cb.total_cost * round(CurrencyRates().get_rate('USD', 'PHP'), 2)}")
    #         print('='*100)
            insert_to_database(user_message, response, start_time, connection_uri, cb)

    return HttpResponse(response.replace('\n', '<br>'))


def coe(request):
    """@neil @eugene :), pa-update na lang ng part na to ty ty"""
    response = "- - - PROCESS FOR COE REQUEST - - -"
    return HttpResponse(response.replace('\n', '<br>'))

