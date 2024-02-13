from langchain import OpenAI, SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

# from langchain.agents import create_sql_agent
# from langchain.agents.agent_types import AgentType
# from langchain.sql_database import SQLDatabase
# from langchain_community.agent_toolkits import SQLDatabaseToolkit
# from langchain_openai import OpenAI

import os
import warnings
import constants
os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY
warnings.filterwarnings('ignore')

# Setup database
# db = SQLDatabase.from_uri(
#     f"postgresql+psycopg2://postgres:{constants.DBPASS}@localhost:5433/{constants.DB}",
# )

db = SQLDatabase.from_uri(
    f"postgresql://{constants.DB_CONFIG['user']}:{constants.DB_CONFIG['password']}@{constants.DB_CONFIG['host']}:{constants.DB_CONFIG['port']}/{constants.DB_CONFIG['dbname']}"
)

# Setup llm
llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Create db chain
QUERY = """
Given an input question, first create a syntactically correct postgresql query to run, then look at the results of the query and return the answer.
Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

{question}
"""

# Setup the database chain
db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)

# Database chain
# toolkit = SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0, verbose=True, model_name="gpt-3.5-turbo"))
# agent_executor = create_sql_agent(
#     llm=OpenAI(temperature=0),
#     toolkit=toolkit,
#     verbose=False,
#     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
# )

def get_prompt():
    print("Type 'exit' to quit")

    while True:
        prompt = input("\nEnter a prompt: ")

        if prompt.lower() == 'exit':
            print("Exiting...")
            break
        else:
            try:
                question = QUERY.format(question=prompt)
                print(db_chain.run(question))
            except Exception as e:
                print(e)

get_prompt()