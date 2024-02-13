import constants
import psycopg2
import os
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import OpenAI

os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY

# Access DB_CONFIG from constants.py
connection_uri = f"postgresql://{constants.DB_CONFIG['user']}:{constants.DB_CONFIG['password']}@{constants.DB_CONFIG['host']}:{constants.DB_CONFIG['port']}/{constants.DB_CONFIG['dbname']}"
db = SQLDatabase.from_uri(connection_uri)

def get_all_tables():
    with psycopg2.connect(connection_uri) as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
            tables = cursor.fetchall()
    return [table[0] for table in tables]

toolkit = SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0.7, verbose=True, model_name="gpt-3.5-turbo"))
agent_executor = create_sql_agent(
    llm=OpenAI(temperature=0.7),
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    all_tables_method=get_all_tables,
)
query = input("Ask me anything? ")
result = agent_executor.run(query)
print("Result:", result)