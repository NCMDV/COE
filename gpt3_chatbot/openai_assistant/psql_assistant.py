from function import Function, Property
from dotenv import load_dotenv
from assistant import AIAssistant
import psycopg2
import constants

load_dotenv()

connection_uri = f"postgresql://{constants.DB_CONFIG['user']}:{constants.DB_CONFIG['password']}@{constants.DB_CONFIG['host']}:{constants.DB_CONFIG['port']}/{constants.DB_CONFIG['dbname']}"

class GetDBSchema(Function):
    def __init__(self):
        super().__init__(
            name="get_db_schema",
            description="Get the schema of the database",
        )

    def function(self):
        with psycopg2.connect(connection_uri) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'gpt_data';")
                tables = cursor.fetchall()
                tables = [table[0] for table in tables]
                columns = []
                schema_statements = []

                for table in tables:
                    cursor.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = 'gpt_data' AND table_name = '{table}'")
                    cols = cursor.fetchall()
                    # columns.append(cursor.fetchall())
                    create_statement = f"CREATE TABLE {table} (\n "
                    create_statement += ",\n ".join(
                        f"{col[0]} {col[1]}" for col in cols
                    )
                    create_statement += "\n);"
                    schema_statements.append(create_statement)

                # tables_with_columns = zip(tables, columns)
                # print(schema_statements)
        
                return "\n\n".join(schema_statements)

class RunSQLQuery(Function):
    def __init__(self):
        super().__init__(
            name="run_sql_query",
            description="Run a SQL query on the database",
            parameters=[
                Property(
                    name="query",
                    description="The SQL query to run",
                    type="string",
                    required=True,
                ),
            ],
        )

    def function(self, query):
        # with psycopg2.connect(connection_uri) as conn:
        #     with conn.cursor() as cursor:
        #         results = cursor.execute(query).fetchall()
        #         return '\n'.join([str(result) for result in results])
        
        try:
            conn = psycopg2.connect(connection_uri)
            cursor = conn.cursor()

            results = cursor.execute(query).fetchall()
            
            return '\n'.join([str(result) for result in results])
        except Exception as e:
            return str(e)
        finally:
            conn.close()
            

if __name__ == "__main__":
    assistant = AIAssistant(
        instruction="""
You are a SQL expert. User asks you questions about the Insurance database.
First obtain the schema of the database to check the tables and columns, then generate SQL queries to answer the questions.
""",
        model="gpt-3.5-turbo-0613",
        functions=[GetDBSchema(), RunSQLQuery()],
        use_code_interpreter=True,
    ) 

    assistant.chat()