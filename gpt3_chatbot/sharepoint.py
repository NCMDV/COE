import pandas as pd
import psycopg2
from office365.runtime.auth.authentication_context import AuthenticationContext
from office365.sharepoint.client_context import ClientContext
import constants

# SharePoint authentication (replace with your credentials and site URL)
ctx_auth = AuthenticationContext(url=constants.sp_url)
ctx_auth.acquire_token_for_user(username=constants.sp_username, password=constants.sp_password)

def create_or_update_table(table_name, data):
    """Creates a table if it doesn't exist, otherwise updates it with new data."""
    with psycopg2.connect(**constants.DB_CONFIG) as conn:
        cursor = conn.cursor()

        # Check if table exists
        cursor.execute("SELECT to_regclass('%s')" % table_name)
        table_exists = cursor.fetchone()[0] is not None

        if not table_exists:
            # Create table
            sql = f"CREATE TABLE {table_name} (" \
                  f"employee_id INT PRIMARY KEY, " \
                  f"firstname VARCHAR(50), " \
                  f"lastname VARCHAR(50), " \
                  f"middlename VARCHAR(50), " \
                  f"department VARCHAR(50), " \
                  f"position VARCHAR(50), " \
                  f"hire_date DATE)"  # Adjust column types as needed
            cursor.execute(sql)
            print(f"Table '{table_name}' created successfully.")
        else:
            # Update existing table (replace with your preferred update logic)
            data.to_sql(table_name, conn, if_exists='replace', index=False)
            print(f"Table '{table_name}' updated successfully.")

def process_folder(folder_url):
    """Processes a SharePoint folder and saves its data to the database."""
    ctx = ClientContext(folder_url, ctx_auth)
    folder = ctx.web.get_folder_by_server_relative_url(folder_url)
    files = folder.files
    ctx.load(files)
    ctx.execute_query()

    table_name = folder.properties["Name"]
    data = pd.DataFrame()

    for file in files:
        file_url = file.serverRelativeUrl
        try:
            # Read file content using appropriate method based on file extension
            df = pd.read_csv(file_url)  # Adjust for other file types
            data = pd.concat([data, df])
        except Exception as e:
            print(f"Error processing file '{file_url}': {e}")

    if not data.empty:
        create_or_update_table(table_name, data)

# Get root folder URL
root_folder_url = constants.sp_root_folder_url

# Process all folders recursively
process_folder(root_folder_url)  # Call the function to start processing
