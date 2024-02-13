# pip install Office365-REST-Python-Client
from office365.runtime.auth.user_credential import UserCredential
from office365.sharepoint.client_context import ClientContext
from sharepoint_creds import username, password

# --------------- SHAREPOINT LINKS -------------------------
url_site = "https://vivalavi.sharepoint.com/sites/GPTChatBot"
site_name = "GPTChatBot"
target_directory = "Shared Documents" # root
# ----------------------------------------------------------

# login using username and password
user_creds = UserCredential(username,password)
ctx = ClientContext(url_site).with_credentials(user_creds)

# getting all files in the target directory
list_source = ctx.web.get_folder_by_server_relative_url(target_directory)
files = list_source.files
ctx.load(files)
ctx.execute_query

# download every file in current directory
for file in files:
    path = str(file.serverRelativeUrl)
    filename = path.split("/")[-1]
    filepath = f"./sharepoint_files/{filename}"
    with open(filepath, "wb") as local_file:
        print(f"downloading {filename}")
        myfile = (ctx.web.get_file_by_server_relative_path(path)
             .download(local_file)
             .execute_query() 
             )
        print(f"{filename} downloaded successfully!")