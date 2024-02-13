print("I entered here")

import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import sys
print(sys.path)
# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def download_file(service, file_id, filepath):
    # Create a media request to download the file content
    request = service.files().get_media(fileId=file_id)
    # Open a local file in write binary mode
    fh = open(filepath, "wb")
    # Use MediaIoBaseDownload for streaming the file content and writing to the local file
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        # Download the file in chunks and display progress
        status, done = downloader.next_chunk()
        print("Download %d%%." % int(status.progress() * 100))
    fh.close()

def download_files_recursively(service, folder_id, parent_path=""):
    # List files and folders inside the specified folder_id
    results = service.files().list(
        pageSize=1000,
        q=f"'{folder_id}' in parents and trashed=false",
        fields="files(id, name, mimeType)"
    ).execute()

    items = results.get("files", [])

    for item in items:
        item_name = item["name"]
        item_id = item["id"]
        item_type = item["mimeType"]
        item_path = os.path.join(parent_path, item_name)

        if "application/vnd.google-apps.folder" in item_type:
            # If the item is a folder, create the folder locally and recursively download its contents
            if not os.path.exists(item_path):
                os.makedirs(item_path)
            download_files_recursively(service, item_id, parent_path=item_path)
        else:
            # If the item is a file, download the file content
            download_file(service, item_id, filepath=item_path)


# Load or obtain user credentials
creds = None
if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)

if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        # Run the authorization flow if credentials are missing or expired
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        creds = flow.run_local_server(port=0)

    # Save the updated credentials for future use
    with open("token.json", "w") as token:
        token.write(creds.to_json())

# Build the Google Drive API service
service = build("drive", "v3", credentials=creds)

# Start downloading files recursively from the root of Google Drive
download_files_recursively(service, folder_id="root")