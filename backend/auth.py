# auth_setup.py
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os, pickle

SCOPES = ["https://www.googleapis.com/auth/drive.file"]  # or 'drive' for full access

flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", SCOPES)
creds = flow.run_local_server(port=8080)     # opens browser for Google login/consent
with open("token.json", "w") as f:
    f.write(creds.to_json())
print("token.json saved.  Upload this file to your server securely.")
