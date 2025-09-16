from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests
import re
from typing import Optional
import io
from urllib.parse import urlparse, parse_qs

router = APIRouter(prefix="/process_link", tags=["links"])

class LinkRequest(BaseModel):
    link: str

class DocumentResponse(BaseModel):
    received_link: str
    file_id: str
    file_name: Optional[str] = None
    file_type: str
    content: Optional[str] = None
    download_url: str

def extract_file_id_from_drive_link(link: str) -> str:
    """
    Extract Google Drive file ID from various Google Drive URL formats
    """
    # Pattern 1: https://drive.google.com/file/d/{file_id}/view
    pattern1 = r'/file/d/([a-zA-Z0-9-_]+)'
    
    # Pattern 2: https://drive.google.com/open?id={file_id}
    pattern2 = r'[?&]id=([a-zA-Z0-9-_]+)'
    
    # Pattern 3: https://docs.google.com/document/d/{file_id}
    pattern3 = r'/document/d/([a-zA-Z0-9-_]+)'
    
    # Pattern 4: https://docs.google.com/spreadsheets/d/{file_id}
    pattern4 = r'/spreadsheets/d/([a-zA-Z0-9-_]+)'
    
    # Pattern 5: https://docs.google.com/presentation/d/{file_id}
    pattern5 = r'/presentation/d/([a-zA-Z0-9-_]+)'
    
    patterns = [pattern1, pattern2, pattern3, pattern4, pattern5]
    
    for pattern in patterns:
        match = re.search(pattern, link)
        if match:
            return match.group(1)
    
    raise ValueError("Could not extract file ID from the provided Google Drive link")

def get_file_metadata(file_id: str) -> dict:
    """
    Get file metadata using Google Drive API (requires API key)
    For now, we'll determine file type from the URL structure
    """
    # This is a simplified version. In production, you'd use Google Drive API
    if 'document' in file_id or '/document/' in file_id:
        return {"name": "Unknown Document", "type": "document"}
    elif 'spreadsheet' in file_id or '/spreadsheets/' in file_id:
        return {"name": "Unknown Spreadsheet", "type": "spreadsheet"}
    else:
        return {"name": "Unknown File", "type": "file"}

def download_google_doc_as_text(file_id: str) -> str:
    """
    Download Google Doc as plain text
    """
    export_url = f"https://docs.google.com/document/d/{file_id}/export?format=txt"
    
    try:
        response = requests.get(export_url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")

def download_google_sheet_as_csv(file_id: str) -> str:
    """
    Download Google Sheet as CSV
    """
    export_url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv"
    
    try:
        response = requests.get(export_url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download spreadsheet: {str(e)}")

def download_drive_file(file_id: str) -> bytes:
    """
    Download file from Google Drive using direct download link
    """
    download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
    
    try:
        response = requests.get(download_url)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download file: {str(e)}")

@router.post("/", response_model=DocumentResponse)
def process_link(request: LinkRequest):
    try:
        # Extract file ID from the Google Drive link
        file_id = extract_file_id_from_drive_link(request.link)
        
        # Determine file type and download content based on URL
        if 'docs.google.com/document' in request.link:
            file_type = "document"
            file_name = "Google Document"
            # Download the document content as text
            content = download_google_doc_as_text(file_id)
            download_url = f"https://docs.google.com/document/d/{file_id}/export?format=txt"
            
        elif 'docs.google.com/spreadsheets' in request.link:
            file_type = "spreadsheet"
            file_name = "Google Spreadsheet"
            # Download the spreadsheet content as CSV
            content = download_google_sheet_as_csv(file_id)
            download_url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv"
            
        elif 'docs.google.com/presentation' in request.link:
            file_type = "presentation"
            file_name = "Google Presentation"
            # Download presentation as plain text
            export_url = f"https://docs.google.com/presentation/d/{file_id}/export/txt"
            try:
                response = requests.get(export_url)
                response.raise_for_status()
                content = response.text
            except requests.RequestException:
                content = "Presentation content could not be extracted as text"
            download_url = export_url
            
        else:
            file_type = "file"
            file_name = "Drive File"
            # For other file types, download binary content
            try:
                binary_content = download_drive_file(file_id)
                # Try to decode as text if possible
                try:
                    content = binary_content.decode('utf-8')
                except UnicodeDecodeError:
                    content = f"Binary file downloaded successfully. Size: {len(binary_content)} bytes. Content type: binary"
                download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to download file: {str(e)}")
        
        return DocumentResponse(
            received_link=request.link,
            file_id=file_id,
            file_name=file_name,
            file_type=file_type,
            content=content,
            download_url=download_url
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the link: {str(e)}")

# Additional endpoint to get just the file content
@router.get("/content/{file_id}")
def get_file_content(file_id: str, file_type: str = "document"):
    """
    Get file content by file ID
    """
    try:
        if file_type == "document":
            content = download_google_doc_as_text(file_id)
        elif file_type == "spreadsheet":
            content = download_google_sheet_as_csv(file_id)
        else:
            binary_content = download_drive_file(file_id)
            content = f"Binary file, size: {len(binary_content)} bytes"
        
        return {"file_id": file_id, "content": content}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve content: {str(e)}")

# Endpoint to check if a link is a valid Google Drive link
@router.post("/validate")
def validate_drive_link(request: LinkRequest):
    """
    Validate if the provided link is a Google Drive link
    """
    try:
        file_id = extract_file_id_from_drive_link(request.link)
        return {
            "valid": True, 
            "file_id": file_id,
            "message": "Valid Google Drive link"
        }
    except ValueError:
        return {
            "valid": False, 
            "file_id": None,
            "message": "Invalid Google Drive link format"
        }