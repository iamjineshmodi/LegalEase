# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import io
import uuid
from datetime import datetime
import json
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Google Drive and AI imports
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import pinecone
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import fitz # PyMuPDF for better PDF extraction
# import docx
from docx import Document
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

# Environment setup
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Legal Document AI API", version="2.0.0")
security = HTTPBearer()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class DocumentAnalysisResponse(BaseModel):
    document_id: str
    filename: str
    upload_date: datetime
    summary: str
    risk_alerts: List[Dict[str, Any]]
    glossary: List[Dict[str, str]]

class ChatMessage(BaseModel):
    message: str
    document_id: str

class ChatResponse(BaseModel):
    response: str
    sources: List[str]

class TranslationRequest(BaseModel):
    text: str
    target_language: str

# --- Global Service Variables ---
drive_service = None
pinecone_index = None
sentence_model = None
gemini_model = None
executor = ThreadPoolExecutor(max_workers=4)  # For parallel processing

def get_user_credentials() -> Credentials:
    """Load stored OAuth token.json and refresh if needed."""
    creds = Credentials.from_authorized_user_file(
        os.getenv("GOOGLE_OAUTH_TOKEN_PATH", "token.json"),
        scopes=["https://www.googleapis.com/auth/drive.file"]
    )
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return creds

# --- Service Initialization ---
def initialize_services():
    global drive_service, pinecone_index, sentence_model, gemini_model
    
    # Google Drive setup
    try:
        credentials = get_user_credentials()
        drive_service = build('drive', 'v3', credentials=credentials)
    except Exception as e:
        print(f"Google Drive initialization failed: {e}")
    
    # Pinecone setup (new API)
    try:
        from pinecone import Pinecone
        pinecone_client = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        index_name = os.getenv('PINECONE_INDEX_NAME')
        # Check if index exists
        if index_name not in pinecone_client.list_indexes().names():
            from pinecone import ServerlessSpec
            pinecone_client.create_index(
                name=index_name,
                dimension=384,  # Fixed dimension for all-MiniLM-L6-v2
                metric='cosine',  # Cosine is generally better for text embeddings
                spec=ServerlessSpec(
                    cloud='aws',
                    region=os.getenv('PINECONE_REGION', 'us-west-2')
                )
            )
        pinecone_index = pinecone_client.Index(index_name)
    except Exception as e:
        print(f"Pinecone initialization failed: {e}")
    
    # Sentence Transformer model
    try:
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Sentence model loading failed: {e}")

    # Gemini AI model
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')
    except Exception as e:
        print(f"Gemini AI initialization failed: {e}")


@app.on_event("startup")
async def startup_event():
    initialize_services()

# --- Authentication ---
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Placeholder for your actual JWT token validation logic
    # In a real app, you would decode the token, verify it, and fetch the user
    token = credentials.credentials
    if not token:
        raise HTTPException(status_code=401, detail="Authentication token required")
    return {"user_id": "user123"}

# --- Document Processing Utilities ---
def extract_text_from_pdf(file_content: bytes) -> str:
    try:
        with fitz.open(stream=file_content, filetype="pdf") as pdf_document:
            return "".join(page.get_text() for page in pdf_document)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {e}")

def extract_text_from_docx(file_content: bytes) -> str:
    try:
        doc = Document(io.BytesIO(file_content))
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading DOCX: {e}")

def extract_text_from_txt(file_content: bytes) -> str:
    try:
        return file_content.decode('utf-8')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading TXT: {e}")

def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def upload_to_drive(file_content: bytes, filename: str, folder_id: str) -> Optional[str]:
    try:
        file_metadata = {'name': filename, 'parents': [folder_id]}
        media = MediaIoBaseUpload(io.BytesIO(file_content), mimetype='application/octet-stream', resumable=True)
        file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        return file.get('id')
    except Exception as e:
        print(f"Drive upload error: {e}")
        # Non-critical, so we don't raise an exception, just return None
        return None

def store_in_pinecone(chunks: List[str], document_id: str, metadata: dict):
    try:
        vectors = []
        for i, chunk in enumerate(chunks):
            embedding = sentence_model.encode(chunk).tolist()
            chunk_metadata = {
                **metadata,
                'chunk_index': i,
                'chunk_text': chunk[:1000] # Store partial text
            }
            vectors.append({
                'id': f"{document_id}_{i}",
                'values': embedding,
                'metadata': chunk_metadata
            })
        
        # Upsert in batches
        for i in range(0, len(vectors), 100):
            batch = vectors[i:i + 100]
            pinecone_index.upsert(vectors=batch)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store in vector database: {e}")

# --- Gemini AI Analysis Functions ---
def clean_json_response(text: str) -> Any:
    """Cleans and parses a JSON string from an LLM response."""
    # Find the start and end of the JSON object/array
    json_match = re.search(r'\[.*\]|{.*}', text, re.DOTALL)
    if not json_match:
        return None
    
    json_str = json_match.group(0)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        print("JSON Decode Error. Raw text:", text)
        return None

def generate_gemini_content(prompt: str) -> str:
    """Wrapper function to make Gemini API calls synchronous for thread execution."""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini generation error: {e}")
        return ""

async def analyze_with_gemini(text: str) -> Dict[str, Any]:
    """Runs analysis using Gemini AI with parallel processing."""
    # Prompts
    summary_prompt = f"Provide a concise, 150-word summary of the key legal points and implications of this document:\n\n{text}"
    risk_prompt = f"Scan this legal text. Identify clauses that could cost money, limit rights, or cause harm. List ONLY risks. Format as a JSON array of objects, each with 'severity' ('HIGH', 'MEDIUM', 'LOW') and 'description' (under 15 words):\n\n{text}"
    glossary_prompt = f"Create a JSON object mapping all the important legal terms which would be difficult to understand for normal user who doesn't know much about legal jargons in this document to its simple English meaning:\n\n{text}"

    try:
        # Run all Gemini requests in parallel using ThreadPoolExecutor
        loop = asyncio.get_event_loop()
        
        tasks = [
            loop.run_in_executor(executor, generate_gemini_content, summary_prompt),
            loop.run_in_executor(executor, generate_gemini_content, risk_prompt),
            loop.run_in_executor(executor, generate_gemini_content, glossary_prompt)
        ]
        
        # Wait for all tasks to complete
        summary_text, risk_text, glossary_text = await asyncio.gather(*tasks)

        # Process and clean responses
        summary = summary_text
        risks = clean_json_response(risk_text) or []
        glossary = clean_json_response(glossary_text) or {}

        # Convert glossary object to list of objects
        glossary_list = []
        for k, v in glossary.items():
            if isinstance(v, dict) and "simple_meaning" in v:
                glossary_list.append({"term": k, "definition": v["simple_meaning"]})
            else:
                glossary_list.append({"term": k, "definition": str(v)})

        return {
            "summary": summary,
            "risk_alerts": risks,
            "glossary": glossary_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during Gemini AI analysis: {e}")


# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "Legal Document AI API is running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "google_drive": "connected" if drive_service else "disconnected",
            "pinecone": "connected" if pinecone_index else "disconnected",
            "sentence_model": "loaded" if sentence_model else "not_loaded",
            "gemini_model": "loaded" if gemini_model else "not_loaded"
        }
    }

@app.post("/upload", response_model=DocumentAnalysisResponse)
async def upload_document(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user)
):
    """Upload, analyze, and index a legal document."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    file_extension = os.path.splitext(file.filename)[1].lower()
    allowed_extensions = ['.pdf', '.docx', '.txt']
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")

    file_content = await file.read()
    
    # 1. Extract Text
    text = ""
    if file_extension == '.pdf':
        text = extract_text_from_pdf(file_content)
    elif file_extension == '.docx':
        text = extract_text_from_docx(file_content)
    elif file_extension == '.txt':
        text = extract_text_from_txt(file_content)
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from document.")
    
    # Generate document ID early
    document_id = str(uuid.uuid4())
    print("Document ID:", document_id)
    
    # 2. Start AI Analysis and Drive upload in parallel
    analysis_task = asyncio.create_task(analyze_with_gemini(text))
    
    # 3. Upload to Google Drive (Optional, non-blocking)
    folder_id = os.getenv('GOOGLE_DRIVE_FOLDER_ID')
    print(f"Folder ID: {folder_id}")
    
    # Run drive upload in executor to not block
    if folder_id:
        drive_upload_task = asyncio.get_event_loop().run_in_executor(
            executor, upload_to_drive, file_content, file.filename, folder_id
        )
    else:
        drive_upload_task = asyncio.create_task(asyncio.sleep(0, result=None))
    
    # 4. Prepare chunking and metadata
    chunks = chunk_text(text)
    
    # Wait for both analysis and drive upload
    analysis_results, drive_file_id = await asyncio.gather(analysis_task, drive_upload_task)
    
    # 5. Store in Pinecone
    metadata = {
        'document_id': document_id,
        'filename': file.filename,
        'user_id': user['user_id'],
        'drive_file_id': drive_file_id or '',
        'upload_date': datetime.now().isoformat(),
    }
    store_in_pinecone(chunks, document_id, metadata)
    
    # 6. Return the comprehensive analysis
    return DocumentAnalysisResponse(
        document_id=document_id,
        filename=file.filename,
        upload_date=datetime.now(),
        **analysis_results
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_with_document(
    chat_request: ChatMessage,
    user: dict = Depends(get_current_user)
):
    """Chat with a specific document using RAG and Gemini."""
    try:
        # 1. Embed the user's query
        query_embedding = sentence_model.encode(chat_request.message).tolist()
        
        # 2. Search Pinecone for relevant context
        search_results = pinecone_index.query(
            vector=query_embedding,
            filter={
                'document_id': chat_request.document_id,
                'user_id': user['user_id']
            },
            top_k=5,
            include_metadata=True
        )
        
        if not search_results.matches:
            return ChatResponse(response="I could not find relevant information in the document.", sources=[])
        
        # 3. Build context and sources for the LLM
        context = "\n---\n".join([match.metadata.get('chunk_text', '') for match in search_results.matches])
        sources = sorted(list(set([f"Chunk {match.metadata.get('chunk_index', 0)}" for match in search_results.matches])))
        
        # 4. Generate a response with Gemini
        prompt = f"""
        Based on the following context from a legal document, answer the user's question.
        Answer in a clear, helpful tone. If the context doesn't contain the answer, say so.

        Context:
        {context}

        User's Question:
        {chat_request.message}

        Answer:
        """
        
        response = gemini_model.generate_content(prompt)
        
        return ChatResponse(
            response=response.text.strip(),
            sources=sources
        )
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Error processing chat request")

@app.get("/documents")
async def get_user_documents(user: dict = Depends(get_current_user)):
    """Get a list of all documents uploaded by the user."""
    try:
        # A dummy query to fetch metadata by filtering
        response = pinecone_index.query(
            vector=[0] * 384,  # Updated to match embedding dimension
            filter={'user_id': user['user_id']},
            top_k=1000,
            include_metadata=True
        )
        
        documents = {}
        for match in response.matches:
            doc_id = match.metadata.get('document_id')
            if doc_id and doc_id not in documents:
                documents[doc_id] = {
                    'document_id': doc_id,
                    'filename': match.metadata.get('filename'),
                    'upload_date': match.metadata.get('upload_date')
                }
        
        return list(documents.values())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {e}")

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str, user: dict = Depends(get_current_user)):
    """Delete a document and all its associated data from the vector store."""
    try:
        # Note: This does not delete the file from Google Drive.
        # A more robust solution would involve a background task to do that.
        pinecone_index.delete(filter={
            'document_id': document_id,
            'user_id': user['user_id']
        })
        return {"message": "Document and its embeddings deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {e}")

@app.post("/translate")
async def translate_text(request: TranslationRequest):
    """Translate text using Gemini AI."""
    supported_languages = ["hindi", "bengali", "tamil", "telugu", "marathi", "gujarati", "kannada", "malayalam", "punjabi", "urdu"]
    if request.target_language.lower() not in supported_languages:
        raise HTTPException(status_code=400, detail=f"Language '{request.target_language}' not supported.")
    
    try:
        prompt = f"Translate the following text to {request.target_language}. Maintain the core meaning precisely:\n\n{request.text}"
        response = gemini_model.generate_content(prompt)
        return {"translated_text": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {e}")

@app.get("/supported-languages")
async def get_supported_languages():
    """Get a list of supported languages for translation."""
    return {
        "languages": [
            {"code": "hindi", "name": "Hindi"},
            {"code": "bengali", "name": "Bengali"},
            {"code": "tamil", "name": "Tamil"},
            {"code": "telugu", "name": "Telugu"},
            {"code": "marathi", "name": "Marathi"},
            {"code": "gujarati", "name": "Gujarati"},
            {"code": "kannada", "name": "Kannada"},
            {"code": "malayalam", "name": "Malayalam"},
            {"code": "punjabi", "name": "Punjabi"},
            {"code": "urdu", "name": "Urdu"}
        ]
    }

# Cleanup executor on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    executor.shutdown(wait=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)