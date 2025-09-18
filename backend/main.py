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

# --- Core ML/AI Imports ---
import torch
import fitz  # PyMuPDF for better PDF extraction
from docx import Document
from dotenv import load_dotenv

# --- Service-specific Imports ---
# Pinecone
from pinecone import Pinecone, ServerlessSpec
# Google
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import google.generativeai as genai
# Transformers
from sentence_transformers import SentenceTransformer, CrossEncoder # MODIFIED: Added SentenceTransformer back
from transformers import AutoTokenizer, AutoModelForMaskedLM

# --- Configuration & Initialization ---
load_dotenv()
app = FastAPI(title="Legal Document AI API - Advanced RAG", version="3.1.0")
security = HTTPBearer()

# --- CORS Middleware ---
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
    file_url: Optional[str] = None

class ChatMessage(BaseModel):
    message: str
    document_id: str

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    
class TranslationRequest(BaseModel):
    text: str
    target_language: str

# --- Constants & Global Variables ---
GENERATION_MODEL_NAME = "gemini-2.5-flash-lite"
# MODIFIED: Define local embedding model
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2" 
PINECONE_DIMENSION = 768  # Dimension for all-mpnet-base-v2

drive_service = None
pinecone_index = None
gemini_model = None
sentence_model = None # ADDED: For local embeddings
splade_encoder = None
cross_encoder = None
executor = ThreadPoolExecutor(max_workers=4)

# --- Sparse Vector Encoder (SPLADE) ---
class SpladeEncoder:
    def __init__(self, model_id='naver/splade-cocondenser-ensembledistil'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Initializing SpladeEncoder on device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForMaskedLM.from_pretrained(model_id).to(self.device)

    def _encode(self, text: str) -> dict:
        tokens = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            output = self.model(**tokens)
        
        vec = torch.max(
            torch.log(1 + torch.relu(output.logits)) * tokens.attention_mask.unsqueeze(-1),
            dim=1
        )[0].squeeze()
        
        cols = vec.nonzero().squeeze().cpu().tolist()
        weights = vec[cols].cpu().tolist()
        
        if not isinstance(cols, list):
            cols, weights = ([cols], [weights])
            
        return {'indices': cols, 'values': weights}

    def encode_documents(self, texts: List[str]) -> List[dict]:
        return [self._encode(text) for text in texts]

    def encode_query(self, query: str) -> dict:
        return self._encode(query)

# --- Service Initialization ---
def get_user_credentials() -> Optional[Credentials]:
    token_path = os.getenv("GOOGLE_OAUTH_TOKEN_PATH", "token.json")
    if not os.path.exists(token_path):
        print(f"Warning: OAuth token file not found at {token_path}")
        return None
    creds = Credentials.from_authorized_user_file(token_path, scopes=["https://www.googleapis.com/auth/drive.file"])
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return creds

@app.on_event("startup")
async def startup_event():
    global drive_service, pinecone_index, gemini_model, sentence_model, splade_encoder, cross_encoder
    
    # Google Drive
    try:
        credentials = get_user_credentials()
        if credentials:
            drive_service = build('drive', 'v3', credentials=credentials)
    except Exception as e:
        print(f"Google Drive initialization failed: {e}")
    
    # Pinecone
    try:
        pinecone_client = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        index_name = os.getenv('PINECONE_INDEX_NAME')
        if index_name not in pinecone_client.list_indexes().names():
            pinecone_client.create_index(
                name=index_name,
                dimension=PINECONE_DIMENSION,
                metric='dotproduct',
                spec=ServerlessSpec(cloud='aws', region=os.getenv('PINECONE_REGION', 'us-west-2'))
            )
        pinecone_index = pinecone_client.Index(index_name)
    except Exception as e:
        print(f"Pinecone initialization failed: {e}")
    
    # Gemini AI
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        gemini_model = genai.GenerativeModel(GENERATION_MODEL_NAME)
    except Exception as e:
        print(f"Gemini AI initialization failed: {e}")

    # MODIFIED: Initialize local and other models
    print(f"Loading local embedding model: {EMBEDDING_MODEL_NAME}")
    sentence_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    splade_encoder = SpladeEncoder()
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-minilm-l-6-v2')
    print("All models initialized successfully.")


# --- Authentication ---
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if not token:
        raise HTTPException(status_code=401, detail="Authentication token required")
    return {"user_id": "user123"}


# --- Document Processing & Embedding Utilities ---
def extract_text_from_pdf(file_content: bytes) -> str:
    with fitz.open(stream=file_content, filetype="pdf") as doc:
        return "".join(page.get_text() for page in doc)

def extract_text_from_docx(file_content: bytes) -> str:
    doc = Document(io.BytesIO(file_content))
    return "\n".join([p.text for p in doc.paragraphs])

def upload_to_drive(file_content: bytes, filename: str, folder_id: str) -> Optional[str]:
    # ... (This function is unchanged)
    if not drive_service:
        print("Drive service not initialized. Skipping upload.")
        return None
    try:
        file_metadata = {'name': filename, 'parents': [folder_id]}
        media = MediaIoBaseUpload(io.BytesIO(file_content), mimetype='application/octet-stream', resumable=True)
        file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        return file.get('id')
    except Exception as e:
        print(f"Drive upload error: {e}")
        return None

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return text_splitter.split_text(text)

# --- REWRITTEN: store_in_pinecone for Hybrid Search with LOCAL embeddings ---
def store_in_pinecone(chunks: List[str], document_id: str, metadata: dict):
    try:
        # 1. Get Dense Embeddings from local SentenceTransformer
        dense_embeds = sentence_model.encode(chunks, convert_to_tensor=False).tolist()

        # 2. Get Sparse Embeddings from SPLADE
        sparse_embeds = splade_encoder.encode_documents(chunks)

        # 3. Prepare vectors for upsert
        vectors = [{
            'id': f"{document_id}_chunk_{i}",
            'values': dense_embeds[i],
            'sparse_values': sparse_embeds[i],
            'metadata': {**metadata, 'chunk_index': i, 'chunk_text': chunk}
        } for i, chunk in enumerate(chunks)]
        
        # 4. Upsert in batches
        for i in range(0, len(vectors), 50):
            pinecone_index.upsert(vectors=vectors[i:i + 50])
        print(f"Successfully stored {len(vectors)} hybrid vectors in Pinecone.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store in vector database: {e}")

# --- AI Analysis & Chat Logic ---
# ... (analyze_with_gemini and clean_json_response are unchanged) ...
def clean_json_response(text: str) -> Any:
    match = re.search(r'```json\s*([\s\S]*?)\s*```|(\[.*\]|\{.*\})', text, re.DOTALL)
    if not match: return None
    json_str = match.group(1) or match.group(2)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        print("JSON Decode Error. Raw text:", text)
        return None

async def analyze_with_gemini(text: str) -> Dict[str, Any]:
    summary_prompt = f"Provide a concise, 150-word summary of the key legal points and implications of this document:\n\n{text}"
    risk_prompt = f"Scan this legal text. Identify clauses that could cost money, limit rights, or cause harm. List ONLY risks. Format as a JSON array of objects, each with 'severity' ('HIGH', 'MEDIUM', 'LOW') and 'description' (under 15 words). Respond with ```json ... ``` block:\n\n{text}"
    glossary_prompt = f"Extract important legal terms from the document that a non-lawyer might not know. Return a single valid JSON object where keys are the terms and values are plain-English explanations. Respond with ```json ... ``` block:\n\n{text}"
    
    async def run_prompt(p):
        return await asyncio.to_thread(gemini_model.generate_content, p)

    summary_res, risk_res, glossary_res = await asyncio.gather(
        run_prompt(summary_prompt), run_prompt(risk_prompt), run_prompt(glossary_prompt)
    )

    summary = summary_res.text.strip()
    risks = clean_json_response(risk_res.text) or []
    glossary_data = clean_json_response(glossary_res.text) or {}
    glossary = [{"term": k, "definition": v} for k, v in glossary_data.items()]

    return {"summary": summary, "risk_alerts": risks, "glossary": glossary}

# --- API Endpoints ---
@app.post("/upload", response_model=DocumentAnalysisResponse)
async def upload_document(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    # ... (This endpoint is unchanged) ...
    file_content = await file.read()
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    text_extractors = {'.pdf': extract_text_from_pdf, '.docx': extract_text_from_docx, '.txt': lambda c: c.decode('utf-8')}
    if file_extension not in text_extractors:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")

    text = text_extractors[file_extension](file_content)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from document.")

    document_id = str(uuid.uuid4())
    chunks = chunk_text(text)
    
    analysis_task = analyze_with_gemini(text)
    
    drive_file_id = None
    if folder_id := os.getenv('GOOGLE_DRIVE_FOLDER_ID'):
        drive_file_id = await asyncio.to_thread(upload_to_drive, file_content, file.filename, folder_id)

    analysis_results = await analysis_task
    
    metadata = {
        'document_id': document_id, 'filename': file.filename, 'user_id': user['user_id'],
        'drive_file_id': drive_file_id or '', 'upload_date': datetime.now().isoformat(),
    }
    await asyncio.to_thread(store_in_pinecone, chunks, document_id, metadata)
    
    file_url = f"https://drive.google.com/file/d/{drive_file_id}/view" if drive_file_id else None
    
    return DocumentAnalysisResponse(
        document_id=document_id, filename=file.filename,
        upload_date=datetime.now(), file_url=file_url, **analysis_results
    )

# --- REWRITTEN: /chat endpoint with LOCAL dense embeddings ---
@app.post("/chat", response_model=ChatResponse)
async def chat_with_document(chat_request: ChatMessage, user: dict = Depends(get_current_user)):
    try:
        # 1. Embed the query for both dense and sparse search
        dense_query_embedding = sentence_model.encode(chat_request.message).tolist()
        sparse_query_embedding = splade_encoder.encode_query(chat_request.message)

        # 2. Perform Hybrid Search in Pinecone
        search_results = pinecone_index.query(
            vector=dense_query_embedding,
            sparse_vector=sparse_query_embedding,
            filter={'document_id': chat_request.document_id, 'user_id': user['user_id']},
            top_k=20,
            include_metadata=True
        )
        
        if not search_results.matches:
            return ChatResponse(response="I could not find relevant information in the document.", sources=[])

        # 3. Re-rank the results
        initial_chunks = [match.metadata.get('chunk_text', '') for match in search_results.matches]
        pairs = [[chat_request.message, chunk] for chunk in initial_chunks]
        scores = await asyncio.to_thread(cross_encoder.predict, pairs)
        
        scored_chunks = sorted(zip(scores, search_results.matches), key=lambda x: x[0], reverse=True)
        
        # 4. Select the best re-ranked chunks for context
        top_k_reranked = 8
        final_context_chunks = [match.metadata.get('chunk_text', '') for _, match in scored_chunks[:top_k_reranked]]
        context = "\n---\n".join(final_context_chunks)
        
        sources = [{
            "chunk_index": match.metadata.get('chunk_index', -1),
            "text": match.metadata.get('chunk_text', '')[:200] + '...',
            "relevance_score": float(score)
        } for score, match in scored_chunks[:top_k_reranked]]

        # 5. Generate a response with Gemini
        prompt = f"""
        Based ONLY on the following context from a legal document, answer the user's question.
        Answer in a clear, helpful tone. If the context doesn't contain the answer, state that.

        Context:
        {context}

        User's Question:
        {chat_request.message}

        Answer:
        """
        response = await asyncio.to_thread(gemini_model.generate_content, prompt)
        
        return ChatResponse(response=response.text.strip(), sources=sources)
        
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Error processing chat request")


@app.get("/documents")
async def get_user_documents(user: dict = Depends(get_current_user)):
    """Get a list of all documents uploaded by the user."""
    try:
        # A dummy query to fetch metadata by filtering
        response = pinecone_index.query(
            vector=[0] * 768,  # Updated to match embedding dimension
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
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)