# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
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
import threading
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
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForMaskedLM

# --- Configuration & Initialization ---
load_dotenv()
app = FastAPI(title="Legal Document AI API - Advanced RAG", version="3.2.0")
security = HTTPBearer()

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "https://legalease1.vercel.app"],
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
    key_points: List[str]  # Added key points
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

class ServiceStatus(BaseModel):
    models_loaded: bool
    services_ready: bool
    loading_progress: Dict[str, bool]

# --- Constants & Global Variables ---
GENERATION_MODEL_NAME = "gemini-1.5-flash"
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2" 
PINECONE_DIMENSION = 768

# Global service variables
drive_service = None
pinecone_index = None
gemini_model = None
sentence_model = None
splade_encoder = None
cross_encoder = None
executor = ThreadPoolExecutor(max_workers=4)

# Loading status tracking
loading_status = {
    "drive_service": False,
    "pinecone": False,
    "gemini": False,
    "sentence_model": False,
    "splade_encoder": False,
    "cross_encoder": False,
    "all_ready": False
}

# Thread lock for status updates
status_lock = threading.Lock()

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

# --- Service Initialization Functions ---
def get_user_credentials() -> Optional[Credentials]:
    token_path = os.getenv("GOOGLE_OAUTH_TOKEN_PATH", "token.json")
    if not os.path.exists(token_path):
        print(f"Warning: OAuth token file not found at {token_path}")
        return None
    creds = Credentials.from_authorized_user_file(token_path, scopes=["https://www.googleapis.com/auth/drive.file"])
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return creds

def update_loading_status(service: str, status: bool):
    """Thread-safe status update"""
    with status_lock:
        loading_status[service] = status
        # Check if all services are ready
        required_services = ["pinecone", "gemini", "sentence_model", "splade_encoder", "cross_encoder"]
        loading_status["all_ready"] = all(loading_status[service] for service in required_services)
        print(f"Service {service}: {'✓' if status else '✗'} | All ready: {loading_status['all_ready']}")

async def initialize_services_background():
    """Background task to initialize all heavy services"""
    global drive_service, pinecone_index, gemini_model, sentence_model, splade_encoder, cross_encoder
    
    print("🚀 Starting background service initialization...")
    
    # Google Drive (optional, non-blocking)
    try:
        credentials = get_user_credentials()
        if credentials:
            drive_service = build('drive', 'v3', credentials=credentials)
            update_loading_status("drive_service", True)
        else:
            update_loading_status("drive_service", True)  # Mark as "ready" even if not configured
    except Exception as e:
        print(f"Google Drive initialization failed: {e}")
        update_loading_status("drive_service", True)  # Non-critical, mark as ready
    
    # Pinecone
    try:
        print("📊 Initializing Pinecone...")
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
        update_loading_status("pinecone", True)
        print("✅ Pinecone ready")
    except Exception as e:
        print(f"❌ Pinecone initialization failed: {e}")
        update_loading_status("pinecone", False)
    
    # Gemini AI
    try:
        print("🤖 Initializing Gemini AI...")
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        gemini_model = genai.GenerativeModel(GENERATION_MODEL_NAME)
        update_loading_status("gemini", True)
        print("✅ Gemini AI ready")
    except Exception as e:
        print(f"❌ Gemini AI initialization failed: {e}")
        update_loading_status("gemini", False)

    # Sentence Transformer (most memory-intensive)
    try:
        print(f"🧠 Loading SentenceTransformer model: {EMBEDDING_MODEL_NAME}")
        sentence_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        update_loading_status("sentence_model", True)
        print("✅ SentenceTransformer ready")
    except Exception as e:
        print(f"❌ SentenceTransformer initialization failed: {e}")
        update_loading_status("sentence_model", False)
    
    # SPLADE Encoder (also memory-intensive)
    try:
        print("🔍 Loading SPLADE encoder...")
        splade_encoder = SpladeEncoder()
        update_loading_status("splade_encoder", True)
        print("✅ SPLADE encoder ready")
    except Exception as e:
        print(f"❌ SPLADE encoder initialization failed: {e}")
        update_loading_status("splade_encoder", False)
    
    # Cross Encoder
    try:
        print("🎯 Loading CrossEncoder...")
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-minilm-l-6-v2')
        update_loading_status("cross_encoder", True)
        print("✅ CrossEncoder ready")
    except Exception as e:
        print(f"❌ CrossEncoder initialization failed: {e}")
        update_loading_status("cross_encoder", False)
    
    if loading_status["all_ready"]:
        print("🎉 All models initialized successfully! API is fully ready.")
    else:
        print("⚠️  Some models failed to initialize. Check logs above.")

@app.on_event("startup")
async def startup_event():
    """Fast startup - only start background loading"""
    print("🚀 FastAPI server starting...")
    print("📦 Models will load in the background...")
    
    # Start background initialization
    asyncio.create_task(initialize_services_background())
    
    print("✅ Server is ready to accept requests!")
    print("💡 Use GET /health to check model loading status")

# --- Helper function to check if services are ready ---
def check_services_ready():
    """Check if required services are loaded"""
    if not loading_status["all_ready"]:
        ready_services = [k for k, v in loading_status.items() if v and k != "all_ready"]
        pending_services = [k for k, v in loading_status.items() if not v and k != "all_ready"]
        raise HTTPException(
            status_code=503, 
            detail=f"Services still loading. Ready: {ready_services}. Pending: {pending_services}. Please wait and try again."
        )

# --- Health Check Endpoint ---
@app.get("/health", response_model=ServiceStatus)
async def health_check():
    """Check the loading status of all services"""
    return ServiceStatus(
        models_loaded=loading_status["all_ready"],
        services_ready=loading_status["all_ready"],
        loading_progress=loading_status.copy()
    )

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
    key_points_prompt = f"Analyze this legal document and extract the 5-7 most critical key points or clauses that a person must know. Present them as a JSON array of strings. Each string should be a concise point. Respond with ```json ... ``` block:\n\n{text}"

    async def run_prompt(p):
        return await asyncio.to_thread(gemini_model.generate_content, p)

    summary_res, risk_res, glossary_res, key_points_res = await asyncio.gather(
        run_prompt(summary_prompt), 
        run_prompt(risk_prompt), 
        run_prompt(glossary_prompt),
        run_prompt(key_points_prompt)
    )

    summary = summary_res.text.strip()
    risks = clean_json_response(risk_res.text) or []
    glossary_data = clean_json_response(glossary_res.text) or {}
    glossary = [{"term": k, "definition": v} for k, v in glossary_data.items()]
    key_points = clean_json_response(key_points_res.text) or []
    
    # Validate key_points is a list of strings
    if not isinstance(key_points, list) or not all(isinstance(item, str) for item in key_points):
        key_points = []

    return {"summary": summary, "risk_alerts": risks, "glossary": glossary, "key_points": key_points}

# --- API Endpoints ---
@app.post("/upload", response_model=DocumentAnalysisResponse)
async def upload_document(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    check_services_ready()  # Ensure services are loaded
    
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

@app.post("/chat", response_model=ChatResponse)
async def chat_with_document(chat_request: ChatMessage, user: dict = Depends(get_current_user)):
    check_services_ready()  # Ensure services are loaded
    
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


@app.post("/translate")
async def translate_text(request: TranslationRequest):
    """Translate text using Gemini AI."""
    if not loading_status["gemini"]:
        raise HTTPException(status_code=503, detail="Gemini AI is still loading. Please try again in a moment.")
    
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
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)