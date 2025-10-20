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
# Firebase Storage
import firebase_admin
from firebase_admin import credentials, storage
import google.generativeai as genai
# Upstash Redis for distributed job storage
from upstash_redis import Redis
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

class JobInitiateResponse(BaseModel):
    job_id: str
    status: str
    message: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: Optional[str] = None
    result: Optional[DocumentAnalysisResponse] = None
    error: Optional[str] = None

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
GENERATION_MODEL_NAME = "gemini-2.5-flash-lite"
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2" 
PINECONE_DIMENSION = 768

# Global service variables
firebase_app = None
bucket = None
pinecone_index = None
gemini_model = None
sentence_model = None
splade_encoder = None
cross_encoder = None
executor = ThreadPoolExecutor(max_workers=4)
redis_client = None

# Job storage (fallback to in-memory if Redis is not available)
job_storage: Dict[str, Dict[str, Any]] = {}
job_storage_lock = threading.Lock()

# Loading status tracking
loading_status = {
    "firebase": False,
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
def initialize_redis():
    """Initialize Upstash Redis client for distributed job storage"""
    global redis_client
    try:
        redis_url = os.getenv("UPSTASH_REDIS_REST_URL")
        redis_token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
        
        if not redis_url or not redis_token:
            print("⚠️  UPSTASH_REDIS_REST_URL or UPSTASH_REDIS_REST_TOKEN not found in environment")
            print("📝 Falling back to in-memory job storage (not recommended for production)")
            redis_client = None
            return False
        
        redis_client = Redis(url=redis_url, token=redis_token)
        
        # Test connection
        redis_client.set("test_connection", "ok", ex=5)
        test_value = redis_client.get("test_connection")
        
        print("✅ Upstash Redis connection established")
        return True
    except Exception as e:
        print(f"⚠️  Upstash Redis connection failed: {e}")
        print("📝 Falling back to in-memory job storage (not recommended for production)")
        redis_client = None
        return False

def initialize_firebase():
    """Initialize Firebase Admin SDK"""
    global firebase_app, bucket
    try:
        # Use service account key file
        cred_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY_PATH", "firebase-service-account.json")
        if not os.path.exists(cred_path):
            print(f"Warning: Firebase service account key not found at {cred_path}")
            return False

        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_path)
            firebase_app = firebase_admin.initialize_app(cred, {
                'storageBucket': os.getenv('FIREBASE_STORAGE_BUCKET')
            })

        bucket = storage.bucket()
        return True
    except Exception as e:
        print(f"Firebase initialization failed: {e}")
        return False

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
    global firebase_app, bucket, pinecone_index, gemini_model, sentence_model, splade_encoder, cross_encoder
    
    print("🚀 Starting background service initialization...")
    
    # Redis (for distributed job storage)
    try:
        initialize_redis()
    except Exception as e:
        print(f"Redis initialization failed: {e}")
    
    # Firebase Storage (optional, non-blocking)
    try:
        if initialize_firebase():
            update_loading_status("firebase", True)
            print("✅ Firebase Storage ready")
        else:
            update_loading_status("firebase", True)  # Mark as "ready" even if not configured
    except Exception as e:
        print(f"Firebase initialization failed: {e}")
        update_loading_status("firebase", True)  # Non-critical, mark as ready
    
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
    
    # Start periodic job cleanup (remove jobs older than 1 hour)
    asyncio.create_task(periodic_job_cleanup())
    
    print("✅ Server is ready to accept requests!")
    print("💡 Use GET /health to check model loading status")

async def periodic_job_cleanup():
    """Periodically clean up old completed/failed jobs"""
    while True:
        await asyncio.sleep(600)  # Run every 10 minutes
        try:
            # Redis jobs expire automatically with TTL, only clean in-memory storage
            if not redis_client:
                with job_storage_lock:
                    now = datetime.now()
                    jobs_to_delete = []
                    
                    for job_id, job_data in job_storage.items():
                        updated_at = datetime.fromisoformat(job_data.get('updated_at', job_data.get('created_at', datetime.now().isoformat())))
                        age_minutes = (now - updated_at).total_seconds() / 60
                        
                        # Delete completed/failed jobs older than 60 minutes
                        if job_data.get('status') in ['completed', 'failed'] and age_minutes > 60:
                            jobs_to_delete.append(job_id)
                    
                    for job_id in jobs_to_delete:
                        del job_storage[job_id]
                        print(f"🧹 Cleaned up old job: {job_id}")
                    
        except Exception as e:
            print(f"Error in job cleanup: {e}")

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
        loading_progress={
            "firebase_storage": loading_status["firebase"],
            "pinecone": loading_status["pinecone"],
            "gemini": loading_status["gemini"],
            "sentence_model": loading_status["sentence_model"],
            "splade_encoder": loading_status["splade_encoder"],
            "cross_encoder": loading_status["cross_encoder"],
            "all_ready": loading_status["all_ready"]
        }
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

def upload_to_firebase(file_content: bytes, filename: str) -> Optional[str]:
    """Upload file to Firebase Storage and return download URL"""
    if not bucket:
        print("Firebase bucket not initialized. Skipping upload.")
        return None
    try:
        # Create a unique filename to avoid conflicts
        unique_filename = f"{uuid.uuid4()}_{filename}"
        blob = bucket.blob(unique_filename)

        # Upload the file
        blob.upload_from_string(file_content, content_type='application/octet-stream')

        # Make the file publicly accessible (optional, depending on your needs)
        blob.make_public()

        # Return the public URL
        return blob.public_url
    except Exception as e:
        print(f"Firebase upload error: {e}")
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

# --- Job Management Functions ---
def update_job_status(job_id: str, status: str, progress: str = None, result: Dict[str, Any] = None, error: str = None):
    """Update job status in Redis or fallback to in-memory storage"""
    job_data = {
        'job_id': job_id,
        'status': status,
        'updated_at': datetime.now().isoformat()
    }
    
    if progress:
        job_data['progress'] = progress
    if result:
        job_data['result'] = json.dumps(result)  # Serialize result for Redis
    if error:
        job_data['error'] = error
    
    # Try Redis first
    if redis_client:
        try:
            # Get existing job data first
            existing_data_raw = redis_client.hgetall(f"job:{job_id}")
            
            # Decode bytes to strings if needed
            existing_data = {}
            if existing_data_raw:
                for key, value in existing_data_raw.items():
                    # Handle both string and bytes keys/values
                    k = key.decode('utf-8') if isinstance(key, bytes) else key
                    v = value.decode('utf-8') if isinstance(value, bytes) else value
                    existing_data[k] = v
            
            # Merge with new data
            existing_data.update(job_data)
            
            # Store each field individually using hmset
            redis_client.hmset(f"job:{job_id}", existing_data)
            redis_client.expire(f"job:{job_id}", 3600)  # 1 hour TTL
            
            print(f"✅ Updated job {job_id} in Redis with status: {status}")
            return
        except Exception as e:
            print(f"❌ Redis error in update_job_status: {e}, falling back to in-memory")
    
    # Fallback to in-memory storage
    with job_storage_lock:
        if job_id in job_storage:
            job_storage[job_id].update(job_data)
        else:
            job_storage[job_id] = job_data

def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve job status from Redis or fallback to in-memory storage"""
    # Try Redis first
    if redis_client:
        try:
            job_data_raw = redis_client.hgetall(f"job:{job_id}")
            
            if not job_data_raw:
                print(f"⚠️  Job {job_id} not found in Redis")
                # Try in-memory as fallback
                with job_storage_lock:
                    return job_storage.get(job_id)
            
            # Decode bytes to strings
            job_data = {}
            for key, value in job_data_raw.items():
                k = key.decode('utf-8') if isinstance(key, bytes) else key
                v = value.decode('utf-8') if isinstance(value, bytes) else value
                job_data[k] = v
            
            # Deserialize result if present
            if 'result' in job_data and job_data['result']:
                try:
                    job_data['result'] = json.loads(job_data['result'])
                except json.JSONDecodeError as e:
                    print(f"⚠️  Failed to decode result JSON: {e}")
                    pass
            
            print(f"✅ Retrieved job {job_id} from Redis: status={job_data.get('status')}")
            return job_data
            
        except Exception as e:
            print(f"❌ Redis error in get_job_status: {e}, falling back to in-memory")
    
    # Fallback to in-memory storage
    with job_storage_lock:
        job = job_storage.get(job_id)
        if job:
            print(f"✅ Retrieved job {job_id} from in-memory storage")
        else:
            print(f"⚠️  Job {job_id} not found in in-memory storage")
        return job

def create_job(job_id: str, initial_data: Dict[str, Any]):
    """Create a new job in Redis or in-memory storage"""
    job_data = {
        'job_id': job_id,
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat(),
        **initial_data
    }
    
    # Try Redis first
    if redis_client:
        try:
            # Serialize result if present
            if 'result' in job_data and job_data['result']:
                job_data['result'] = json.dumps(job_data['result'])
            
            # Use hmset to set all fields at once
            redis_client.hmset(f"job:{job_id}", job_data)
            redis_client.expire(f"job:{job_id}", 3600)  # 1 hour TTL
            
            print(f"✅ Created job {job_id} in Redis")
            return
        except Exception as e:
            print(f"❌ Redis error in create_job: {e}, falling back to in-memory")
    
    # Fallback to in-memory storage
    with job_storage_lock:
        job_storage[job_id] = job_data
        print(f"✅ Created job {job_id} in in-memory storage")

def delete_job(job_id: str) -> bool:
    """Delete a job from Redis or in-memory storage"""
    # Try Redis first
    if redis_client:
        try:
            deleted = redis_client.delete(f"job:{job_id}")
            print(f"✅ Deleted job {job_id} from Redis: {deleted > 0}")
            return deleted > 0
        except Exception as e:
            print(f"❌ Redis error in delete_job: {e}, falling back to in-memory")
    
    # Fallback to in-memory storage
    with job_storage_lock:
        if job_id in job_storage:
            del job_storage[job_id]
            print(f"✅ Deleted job {job_id} from in-memory storage")
            return True
        return False

async def process_document_background(job_id: str, file_content: bytes, filename: str, file_extension: str, user_id: str):
    """Background task to process document upload and analysis"""
    try:
        # Update status to processing
        update_job_status(job_id, "processing", "Extracting text from document...")
        
        # Extract text based on file type
        text_extractors = {
            '.pdf': extract_text_from_pdf, 
            '.docx': extract_text_from_docx, 
            '.txt': lambda c: c.decode('utf-8')
        }
        
        if file_extension not in text_extractors:
            update_job_status(job_id, "failed", error=f"Unsupported file type: {file_extension}")
            return
        
        text = text_extractors[file_extension](file_content)
        if not text.strip():
            update_job_status(job_id, "failed", error="Could not extract text from document.")
            return
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Update status
        update_job_status(job_id, "processing", "Chunking document text...")
        chunks = chunk_text(text)
        
        # Upload to Firebase (if configured)
        update_job_status(job_id, "processing", "Uploading file to storage...")
        file_url = None
        if bucket:
            file_url = await asyncio.to_thread(upload_to_firebase, file_content, filename)
        
        # Analyze with Gemini
        update_job_status(job_id, "processing", "Analyzing document with AI...")
        analysis_results = await analyze_with_gemini(text)
        
        # Store in Pinecone
        update_job_status(job_id, "processing", "Storing embeddings in vector database...")
        metadata = {
            'document_id': document_id,
            'filename': filename,
            'user_id': user_id,
            'file_url': file_url or '',
            'upload_date': datetime.now().isoformat(),
        }
        await asyncio.to_thread(store_in_pinecone, chunks, document_id, metadata)
        
        # Prepare final result
        result = {
            'document_id': document_id,
            'filename': filename,
            'upload_date': datetime.now().isoformat(),
            'file_url': file_url,
            **analysis_results
        }
        
        # Mark as completed
        update_job_status(job_id, "completed", "Document analysis completed successfully", result=result)
        print(f"✅ Job {job_id} completed successfully")
        
    except Exception as e:
        error_msg = f"Error processing document: {str(e)}"
        print(f"❌ Job {job_id} failed: {error_msg}")
        update_job_status(job_id, "failed", error=error_msg)

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
    risk_prompt = (
        f"Scan this legal text. Identify clauses that could cost money, limit rights, or cause harm. "
        f"List ONLY risks. For each risk, provide:\n"
        f"- 'title': A short, descriptive name for the risk (under 5-6 words)\n"
        f"- 'severity': One of 'HIGH', 'MEDIUM', 'LOW'\n"
        f"- 'description': A concise summary (under 15 words)\n"
        f"- 'suggested_action': A practical step to mitigate or address the risk (under 15 words)\n"
        f"- 'clause_text': The exact text/clause from the document where this risk appears (under 50 words). Get the exact clause from the document word to word don't make anything from your own side just keep this to the document.\n"
        f"Format as a JSON array of objects, each with these five fields. Respond with a single valid ```json ... ``` block:\n\n{text}"
    )
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
@app.post("/upload/initiate", response_model=JobInitiateResponse, status_code=202)
async def initiate_document_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...), 
    user: dict = Depends(get_current_user)
):
    """Initiate document upload and return job ID immediately (202 Accepted)"""
    check_services_ready()  # Ensure services are loaded
    
    # Read file content
    file_content = await file.read()
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    # Validate file type
    supported_types = ['.pdf', '.docx', '.txt']
    if file_extension not in supported_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job in storage (Redis or in-memory)
    create_job(job_id, {
        'status': 'pending',
        'progress': 'Job initiated, waiting to start processing...',
        'result': None,
        'error': None
    })
    
    # Verify job was created
    verify_job = get_job_status(job_id)
    if not verify_job:
        print(f"❌ WARNING: Job {job_id} was not found immediately after creation!")
    else:
        print(f"✅ Job {job_id} verified in storage after creation")
    
    # Schedule background processing
    background_tasks.add_task(
        process_document_background,
        job_id=job_id,
        file_content=file_content,
        filename=file.filename,
        file_extension=file_extension,
        user_id=user['user_id']
    )
    
    print(f"📋 Job {job_id} initiated for file: {file.filename}")
    
    return JobInitiateResponse(
        job_id=job_id,
        status="pending",
        message="Document upload initiated. Use the job_id to check status."
    )

# --- Enhanced status endpoint with better error messages ---
@app.get("/upload/status/{job_id}", response_model=JobStatusResponse)
async def get_upload_status(job_id: str, user: dict = Depends(get_current_user)):
    """Check the status of a document upload job"""
    print(f"🔍 Checking status for job: {job_id}")
    
    job = get_job_status(job_id)
    
    if not job:
        # Try to list all jobs for debugging
        if redis_client:
            try:
                keys = redis_client.keys("job:*")
                print(f"📋 Active jobs in Redis: {len(keys) if keys else 0}")
                if keys:
                    print(f"   Sample keys: {[k.decode('utf-8') if isinstance(k, bytes) else k for k in keys[:5]]}")
            except Exception as e:
                print(f"⚠️  Could not list Redis keys: {e}")
        else:
            with job_storage_lock:
                print(f"📋 Active jobs in memory: {len(job_storage)}")
                if job_storage:
                    print(f"   Sample keys: {list(job_storage.keys())[:5]}")
        
        raise HTTPException(
            status_code=404, 
            detail=f"Job {job_id} not found. It may have expired or never existed."
        )
    
    # Prepare response
    response = JobStatusResponse(
        job_id=job['job_id'],
        status=job['status'],
        progress=job.get('progress'),
        error=job.get('error')
    )
    
    # If completed, include the result
    if job['status'] == 'completed' and job.get('result'):
        response.result = DocumentAnalysisResponse(**job['result'])
    
    print(f"✅ Returning status for job {job_id}: {job['status']}")
    return response
# --- Legacy synchronous upload endpoint (kept for backward compatibility) ---
@app.post("/upload", response_model=DocumentAnalysisResponse)
async def upload_document(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    """Legacy synchronous upload endpoint - use /upload/initiate for production"""
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
    
    file_url = None
    if bucket:  # Check if Firebase is initialized
        file_url = await asyncio.to_thread(upload_to_firebase, file_content, file.filename)

    analysis_results = await analysis_task
    
    metadata = {
        'document_id': document_id, 'filename': file.filename, 'user_id': user['user_id'],
        'file_url': file_url or '', 'upload_date': datetime.now().isoformat(),
    }
    await asyncio.to_thread(store_in_pinecone, chunks, document_id, metadata)
    
    return DocumentAnalysisResponse(
        document_id=document_id, filename=file.filename,
        upload_date=datetime.now(), file_url=file_url, **analysis_results
    )

@app.delete("/upload/job/{job_id}")
async def delete_job_endpoint(job_id: str, user: dict = Depends(get_current_user)):
    """Delete a job from storage"""
    if delete_job(job_id):
        return {"message": f"Job {job_id} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Job not found")

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
        You are a legal assistant helping users understand their legal documents. Answer questions based primarily on the provided context from the legal document.

        Guidelines:
        - Answer questions directly from the document context when possible
        - If the user asks for advice or recommendations or your opinion, provide practical, general legal guidance based on the document content
        - Be clear and helpful in your responses
        - If information is not available in the context, clearly state this limitation
        - When giving advice, emphasize that you're not a substitute for professional legal counsel

        Context from the legal document:
        {context}

        User's Question:
        {chat_request.message}

        Response:
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

@app.get("/debug/redis-test")
async def test_redis():
    """Test Redis connection and operations"""
    if not redis_client:
        return {"error": "Redis client not initialized"}
    
    try:
        # Test basic operations
        test_key = "test:connection"
        redis_client.set(test_key, "working", ex=10)
        value = redis_client.get(test_key)
        
        # Test hash operations
        test_hash = "test:hash"
        redis_client.hmset(test_hash, {"field1": "value1", "field2": "value2"})
        redis_client.expire(test_hash, 10)
        hash_data = redis_client.hgetall(test_hash)
        
        return {
            "redis_connected": True,
            "test_value": value.decode('utf-8') if isinstance(value, bytes) else value,
            "hash_data": {
                k.decode('utf-8') if isinstance(k, bytes) else k: 
                v.decode('utf-8') if isinstance(v, bytes) else v 
                for k, v in hash_data.items()
            }
        }
    except Exception as e:
        return {"error": str(e), "redis_connected": False}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)