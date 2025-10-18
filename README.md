# LegalEase Backend

A sophisticated FastAPI backend for AI-powered legal document analysis, featuring advanced RAG (Retrieval-Augmented Generation) capabilities, hybrid search, and multi-language support.

## Features

- **Document Processing**: Support for PDF, DOCX, and TXT file formats
- **Asynchronous Upload**: Background processing to handle long-running tasks (solves Vercel timeout issues)
- **AI Analysis**: Comprehensive document analysis using Google Gemini AI
  - Executive summaries
  - Risk assessment with suggested actions
  - Legal term glossary extraction
  - Key points identification
- **Advanced Search**: Hybrid vector search combining dense and sparse embeddings
- **Real-time Chat**: RAG-powered document Q&A with source citations and advisory capabilities
- **Cloud Storage**: Firebase Storage integration for reliable file storage
- **Multi-language Support**: Translation services for 10 Indian languages
- **Fast Startup**: Background model loading for immediate API availability
- **Job Queue System**: Track upload progress with polling-based status updates
- **Production Ready**: CORS enabled, authentication middleware, error handling

## Tech Stack

### Core Framework
- **FastAPI**: High-performance async web framework
- **Python 3.8+**: Modern Python with async/await support

### AI & ML
- **Google Gemini AI**: Generative AI for document analysis and chat
- **Sentence Transformers**: Local dense embeddings (`all-mpnet-base-v2`)
- **SPLADE**: Sparse vector encoding for hybrid search
- **Cross-Encoder**: Result re-ranking for improved relevance

### Vector Database & Search
- **Pinecone**: Managed vector database with hybrid search support
- **LangChain**: Advanced text chunking and splitting

### Cloud Services
- **Firebase Storage**: Reliable file storage with public URL access
- **Google OAuth2**: Secure authentication (for Drive if needed)

### Document Processing
- **PyMuPDF (fitz)**: High-performance PDF text extraction
- **python-docx**: Microsoft Word document processing

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Google Cloud Project with Drive API enabled
- Pinecone account and API key
- Google Gemini API key

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/iamjineshmodi/LegalEase.git
   cd LegalEase/backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration**

   Create a `.env` file in the backend directory:

   ```env
   # Google Services
   GEMINI_API_KEY=your_gemini_api_key_here

   # Firebase Configuration
   FIREBASE_SERVICE_ACCOUNT_KEY_PATH=firebase-service-account.json
   FIREBASE_STORAGE_BUCKET=your-project-id.appspot.com

   # Pinecone Configuration
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX_NAME=legalease-documents
   PINECONE_REGION=us-west-2
   ```

5. **Firebase Setup** (See [Firebase Setup Guide](ASYNC_UPLOAD_GUIDE.md) for details)
   - Create a Firebase project
   - Enable Firebase Storage
   - Download service account key JSON
   - Place it as `firebase-service-account.json` in the backend directory

## Important: Asynchronous Upload System

**⚠️ For Production (Vercel/Serverless)**: Use the asynchronous upload endpoints to avoid timeout issues.

The system supports **asynchronous background processing** for document uploads, which is essential for serverless deployments with timeout limitations (e.g., Vercel's 60-second limit).

### Quick Start (Async Upload)

```bash
# 1. Initiate upload (returns immediately with job_id)
curl -X POST "http://localhost:8000/upload/initiate" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@document.pdf"

# Response: {"job_id": "abc-123", "status": "pending", "message": "..."}

# 2. Poll for status (repeat every 3-5 seconds)
curl -X GET "http://localhost:8000/upload/status/abc-123" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Response includes status: pending, processing, completed, or failed
```

📚 **[Read the complete Async Upload Guide](ASYNC_UPLOAD_GUIDE.md)** for:
- Detailed API documentation
- Frontend implementation examples (JavaScript/React)
- Best practices and polling strategies
- Migration guide from synchronous to asynchronous

## Configuration

### Firebase Storage Setup

1. **Create Firebase Project**
   - Go to [Firebase Console](https://console.firebase.google.com/)
   - Create a new project

2. **Enable Storage**
   - Navigate to Storage in the Firebase console
   - Click "Get Started" and follow the setup

3. **Create Service Account**
   - Go to Project Settings > Service Accounts
   - Click "Generate New Private Key"
   - Save the JSON file as `firebase-service-account.json`

4. **Update Environment Variables**
   ```env
   FIREBASE_SERVICE_ACCOUNT_KEY_PATH=firebase-service-account.json
   FIREBASE_STORAGE_BUCKET=your-project-id.appspot.com
   ```

### Google Drive Setup (Legacy - Optional)

1. **Create Google Cloud Project**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing one

2. **Enable Drive API**
   - Navigate to "APIs & Services" > "Library"
   - Search for "Google Drive API" and enable it

3. **Create OAuth Credentials**
   - Go to "APIs & Services" > "Credentials"
   - Create "OAuth 2.0 Client ID"
   - Download the credentials JSON file
   - Run the OAuth flow to generate `token.json`

### Pinecone Setup

1. **Create Account**
   - Sign up at [Pinecone](https://www.pinecone.io/)

2. **Create Index**
   - Index name: `legalease-documents`
   - Dimension: `768` (matches the embedding model)
   - Metric: `dotproduct`
   - Environment: Choose your preferred region

### Model Configuration

The system uses several AI models that load in the background:

- **Embedding Model**: `all-mpnet-base-v2` (768 dimensions)
- **Sparse Encoder**: `naver/splade-cocondenser-ensembledistil`
- **Cross-Encoder**: `cross-encoder/ms-marco-minilm-l-6-v2`
- **Generation Model**: `gemini-2.5-flash-lite`

## API Endpoints

### Health Check
```http
GET /health
```

Returns loading status of all services and models.

**Response:**
```json
{
  "models_loaded": true,
  "services_ready": true,
  "loading_progress": {
    "firebase_storage": true,
    "pinecone": true,
    "gemini": true,
    "sentence_model": true,
    "splade_encoder": true,
    "cross_encoder": true,
    "all_ready": true
  }
}
```

### Document Upload (Async - Recommended for Production)

#### Initiate Upload
```http
POST /upload/initiate
Content-Type: multipart/form-data
Authorization: Bearer <token>
```

Returns immediately (202 Accepted) with a job ID for background processing.

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "pending",
  "message": "Document upload initiated. Use the job_id to check status."
}
```

#### Check Upload Status
```http
GET /upload/status/{job_id}
Authorization: Bearer <token>
```

Poll this endpoint to check processing status.

**Response (Processing):**
```json
{
  "job_id": "uuid-string",
  "status": "processing",
  "progress": "Analyzing document with AI...",
  "result": null,
  "error": null
}
```

**Response (Completed):**
```json
{
  "job_id": "uuid-string",
  "status": "completed",
  "progress": "Document analysis completed successfully",
  "result": {
    "document_id": "uuid-string",
    "filename": "contract.pdf",
    "upload_date": "2025-01-15T10:30:00",
    "summary": "Executive summary of the document...",
    "key_points": ["Point 1", "Point 2", "Point 3"],
    "risk_alerts": [
      {
        "title": "High Penalty Clause",
        "severity": "HIGH",
        "description": "Penalty exceeds industry standards",
        "suggested_action": "Negotiate penalty reduction or cap"
      }
    ],
    "glossary": [
      {"term": "Indemnification", "definition": "Legal protection against loss"}
    ],
    "file_url": "https://storage.googleapis.com/..."
  },
  "error": null
}
```

#### Delete Job
```http
DELETE /upload/job/{job_id}
Authorization: Bearer <token>
```

Clean up job data after retrieving results.

### Document Upload (Sync - Legacy)
```http
POST /upload
Content-Type: multipart/form-data
Authorization: Bearer <token>
```

**⚠️ Not recommended for production on Vercel** - Use `/upload/initiate` instead to avoid timeout issues.

Upload and analyze a legal document synchronously.

**Parameters:**
- `file`: Document file (PDF, DOCX, or TXT)

**Response:**
```json
{
  "document_id": "uuid-string",
  "filename": "contract.pdf",
  "upload_date": "2025-01-15T10:30:00",
  "summary": "Executive summary of the document...",
  "key_points": ["Point 1", "Point 2", "Point 3"],
  "risk_alerts": [
    {"severity": "HIGH", "description": "Penalty clause found"}
  ],
  "glossary": [
    {"term": "Indemnification", "definition": "Legal protection against loss"}
  ],
  "file_url": "https://storage.googleapis.com/..."
}
```

### Document Chat
```http
POST /chat
Content-Type: application/json
Authorization: Bearer <token>
```

Ask questions about a specific document using RAG.

**Request Body:**
```json
{
  "message": "What are the payment terms? Should I accept this clause?",
  "document_id": "uuid-string"
}
```

**Response:**
```json
{
  "response": "According to the document, payment terms require 30 days net. Given the late payment penalties, I recommend setting up automatic payments to avoid fees. However, you should consult with legal counsel before signing.",
  "sources": [
    {
      "chunk_index": 5,
      "text": "Payment shall be made within 30 days...",
      "relevance_score": 0.95
    }
  ]
}
```

### List Documents
```http
GET /documents
Authorization: Bearer <token>
```

Get all documents uploaded by the authenticated user.

**Response:**
```json
[
  {
    "document_id": "uuid-string",
    "filename": "contract.pdf",
    "upload_date": "2025-01-15T10:30:00"
  }
]
```

### Delete Document
```http
DELETE /documents/{document_id}
Authorization: Bearer <token>
```

Delete a document and all associated embeddings.

### Translation
```http
POST /translate
Content-Type: application/json
```

Translate text to supported Indian languages.

**Request Body:**
```json
{
  "text": "Legal contract terms",
  "target_language": "hindi"
}
```

**Response:**
```json
{
  "translated_text": "कानूनी अनुबंध की शर्तें"
}
```

### Supported Languages
```http
GET /supported-languages
```

Get list of supported translation languages.

## Usage Examples

### Python Client

```python
import requests
import time

# Check service health
response = requests.get("http://localhost:8000/health")
print(response.json())

# Async upload (recommended)
files = {"file": open("contract.pdf", "rb")}
headers = {"Authorization": "Bearer your-token"}

# 1. Initiate upload
response = requests.post(
    "http://localhost:8000/upload/initiate", 
    files=files, 
    headers=headers
)
job_data = response.json()
job_id = job_data["job_id"]

# 2. Poll for completion
while True:
    response = requests.get(
        f"http://localhost:8000/upload/status/{job_id}",
        headers=headers
    )
    status_data = response.json()
    
    print(f"Status: {status_data['status']} - {status_data.get('progress', '')}")
    
    if status_data['status'] == 'completed':
        doc_data = status_data['result']
        break
    elif status_data['status'] == 'failed':
        print("Error:", status_data['error'])
        break
    
    time.sleep(3)  # Wait 3 seconds

# 3. Chat with document
chat_data = {
    "message": "What are the key obligations?",
    "document_id": doc_data["document_id"]
}
response = requests.post(
    "http://localhost:8000/chat", 
    json=chat_data, 
    headers=headers
)
print(response.json())
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Async upload (recommended)
# 1. Initiate
curl -X POST "http://localhost:8000/upload/initiate" \
  -H "Authorization: Bearer your-token" \
  -F "file=@contract.pdf"

# 2. Check status (use job_id from response)
curl -X GET "http://localhost:8000/upload/status/{job_id}" \
  -H "Authorization: Bearer your-token"

# Sync upload (legacy - not recommended for production)
curl -X POST "http://localhost:8000/upload" \
  -H "Authorization: Bearer your-token" \
  -F "file=@contract.pdf"

# Chat with document
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-token" \
  -d '{
    "message": "Summarize the payment terms",
    "document_id": "your-document-id"
  }'
```

## Development

### Running Locally

```bash
# Start the server
python main.py

# Or with uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The server will start immediately and begin loading models in the background. Use the `/health` endpoint to monitor loading progress.

### Testing

```bash
# Run with pytest (if tests are added)
pytest

# Manual testing with requests
python -c "
import requests
resp = requests.get('http://localhost:8000/health')
print('Status:', resp.json())
"
```

### Code Structure

```
backend/
├── main.py                 # Main FastAPI application
├── requirements.txt        # Python dependencies
├── prompts/               # AI prompt templates
│   ├── chatbot.txt
│   ├── high_risk.txt
│   ├── legal_definitions.txt
│   ├── para_by_para.txt
│   ├── short_summary.txt
│   └── translate.txt
├── service_account.json   # Google service account (if used)
├── temp.py               # Temporary/test code
├── sample_main.py        # Alternative main file
└── __pycache__/         # Python cache
```

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables for Production

```env
# Server Configuration
PORT=8000
HOST=0.0.0.0

# Logging
LOG_LEVEL=INFO

# Model Configuration
EMBEDDING_MODEL=all-mpnet-base-v2
GENERATION_MODEL=gemini-1.5-flash

# Performance Tuning
MAX_WORKERS=4
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
```

### Production Checklist

- [ ] Set strong authentication tokens
- [ ] Configure proper CORS origins
- [ ] Set up monitoring and logging
- [ ] Configure backup strategies for Pinecone data
- [ ] Set up proper error tracking (Sentry, etc.)
- [ ] Configure rate limiting
- [ ] Set up health check monitoring
- [ ] Configure proper SSL/TLS certificates

## Security Considerations

⚠️ **Important Security Notes**

- JWT tokens are currently placeholder implementations
- Implement proper token validation for production use
- Google Drive integration requires secure OAuth flow
- API keys should be stored securely (not in code)
- Consider implementing rate limiting and request validation
- Regular security audits of dependencies recommended

## Performance Optimization

### Model Loading Strategy
- Models load asynchronously on startup for immediate API availability
- Use `/health` endpoint to check loading status before heavy operations

### Search Optimization
- Hybrid search combines dense and sparse retrieval
- Cross-encoder re-ranking improves result relevance
- Configurable chunk size and overlap for different document types

### Memory Management
- Models are loaded once and reused across requests
- ThreadPoolExecutor manages concurrent operations
- Consider GPU acceleration for better performance

## Troubleshooting

### Common Issues

**Models Still Loading**
```
HTTP 503: Services still loading. Ready: [...]. Pending: [...]
```
Wait for models to finish loading or check logs for errors.

**Pinecone Connection Failed**
- Verify API key and index configuration
- Check network connectivity
- Ensure index dimensions match embedding model (768)

**Google Drive Upload Failed**
- Verify OAuth credentials are valid
- Check token refresh status
- Ensure proper Drive API permissions

**Memory Issues**
- Reduce batch sizes in Pinecone operations
- Consider using smaller embedding models
- Monitor system resources during model loading

### Logs and Debugging

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check model loading progress:
```bash
curl http://localhost:8000/health
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section above
- Review the API documentation at `/docs` when running locally
