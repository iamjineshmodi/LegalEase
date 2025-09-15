# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import google.generativeai as genai
import PyPDF2
import docx
import io
import os
from typing import List, Dict, Any
from pydantic import BaseModel
import json
import re
import uvicorn
from dotenv import load_dotenv

app = FastAPI(title="LegalEase API", version="1.0.0")

# CORS middleware for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini AI
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# print("Gemini API Key Set:", os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

class DocumentAnalysis(BaseModel):
    summary: str
    risk_alerts: List[Dict[str, Any]]
    paragraph_summaries: List[Dict[str, str]]
    glossary: List[Dict[str, str]]

class TranslationRequest(BaseModel):
    text: str
    target_language: str

# File processing utilities
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")

def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading DOCX: {str(e)}")

def extract_text_from_txt(file_bytes: bytes) -> str:
    """Extract text from TXT file"""
    try:
        return file_bytes.decode('utf-8')
    except UnicodeDecodeError:
        try:
            return file_bytes.decode('latin-1')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading TXT: {str(e)}")

# Gemini AI prompts
def create_summary_prompt(text: str) -> str:
    return f"""
    You are an expert legal analyst with extensive knowledge of various legal domains including contracts, rental agreements, and terms of service. Your task is to analyze the following legal document text and provide a clear, concise summary of its key legal points and implications.

    The summary should be between 100 - 300 words and should cover the following aspects:

    What type of document is this? (e.g., rental lease, loan agreement, terms of service)
    Parties Involved: Identify the main parties entering into the agreement (e.g., landlord and tenant, lender and borrower).
    Key Obligations: Outline the primary obligations and responsibilities of each party. For example, in a rental agreement, this could include the tenant's obligation to pay rent on time and the landlord's obligation to maintain the property.
    Important Clauses: Highlight any critical clauses that could have significant legal or financial implications. This might include clauses related to termination, penalties, or dispute resolution.
    Potential Risks: Identify any potential risks or pitfalls that a party should be aware of. For instance, in a loan contract, this could be high - interest rates or strict repayment terms.
    Overall Purpose: Explain the overall purpose and intent of the document in simple terms, so that a non - legal professional can understand its significance.

    Do NOT invent information. Only use what's in the document.

    Here is the legal document text for your analysis:
    {text}

    Summary:
    """

def create_risk_analysis_prompt(text: str) -> str:
    return f"""
    You are a risk analyst for everyday people. Scan this legal document and identify every clause that could cause harm, cost money, or limit rights — even if it seems small.

    List ONLY the risks — no explanations, no fluff. For each risk:

    - Use plain language (no legalese)
    - Label severity as: ⚠️ HIGH RISK, 🟡 MEDIUM RISK, or 🔶 LOW RISK
    - Keep each item under 15 words

    Format like this:

    ⚠️ HIGH RISK: [Brief description]  
    🟡 MEDIUM RISK: [Brief description]  
    🔶 LOW RISK: [Brief description]

    Rules:
    - Only include risks actually stated in the document.
    - If something is unclear, skip it — don't guess.
    - Do NOT list generic warnings like “laws may change.”
    - Focus on what affects the user directly: money, freedom, privacy, access, or penalties.

    Here is the document text:
    {text}
    
    Return only valid JSON array:
    """

def create_paragraph_by_paragraph_summary_prompt(text: str) -> str:
    return f"""
    You are a patient, empathetic legal guide for non-lawyers. Your task is to take this legal document paragraph by paragraph and translate each one into clear, simple, everyday language — while preserving all critical meaning.

    For EACH paragraph in the document:

    1. Quote the original text exactly (keep it short — max 1-2 sentences per quote).
    2. Explain it in plain English: Rewrite it as if you're talking to a friend who has never read a contract. Use no legal jargon.
    3. Why it matters: Explain what this means for the person reading it — their rights, obligations, money, privacy, or freedom.
    4. Hidden risk or trap: What could go wrong if they don't understand this? (e.g., automatic renewal, fees, loss of rights, liability)
    5. Actionable tip: What should they DO right now? (e.g., “Save this email,” “Call them before signing,” “Ask for this in writing”)

    Format your response in a JSON RESPONSE AS THIS:

    --- {{
    "Paragraph 1" : "Plain English: [Your clear rewrite] + Why it matters: [Impact on user]  + Any Risk/Trap: [What they might miss]  + Tip: [One concrete next step]" , 

    "Paragraph 2" : "Plain English: [Your clear rewrite] + Why it matters: [Impact on user]  + Any Risk/Trap: [What they might miss]  + Tip: [One concrete next step]"  
    }}

    Rules:
    - Do NOT summarize the whole document. Go paragraph by paragraph.
    - Do NOT invent terms or assume outside knowledge. Only use what's written.
    - If a paragraph is irrelevant (e.g., boilerplate jurisdiction clause), say: “This is standard legal filler — it doesn't affect your day-to-day rights.”
    - Keep explanations detailed but concise — aim for 50-100 words per paragraph.

    Here is the full document text:{text}
    
    Return only valid JSON array:
    """

def create_glossary_prompt(text: str) -> str:
    return f"""
    You are a plain-language legal translator. Your task is to scan this document and create a JSON object that maps every unique legal term or jargon phrase — exactly as it appears in the text — to its simplest, everyday English meaning.

    Rules:
    - ONLY include words or phrases that are actual legal terms used in the document.
    - DO NOT include common words like “party,” “agreement,” “date,” or “sign” unless they have a specific legal meaning in context (e.g., “indemnify”).
    - DO NOT repeat any term — each key must be unique.
    - DO NOT invent definitions. Use only what's implied by context in the document.
    - Use simple, conversational language — explain like you’re talking to someone who never read a contract.
    - Output MUST be valid JSON with this structure: {{ "legal_word": "simple meaning", ... }}
    - If no legal terms are found, return an empty object: {{}}
    - Do not add comments, explanations, or markdown.

    Examples of good mappings:
    "liquidated damages": "a fixed fee you pay if you break the contract"
    "automatic renewal": "the contract renews itself unless you cancel in time"
    "indemnify": "you promise to pay if someone else gets sued because of this agreement"

    Here is the document text: {text}

    Return only valid JSON array:
    """

def create_translation_prompt(text: str, language: str) -> str:
    return f"""
    Translate the following legal document summary to {language}.
    Maintain the legal context and meaning while making it accessible.
    
    Text to translate: {text}
    
    Translation:
    """

# API Routes
@app.get("/")
async def root():
    return {"message": "LegalEase API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

class SummaryRequest(BaseModel):
    text: str

@app.post("/short-summary") 
async def short_summary(request: SummaryRequest):
    try:
        summary_response = model.generate_content(create_summary_prompt(request.text))
        summary = summary_response.text.strip()
        return {"summary": summary}
    except Exception as e:  
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

@app.post("/risk-analysis")
async def risk_analysis(request: SummaryRequest): 
    try:
        risk_response = model.generate_content(create_risk_analysis_prompt(request.text))
        risk_text = risk_response.text.strip()
        
        # Clean and parse JSON response
        risk_text = re.sub(r'^```json\s*', '', risk_text)
        risk_text = re.sub(r'\s*```$', '', risk_text)
        
        try:
            risk_alerts = json.loads(risk_text)
        except json.JSONDecodeError:
            risk_alerts = [{"level": "medium", "title": "Unable to parse risk analysis", "description": risk_text}]
        
        return {"risk_alerts": risk_alerts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating risk analysis: {str(e)}")
    

@app.post("/each-paragraph-summaries")
async def paragraph_summaries(request: SummaryRequest):
    try:
        para_response = model.generate_content(create_paragraph_by_paragraph_summary_prompt(request.text))
        para_text = para_response.text.strip()
        
        para_text = re.sub(r'^```json\s*', '', para_text)
        para_text = re.sub(r'\s*```$', '', para_text)
        
        try:
            paragraph_summaries = json.loads(para_text)
        except json.JSONDecodeError:
            paragraph_summaries = [{"paragraph": "Unable to parse", "summary": para_text}]
        
        return {"paragraph_summaries": paragraph_summaries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating paragraph summaries: {str(e)}")


@app.post("/glossary-definitions")
async def glossary_definitions(request: SummaryRequest):
    try:
        glossary_response = model.generate_content(create_glossary_prompt(request.text))
        glossary_text = glossary_response.text.strip()
        
        glossary_text = re.sub(r'^```json\s*', '', glossary_text)
        glossary_text = re.sub(r'\s*```$', '', glossary_text)
        
        try:
            glossary = json.loads(glossary_text)
        except json.JSONDecodeError:
            glossary = [{"term": "Parsing Error", "definition": glossary_text}]
        
        return {"glossary": glossary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating glossary: {str(e)}")




@app.post("/translate")
async def translate_text(request: TranslationRequest):
    """
    Translate text to specified Indian language
    """
    language_map = {
        "hindi": "Hindi",
        "bengali": "Bengali", 
        "tamil": "Tamil",
        "telugu": "Telugu",
        "marathi": "Marathi",
        "gujarati": "Gujarati",
        "kannada": "Kannada",
        "malayalam": "Malayalam",
        "punjabi": "Punjabi",
        "urdu": "Urdu"
    }
    
    target_lang = language_map.get(request.target_language.lower())
    if not target_lang:
        raise HTTPException(status_code=400, detail="Language not supported")
    
    try:
        translation_response = model.generate_content(
            create_translation_prompt(request.text, target_lang)
        )
        
        return {"translated_text": translation_response.text.strip()}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")

@app.get("/supported-languages")
async def get_supported_languages():
    """
    Get list of supported languages for translation
    """
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
    uvicorn.run(app, host="0.0.0.0", port=8000)