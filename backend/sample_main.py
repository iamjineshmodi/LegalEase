# python script to get rental.txt and parse jt and give back analysis




import os
import requests
import json
from typing import Optional
from pathlib import Path
import logging

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalDocumentAnalyzer:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Legal Document Analyzer
        
        Args:
            api_key: Gemini API key. If not provided, will look for GEMINI_API_KEY environment variable
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
        
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

        self.legal_analysis_prompt = """You are an expert legal analyst with extensive knowledge of various legal domains including contracts, rental agreements, and terms of service. Your task is to analyze the following legal document text and provide a clear, concise summary of its key legal points and implications.

The summary should be between 100 - 300 words and should cover the following aspects:

What type of document is this? (e.g., rental lease, loan agreement, terms of service)
Parties Involved: Identify the main parties entering into the agreement (e.g., landlord and tenant, lender and borrower).
Key Obligations: Outline the primary obligations and responsibilities of each party. For example, in a rental agreement, this could include the tenant's obligation to pay rent on time and the landlord's obligation to maintain the property.
Important Clauses: Highlight any critical clauses that could have significant legal or financial implications. This might include clauses related to termination, penalties, or dispute resolution.
Potential Risks: Identify any potential risks or pitfalls that a party should be aware of. For instance, in a loan contract, this could be high-interest rates or strict repayment terms.
Overall Purpose: Explain the overall purpose and intent of the document in simple terms, so that a non-legal professional can understand its significance.

Do NOT invent information. Only use what's in the document.

Here is the legal document text for your analysis:

"""

    def read_file(self, file_path: str) -> str:
        """
        Read text content from various file formats
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extracted text content
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == '.txt':
                return self._read_txt(file_path)
            elif file_extension == '.docx':
                return self._read_docx(file_path)
            elif file_extension == '.pdf':
                return self._read_pdf(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}. Supported formats: .txt, .docx, .pdf")
        
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            raise

    def _read_txt(self, file_path: Path) -> str:
        """Read text file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def _read_docx(self, file_path: Path) -> str:
        """Read DOCX file using python-docx"""
        try:
            from docx import Document
        except ImportError:
            raise ImportError("python-docx package is required for DOCX files. Install with: pip install python-docx")
        
        doc = Document(file_path)
        text_content = []
        
        for paragraph in doc.paragraphs:
            text_content.append(paragraph.text)
        
        return '\n'.join(text_content)

    def _read_pdf(self, file_path: Path) -> str:
        """Read PDF file using PyPDF2"""
        try:
            import PyPDF2
        except ImportError:
            raise ImportError("PyPDF2 package is required for PDF files. Install with: pip install PyPDF2")
        
        text_content = []
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page in pdf_reader.pages:
                text_content.append(page.extract_text())
        
        return '\n'.join(text_content)

    def analyze_document(self, text: str) -> str:
        """
        Analyze document text using Gemini API
        
        Args:
            text: Document text to analyze
            
        Returns:
            Analysis result from Gemini API
        """
        prompt = self.legal_analysis_prompt + text
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        
        try:
            logger.info("Sending request to Gemini API...")
            response = requests.post(
                f"{self.base_url}?key={self.api_key}",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if not response.ok:
                error_msg = f"Gemini API request failed with status {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise requests.RequestException(error_msg)
            
            response_data = response.json()
            
            # Extract the generated text from the response
            if 'candidates' in response_data and len(response_data['candidates']) > 0:
                candidate = response_data['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    return candidate['content']['parts'][0]['text']
            
            raise ValueError("Unexpected response format from Gemini API")
            
        except requests.RequestException as e:
            logger.error(f"Request to Gemini API failed: {str(e)}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini API response: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during API call: {str(e)}")
            raise

    def analyze_file(self, file_path: str) -> dict:
        """
        Complete workflow: read file and analyze with Gemini API
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing file info and analysis result
        """
        logger.info(f"Starting analysis of file: {file_path}")
        
        try:
            # Read the file
            logger.info("Reading file content...")
            text_content = self.read_file(file_path)
            
            if not text_content.strip():
                raise ValueError("File appears to be empty or contains no readable text")
            
            logger.info(f"Successfully read {len(text_content)} characters from file")
            
            # Analyze with Gemini API
            logger.info("Analyzing document with Gemini API...")
            analysis_result = self.analyze_document(text_content)
            
            logger.info("Analysis completed successfully!")
            
            return {
                'file_path': file_path,
                'file_size_chars': len(text_content),
                'analysis': analysis_result,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return {
                'file_path': file_path,
                'analysis': None,
                'error': str(e),
                'status': 'error'
            }


def main():
    """
    Example usage of the LegalDocumentAnalyzer
    """
    # Initialize analyzer (make sure to set GEMINI_API_KEY environment variable)
    try:
        analyzer = LegalDocumentAnalyzer()
    except ValueError as e:
        print(f"Error: {e}")
        print("Please add your Gemini API key to a .env file:")
        print("Create a .env file in your project directory with:")
        print("GEMINI_API_KEY=your_api_key_here")
        return
    
    # Example file path - update this to your actual file
    file_path = "/Users/jineshmodi/Documents/GitHub/LegalEase/examples/rental.txt"
    
    # Analyze the document
    result = analyzer.analyze_file(file_path)
    
    # Display results
    if result['status'] == 'success':
        print("=" * 60)
        print("LEGAL DOCUMENT ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"File: {result['file_path']}")
        print(f"Size: {result['file_size_chars']} characters")
        print()
        print("ANALYSIS:")
        print("-" * 40)
        print(result['analysis'])
        print("=" * 60)
    else:
        print(f"Analysis failed: {result['error']}")


if __name__ == "__main__":
    main()