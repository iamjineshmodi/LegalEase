import React, { useState } from 'react';
import { Upload, FileText, AlertCircle, CheckCircle, Loader2, File } from 'lucide-react';
import * as mammoth from 'mammoth';

const LegalDocumentAnalyzer = () => {
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [documentText, setDocumentText] = useState('');
  const [analysis, setAnalysis] = useState('');
  const [error, setError] = useState('');
  const [currentStep, setCurrentStep] = useState('upload'); // upload, parse, analyze, complete

  // Import environment variables
  const GEMINI_API_KEY = process.env.GEMINI_API_KEY

  const LEGAL_ANALYSIS_PROMPT = `You are an expert legal analyst with extensive knowledge of various legal domains including contracts, rental agreements, and terms of service. Your task is to analyze the following legal document text and provide a clear, concise summary of its key legal points and implications.

The summary should be between 100 - 300 words and should cover the following aspects:

What type of document is this? (e.g., rental lease, loan agreement, terms of service)
Parties Involved: Identify the main parties entering into the agreement (e.g., landlord and tenant, lender and borrower).
Key Obligations: Outline the primary obligations and responsibilities of each party. For example, in a rental agreement, this could include the tenant's obligation to pay rent on time and the landlord's obligation to maintain the property.
Important Clauses: Highlight any critical clauses that could have significant legal or financial implications. This might include clauses related to termination, penalties, or dispute resolution.
Potential Risks: Identify any potential risks or pitfalls that a party should be aware of. For instance, in a loan contract, this could be high-interest rates or strict repayment terms.
Overall Purpose: Explain the overall purpose and intent of the document in simple terms, so that a non-legal professional can understand its significance.

Do NOT invent information. Only use what's in the document.

Here is the legal document text for your analysis:

`;

  const parseFile = async (file) => {
    const fileExtension = file.name.split('.').pop().toLowerCase();
    
    try {
      switch (fileExtension) {
        case 'txt':
          return await file.text();
        
        case 'pdf':
          // Note: This is a simplified approach. In production, you'd want server-side PDF parsing
          setError('PDF parsing requires server-side processing. Please convert to TXT or DOCX for now.');
          return null;
        
        case 'docx':
          const arrayBuffer = await file.arrayBuffer();
          const result = await mammoth.extractRawText({ arrayBuffer });
          return result.value;
        
        default:
          throw new Error('Unsupported file format. Please upload PDF, TXT, or DOCX files.');
      }
    } catch (error) {
      throw new Error(`Error parsing file: ${error.message}`);
    }
  };

  const analyzeWithGemini = async (text) => {
    const prompt = LEGAL_ANALYSIS_PROMPT + text;
    
    try {
      const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=${GEMINI_API_KEY}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          contents: [{
            parts: [{ text: prompt }]
          }]
        })
      });

      if (!response.ok) {
        throw new Error('Failed to analyze document with Gemini API');
      }

      const data = await response.json();
      return data.candidates[0].content.parts[0].text;
    } catch (error) {
      throw new Error(`Gemini API error: ${error.message}`);
    }
  };

  const handleFileUpload = async (event) => {
    const selectedFile = event.target.files[0];
    if (!selectedFile) return;

    setFile(selectedFile);
    setError('');
    setAnalysis('');
    setCurrentStep('parse');
    setIsUploading(true);

    try {
      // Parse the file
      const text = await parseFile(selectedFile);
      if (!text) {
        setIsUploading(false);
        setCurrentStep('upload');
        return;
      }

      setDocumentText(text);
      setCurrentStep('analyze');
      setIsAnalyzing(true);
      setIsUploading(false);

      // Analyze with Gemini
      const analysisResult = await analyzeWithGemini(text);
      setAnalysis(analysisResult);
      setCurrentStep('complete');
      setIsAnalyzing(false);

    } catch (error) {
      setError(error.message);
      setIsUploading(false);
      setIsAnalyzing(false);
      setCurrentStep('upload');
    }
  };

  const resetApp = () => {
    setFile(null);
    setDocumentText('');
    setAnalysis('');
    setError('');
    setCurrentStep('upload');
    setIsUploading(false);
    setIsAnalyzing(false);
  };

  const getStepStatus = (step) => {
    const stepOrder = ['upload', 'parse', 'analyze', 'complete'];
    const currentIndex = stepOrder.indexOf(currentStep);
    const stepIndex = stepOrder.indexOf(step);
    
    if (stepIndex < currentIndex) return 'completed';
    if (stepIndex === currentIndex) return 'current';
    return 'pending';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold text-gray-800 mb-4">
              Legal Document Analyzer
            </h1>
            <p className="text-lg text-gray-600">
              Upload your legal documents and get AI-powered analysis in seconds
            </p>
          </div>

          {/* Progress Steps */}
          <div className="mb-8">
            <div className="flex items-center justify-center space-x-8">
              {[
                { key: 'upload', label: 'Upload', icon: Upload },
                { key: 'parse', label: 'Parse', icon: FileText },
                { key: 'analyze', label: 'Analyze', icon: AlertCircle },
                { key: 'complete', label: 'Complete', icon: CheckCircle }
              ].map(({ key, label, icon: Icon }) => {
                const status = getStepStatus(key);
                return (
                  <div key={key} className="flex items-center space-x-2">
                    <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                      status === 'completed' ? 'bg-green-500 text-white' :
                      status === 'current' ? 'bg-blue-500 text-white' :
                      'bg-gray-200 text-gray-400'
                    }`}>
                      <Icon className="w-5 h-5" />
                    </div>
                    <span className={`text-sm font-medium ${
                      status === 'completed' ? 'text-green-600' :
                      status === 'current' ? 'text-blue-600' :
                      'text-gray-400'
                    }`}>
                      {label}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Main Content */}
          <div className="bg-white rounded-lg shadow-lg p-8">
            {currentStep === 'upload' && !isUploading && (
              <div className="text-center">
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-12 mb-6 hover:border-blue-400 transition-colors">
                  <Upload className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                  <h3 className="text-xl font-semibold text-gray-700 mb-2">
                    Upload Your Legal Document
                  </h3>
                  <p className="text-gray-500 mb-6">
                    Supported formats: PDF, TXT, DOCX (up to 10MB)
                  </p>
                  <label className="inline-block">
                    <input
                      type="file"
                      accept=".pdf,.txt,.docx"
                      onChange={handleFileUpload}
                      className="hidden"
                    />
                    <span className="bg-blue-500 hover:bg-blue-600 text-white font-medium py-3 px-8 rounded-lg cursor-pointer transition-colors">
                      Choose File
                    </span>
                  </label>
                </div>
                
                <div className="text-sm text-gray-500">
                  <p className="mb-2">Supported document types:</p>
                  <div className="flex justify-center space-x-6">
                    <span className="flex items-center"><File className="w-4 h-4 mr-1" /> Contracts</span>
                    <span className="flex items-center"><File className="w-4 h-4 mr-1" /> Leases</span>
                    <span className="flex items-center"><File className="w-4 h-4 mr-1" /> Terms of Service</span>
                    <span className="flex items-center"><File className="w-4 h-4 mr-1" /> Legal Agreements</span>
                  </div>
                </div>
              </div>
            )}

            {(isUploading || isAnalyzing) && (
              <div className="text-center py-12">
                <Loader2 className="w-12 h-12 text-blue-500 mx-auto mb-4 animate-spin" />
                <h3 className="text-xl font-semibold text-gray-700 mb-2">
                  {isUploading ? 'Parsing Document...' : 'Analyzing with AI...'}
                </h3>
                <p className="text-gray-500">
                  {isUploading ? 'Extracting text from your document' : 'Getting legal insights from Gemini AI'}
                </p>
              </div>
            )}

            {currentStep === 'complete' && analysis && (
              <div>
                <div className="mb-6">
                  <h3 className="text-xl font-semibold text-gray-800 mb-3 flex items-center">
                    <CheckCircle className="w-6 h-6 text-green-500 mr-2" />
                    Analysis Complete
                  </h3>
                  <div className="bg-gray-50 rounded-lg p-4 mb-4">
                    <p className="text-sm text-gray-600">
                      <strong>Document:</strong> {file?.name}
                    </p>
                    <p className="text-sm text-gray-600">
                      <strong>Size:</strong> {(file?.size / 1024).toFixed(1)} KB
                    </p>
                  </div>
                </div>

                <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 mb-6">
                  <h4 className="text-lg font-semibold text-blue-800 mb-3">
                    Legal Analysis Summary
                  </h4>
                  <div className="prose prose-blue max-w-none">
                    {analysis.split('\n').map((paragraph, index) => (
                      paragraph.trim() && (
                        <p key={index} className="mb-3 text-gray-700 leading-relaxed">
                          {paragraph}
                        </p>
                      )
                    ))}
                  </div>
                </div>

                <div className="text-center">
                  <button
                    onClick={resetApp}
                    className="bg-gray-500 hover:bg-gray-600 text-white font-medium py-2 px-6 rounded-lg transition-colors"
                  >
                    Analyze Another Document
                  </button>
                </div>
              </div>
            )}

            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
                <div className="flex items-center">
                  <AlertCircle className="w-5 h-5 text-red-500 mr-2" />
                  <span className="text-red-800 font-medium">Error</span>
                </div>
                <p className="text-red-700 mt-1">{error}</p>
                <button
                  onClick={resetApp}
                  className="mt-3 text-red-600 hover:text-red-700 underline text-sm"
                >
                  Try again
                </button>
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="text-center mt-8 text-gray-500 text-sm">
            <p>⚠️ This tool provides general analysis only. Always consult with a qualified attorney for legal advice.</p>
            <p className="mt-2">Remember to replace 'YOUR_GEMINI_API_KEY' with your actual Gemini API key.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LegalDocumentAnalyzer;