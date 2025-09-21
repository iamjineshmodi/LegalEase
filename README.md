BACKEND for LegalEase using FASTAPI for Gemini Hackathon

# LegalEase
Simlpify your legal documents in 1 click

## BACKEND EXAMPLES AND HOW TO USE:
go to backend directory
```
    python3 main.py 
```



1. /short-summary
```
curl -X POST "http://localhost:8000/short-summary" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a legal contract. you can assume some things that are heppening in this document"
  }'
```

``` 
RESPONSE:

{"summary":"The provided text states only that the document is a legal contract.  Without the actual content of the contract, a meaningful legal analysis is impossible.  To provide a summary as requested, the specific terms and conditions of the contract are absolutely necessary.  The following is a template of what the analysis *would* contain if the contract text were provided:\n\n\n**Legal Document Analysis (Template)**\n\n**Type of Document:**  (This would be determined from the contract's content. Examples: Rental Lease Agreement, Loan Agreement, Terms of Service, Employment Contract, etc.)\n\n**Parties Involved:** (This section would name the specific parties involved,  e.g., \"Landlord: John Smith; Tenant: Jane Doe\" or \"Company: Acme Corp; User:  Robert Jones\").\n\n**Key Obligations:**  (This section would detail the key obligations of each party as explicitly stated in the contract. For example, a tenant's obligation to pay rent, maintain the property in good condition, or adhere to specific rules; a landlord's obligation to provide habitable premises, make necessary repairs, etc.)\n\n**Important Clauses:** (This would highlight any significant clauses, such as those pertaining to termination (e.g., notice periods, grounds for termination), penalties for breach of contract (e.g., late fees, liquidated damages), dispute resolution mechanisms (e.g., arbitration, mediation, litigation), and liability limitations.)\n\n\n**Potential Risks:** (This would identify any potential risks or negative consequences for either party,  such as the financial risks associated with a loan agreement, potential for eviction in a lease, or limitations on liability in a terms of service agreement).\n\n\n**Overall Purpose:** (This section would explain the core objective of the contract in plain language. For example, \"This rental agreement establishes the terms under which John Smith leases his property to Jane Doe,\" or \"This contract outlines the terms and conditions governing the use of Acme Corp's software by Robert Jones.\")\n\n\nIn short, providing a proper legal analysis requires the actual contract text.  The current prompt only offers a statement that a legal contract exists, rendering a substantive analysis impossible."}

```


2. /risk-summary
```
curl -X POST "http://localhost:8000/risk-analysis" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This legal contract establishes a commercial arrangement between the parties, but contains several clauses that may present material risks. The limitation of liability provision appears excessively broad, potentially leaving one party without meaningful remedies in the event of gross negligence or willful misconduct. The indemnification clause is unilateral, requiring only one party to assume extensive financial responsibility for third-party claims. Additionally, the termination rights are asymmetrical, permitting one party to exit the agreement with minimal notice while restricting the other party’s ability to do so. Confidentiality obligations are vaguely defined, creating uncertainty about the scope of protected information and potential exposure to liability. Finally, the absence of a clear dispute resolution mechanism may increase the likelihood of costly litigation."
  }'

```

``` 
RESPONSE:
{"risk_alerts":["⚠️ HIGH RISK: Limited legal recourse for negligence","⚠️ HIGH RISK: One-sided financial responsibility","⚠️ HIGH RISK: Unequal termination rights","🟡 MEDIUM RISK: Unclear confidentiality rules","🟡 MEDIUM RISK: No clear dispute resolution"]}
```

3. /each-paragraph-summaries
```
curl -X POST "http://localhost:8000/each-paragraph-summaries" \
  -H "Content-Type: application/json" \
  -d '{                                                                              
    "text": "This is a legal contract. you can assume some things that are heppening in this document"
  }'
```

``` 
RESPONSE:

{"paragraph_summaries":{"Paragraph 1":"Plain English: This is a legally binding agreement.  Why it matters: This means everything written here is enforceable by law; if either party breaks the agreement, there could be legal consequences."}}%  

```

4. /glossary-definitions
```
curl -X POST "http://localhost:8000/glossary-definitions" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a legal contract. The parties hereto acknowledge and agree that any representations, warranties, and covenants contained herein shall be binding and enforceable to the fullest extent permissible under applicable law. The obligations stipulated are subject to conditions precedent, and any waiver thereof must be executed in writing by the duly authorized signatories. In the event of a material breach, the non-defaulting party shall be entitled to equitable relief, including but not limited to specific performance and injunctive remedies. Furthermore, all indemnification provisions shall survive termination and shall inure to the benefit of the successors and assigns of the parties."
  }'
```

``` 
RESPONSE::
{"glossary":{"representations, warranties, and covenants":"promises and guarantees made in the contract","applicable law":"relevant laws","conditions precedent":"things that must happen before the contract is valid","waiver":"giving up a right","duly authorized signatories":"people officially allowed to sign","material breach":"a serious violation of the contract","non-defaulting party":"the person who didn't break the contract","equitable relief":"a fair solution","specific performance":"making someone do what the contract says","injunctive remedies":"court orders to stop someone from doing something","indemnification provisions":"parts of the contract about who pays if someone gets sued","shall survive termination":"will still apply even after the contract ends","successors and assigns":"people or companies that take over the contract from the original parties"}}
```
        

5. \translate

```
curl -X POST "http://localhost:8000/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a legal contract. The parties hereto acknowledge and agree that any representations, warranties, and covenants contained herein shall be binding and enforceable to the fullest extent permissible under applicable law. The obligations stipulated are subject to conditions precedent, and any waiver thereof must be executed in writing by the duly authorized signatories. In the event of a material breach, the non-defaulting party shall be entitled to equitable relief, including but not limited to specific performance and injunctive remedies. Furthermore, all indemnification provisions shall survive termination and shall inure to the benefit of the successors and assigns of the parties.",
    "target_language": "hindi"
  }'
```

``` 
RESPONSE:

{"translated_text":"यह एक विधिक अनुबंध है।  यहाँ पर पक्षकार स्वीकार करते हैं और सहमत होते हैं कि इसमें निहित कोई भी निरूपण, वारंटी और करार लागू कानून के अंतर्गत अधिकतम सीमा तक बाध्यकारी और प्रवर्तनीय होंगे।  निर्धारित दायित्व पूर्व शर्तों के अधीन हैं, और इनकी किसी भी छूट को विधिवत अधिकृत हस्ताक्षरकर्ताओं द्वारा लिखित में क्रियान्वित किया जाना चाहिए।  गंभीर उल्लंघन की स्थिति में, डिफ़ॉल्ट न करने वाले पक्ष को इक्विटेबल राहत का अधिकार होगा, जिसमें विशेष प्रदर्शन और निषेधात्मक उपचार शामिल हैं, लेकिन इन्हीं तक सीमित नहीं हैं।  इसके अलावा, सभी क्षतिपूर्ति प्रावधान समापन के बाद भी प्रभावी रहेंगे और पक्षकारों के उत्तराधिकारियों और अधिग्रहीताओं को लाभान्वित करेंगे।"}

```

show homepage with a big action towards uploading pdf/txt/docx

# MVP:
1. show short summary of document
2. high risk alerts
3. show para by para summary
4. legal glossary definitions

optional:
5. transaltion option to other indian languages

### keep these as 1 and 2 in direct show, others to click on dropbox type something and then show the rest.


The workflow in the Home screen like: 
Ask user to upload a pdf, or .txt file, parse it, and then send it to gemini api and give a short summary
