from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import httpx
import PyPDF2
import pandas as pd
import docx
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
import io
import os
from datetime import datetime
import uuid
import redis
import json
from celery import Celery
import asyncio

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis connection
redis_client = redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379"),
    decode_responses=True
)

# Celery configuration
celery_app = Celery(
    "research_assistant",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379"),
    backend=os.getenv("REDIS_URL", "redis://localhost:6379")
)

# Groq API configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Semaphore for rate limiting
api_semaphore = asyncio.Semaphore(10)  # Max 10 concurrent API calls

class SessionData:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.research_pdfs = []
        self.research_texts = []
        self.analysis_data = None
        self.analysis_guide = None
        self.analysis_objectives = None
        self.research_responses = []
        self.analysis_steps = []
        self.analysis_results = []
    
    def to_dict(self):
        return {
            "session_id": self.session_id,
            "research_pdfs": self.research_pdfs,
            "research_texts": self.research_texts,
            "analysis_data": self.analysis_data.to_json() if self.analysis_data is not None else None,
            "analysis_guide": self.analysis_guide,
            "analysis_objectives": self.analysis_objectives,
            "research_responses": self.research_responses,
            "analysis_steps": self.analysis_steps,
            "analysis_results": self.analysis_results
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        session = cls(data["session_id"])
        session.research_pdfs = data.get("research_pdfs", [])
        session.research_texts = data.get("research_texts", [])
        
        if data.get("analysis_data"):
            session.analysis_data = pd.read_json(data["analysis_data"])
        
        session.analysis_guide = data.get("analysis_guide")
        session.analysis_objectives = data.get("analysis_objectives")
        session.research_responses = data.get("research_responses", [])
        session.analysis_steps = data.get("analysis_steps", [])
        session.analysis_results = data.get("analysis_results", [])
        
        return session

def get_session(session_id: str) -> SessionData:
    try:
        data = redis_client.get(f"session:{session_id}")
        if data:
            return SessionData.from_dict(json.loads(data))
    except Exception as e:
        print(f"Redis get error: {e}")
    
    return SessionData(session_id)

def save_session(session: SessionData):
    try:
        redis_client.setex(
            f"session:{session.session_id}",
            7200,  # 2 hours TTL
            json.dumps(session.to_dict())
        )
    except Exception as e:
        print(f"Redis save error: {e}")

def extract_pdf_text(file_bytes: bytes) -> str:
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_docx_text(file_bytes: bytes) -> str:
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join([para.text for para in doc.paragraphs])

async def call_groq(prompt: str, context: str = "", model: str = "llama-3.3-70b-versatile") -> str:
    """Call Groq API with rate limiting"""
    async with api_semaphore:
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional research assistant with expertise in academic writing, data analysis, and research methodology."
                },
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 8000
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(GROQ_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

# Celery task for background PDF processing
@celery_app.task
def process_pdf_background(session_id: str, filename: str, content_base64: str):
    import base64
    content = base64.b64decode(content_base64)
    text = extract_pdf_text(content)
    
    session = get_session(session_id)
    session.research_pdfs.append({
        "name": filename,
        "size": len(content),
        "processed": True
    })
    session.research_texts.append(f"File: {filename}\n{text}")
    save_session(session)
    
    return {"status": "completed", "filename": filename}

class ResearchTaskRequest(BaseModel):
    prompt: str
    session_id: str

class ApprovalRequest(BaseModel):
    step_id: str
    approved: bool
    session_id: str

class ReportRequest(BaseModel):
    type: str
    sections: List[str]
    session_id: str

@app.post("/research/upload")
async def upload_research_pdfs(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
):
    session_id = str(uuid.uuid4())
    session = get_session(session_id)
    uploaded_files = []
    
    for file in files:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(400, "Only PDF files are allowed")
        
        content = await file.read()
        
        # Process immediately for small files, background for large files
        if len(content) < 1_000_000:  # < 1MB
            text = extract_pdf_text(content)
            session.research_pdfs.append({
                "name": file.filename,
                "size": len(content),
                "processed": True
            })
            session.research_texts.append(f"File: {file.filename}\n{text}")
        else:
            # Process in background for large files
            import base64
            content_base64 = base64.b64encode(content).decode()
            background_tasks.add_task(process_pdf_background, session_id, file.filename, content_base64)
            session.research_pdfs.append({
                "name": file.filename,
                "size": len(content),
                "processed": False
            })
        
        uploaded_files.append({"name": file.filename, "size": len(content)})
    
    save_session(session)
    return {"files": uploaded_files, "session_id": session_id}

@app.post("/research/execute")
async def execute_research_task(request: ResearchTaskRequest):
    session = get_session(request.session_id)
    
    if not session.research_texts:
        raise HTTPException(400, "No research PDFs uploaded or still processing")
    
    # Combine all PDF texts for context (truncate if too long)
    context = "\n\n---\n\n".join(session.research_texts)
    
    # Truncate context if too long (keep within Groq's context window)
    max_context_chars = 100000  # ~25k tokens
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "\n\n[Context truncated...]"
    
    context_prompt = f"""You are a professional research assistant. You have access to the following research papers:

{context}

Based on these papers, please complete the following task. Use APA citation format when referencing the papers.

Task: {request.prompt}"""
    
    response = await call_groq(context_prompt)
    session.research_responses.append({"prompt": request.prompt, "response": response})
    save_session(session)
    
    return {"response": response}

@app.post("/analysis/upload")
async def upload_analysis_file(
    file: UploadFile = File(...),
    type: str = Form(...),
    session_id: str = Form(...)
):
    session = get_session(session_id)
    content = await file.read()
    
    if type == "data":
        if file.filename.endswith('.csv'):
            session.analysis_data = pd.read_csv(io.BytesIO(content))
        elif file.filename.endswith('.xlsx'):
            session.analysis_data = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(400, "Data file must be CSV or XLSX")
    
    elif type in ["guide", "objectives"]:
        if not file.filename.endswith(('.doc', '.docx')):
            raise HTTPException(400, "Guide and objectives must be DOC/DOCX files")
        text = extract_docx_text(content)
        if type == "guide":
            session.analysis_guide = text
        else:
            session.analysis_objectives = text
    
    save_session(session)
    return {"file": {"name": file.filename, "type": type}}

@app.post("/analysis/start")
async def start_analysis(session_id: str = Form(...)):
    session = get_session(session_id)
    
    if session.analysis_data is None or session.analysis_objectives is None:
        raise HTTPException(400, "Data and objectives files are required")
    
    # Prepare context
    data_summary = f"""Dataset Overview:
- Columns: {', '.join(session.analysis_data.columns)}
- Rows: {len(session.analysis_data)}
- Data types: {session.analysis_data.dtypes.to_dict()}
- Sample data (first 10 rows):
{session.analysis_data.head(10).to_string()}

Statistical summary:
{session.analysis_data.describe().to_string()}"""
    
    objectives_context = f"Research Objectives and Methods:\n{session.analysis_objectives}"
    
    guide_context = ""
    if session.analysis_guide:
        guide_context = f"\n\nFormat Guide (follow this structure):\n{session.analysis_guide}"
    
    # Ask Groq to create analysis plan
    planning_prompt = f"""{objectives_context}

{data_summary}
{guide_context}

Based on the objectives and methods described above, and the dataset provided, create a detailed step-by-step analysis plan. For each step:
1. Clearly state what analysis will be performed
2. Explain why this step is necessary
3. Describe what output will be generated

Provide EXACTLY 5 clear, actionable steps. Format each step as:
Step X: [Title]
Description: [What will be done]
Expected Output: [What will be produced]"""
    
    plan_response = await call_groq(planning_prompt)
    
    # Parse into steps
    steps = []
    lines = plan_response.split('\n')
    current_step = None
    
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.startswith('Step') and ':' in line_stripped:
            if current_step:
                steps.append(current_step)
            
            title_part = line_stripped.split(':', 1)[1].strip() if ':' in line_stripped else line_stripped
            current_step = {
                "id": str(uuid.uuid4()),
                "title": title_part[:100],
                "description": title_part,
                "completed": False
            }
        elif current_step and line_stripped and not line_stripped.startswith('Step'):
            current_step["description"] += " " + line_stripped
    
    if current_step:
        steps.append(current_step)
    
    # Ensure we have at least one step
    if not steps:
        steps = [{
            "id": str(uuid.uuid4()),
            "title": "Data Analysis",
            "description": "Perform comprehensive data analysis based on objectives",
            "completed": False
        }]
    
    session.analysis_steps = steps
    save_session(session)
    
    return {"steps": steps[:1] if steps else []}  # Return first step

@app.post("/analysis/approve")
async def approve_analysis_step(request: ApprovalRequest):
    session = get_session(request.session_id)
    
    if not request.approved:
        return {"status": "rejected"}
    
    # Find the current step
    step = next((s for s in session.analysis_steps if s["id"] == request.step_id), None)
    if not step:
        raise HTTPException(404, "Step not found")
    
    # Execute the step
    data_context = f"""Dataset Information:
Columns: {', '.join(session.analysis_data.columns)}
Shape: {session.analysis_data.shape}
Sample data:
{session.analysis_data.head(10).to_string()}

Previous analysis results:
{chr(10).join([r.get('summary', '') for r in session.analysis_results[-3:]])}
"""
    
    execution_prompt = f"""{data_context}

Execute this analysis step:
{step['description']}

Provide:
1. Detailed methodology used
2. Key findings and results
3. Statistical values (means, correlations, p-values, etc. as applicable)
4. Tables in text format if applicable
5. Clear interpretation of results

Keep the response comprehensive but concise (max 500 words)."""
    
    result = await call_groq(execution_prompt, model="llama-3.1-70b-versatile")
    
    session.analysis_results.append({
        "step_id": request.step_id,
        "summary": result
    })
    save_session(session)
    
    return {"result": result}

@app.post("/report/generate")
async def generate_report(request: ReportRequest):
    session = get_session(request.session_id)
    
    # Create Word document
    doc = docx.Document()
    
    # Title
    title = doc.add_heading(f"{request.type.capitalize()} Report", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Date
    doc.add_paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}")
    doc.add_paragraph()
    
    if request.type == "research":
        # Compile research responses
        for section in request.sections:
            doc.add_heading(section, 1)
            
            # Find relevant responses
            relevant = [r for r in session.research_responses if section.lower() in r['prompt'].lower()]
            if relevant:
                for resp in relevant:
                    doc.add_paragraph(resp['response'])
            else:
                # Generate content for this section using Groq
                context = "\n\n".join(session.research_texts[:3])  # Use first 3 PDFs
                prompt = f"Based on the research papers, write a comprehensive {section} section for an academic paper."
                
                try:
                    content = await call_groq(prompt, context)
                    doc.add_paragraph(content)
                except:
                    doc.add_paragraph(f"[Content for {section} section - please add manually]")
            
            doc.add_paragraph()
    
    else:  # analysis
        for section in request.sections:
            doc.add_heading(section, 1)
            
            if section == "Objectives":
                doc.add_paragraph(session.analysis_objectives or "[Objectives not provided]")
            elif section == "Data Description":
                if session.analysis_data is not None:
                    desc = f"Dataset contains {len(session.analysis_data)} observations across {len(session.analysis_data.columns)} variables.\n\n"
                    desc += f"Variables: {', '.join(session.analysis_data.columns)}"
                    doc.add_paragraph(desc)
            elif section in ["Methods", "Statistical Analysis", "Results", "Interpretation", "Tables & Figures"]:
                # Include analysis results
                for result in session.analysis_results:
                    doc.add_paragraph(result.get('summary', ''))
            else:
                doc.add_paragraph(f"[Content for {section} section]")
            
            doc.add_paragraph()
    
    # Save to bytes
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    
    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f"attachment; filename={request.type}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"}
    )

@app.post("/session/clear")
async def clear_session(session_id: str = Form(...)):
    try:
        redis_client.delete(f"session:{session_id}")
    except Exception as e:
        print(f"Redis delete error: {e}")
    
    return {"status": "cleared"}

@app.get("/health")
async def health_check():
    try:
        redis_client.ping()
        redis_status = "connected"
    except:
        redis_status = "disconnected"
    
    return {
        "status": "healthy",
        "redis": redis_status,
        "groq_api": "configured" if GROQ_API_KEY else "missing"
    }

@app.get("/")
async def root():
    return {"message": "AI Research Assistant API with Groq", "version": "2.0"}