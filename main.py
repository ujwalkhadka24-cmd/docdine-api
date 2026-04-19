# ============================================================
#  DocDine — Contract Extraction API
#  Hosted at: https://api.docdine.com
#  Frontend:  https://www.docdine.com
# ============================================================
#
#  Install dependencies:
#    pip install fastapi uvicorn python-multipart pydantic
#                pdfplumber python-docx pytesseract Pillow
#                anthropic python-dotenv
#
#  Run locally:
#    uvicorn main:app --reload --port 8000
#
#  Environment variables (.env):
#    ANTHROPIC_API_KEY=sk-ant-...
#    ALLOWED_ORIGINS=https://www.docdine.com,http://localhost:5173
# ============================================================

import os, uuid, time, tempfile, traceback
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import pdfplumber
from docx import Document as DocxDocument
from PIL import Image
import pytesseract
import anthropic

# ── App setup ───────────────────────────────────────────────
app = FastAPI(
    title="DocDine Extraction API",
    description="Extracts structured JSON from messy contract documents.",
    version="1.0.0",
)

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "https://www.docdine.com,http://localhost:5173,http://localhost:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store (swap for Redis/DB in production)
jobs: dict[str, dict] = {}

# ── Pydantic schemas ─────────────────────────────────────────
class Party(BaseModel):
    role: str
    name: str
    confidence: float = Field(ge=0, le=1)

class ContractValue(BaseModel):
    amount: float | None = None
    currency: str = "AUD"

class Clause(BaseModel):
    type: str
    text: str
    confidence: float = Field(ge=0, le=1)

class ExtractionResult(BaseModel):
    document_id: str
    processed_at: str
    confidence_score: float = Field(ge=0, le=1)
    parties: list[Party] = []
    effective_date: str | None = None
    expiry_date: str | None = None
    contract_value: ContractValue | None = None
    jurisdiction: str | None = None
    clauses: list[Clause] = []
    flags: list[str] = []
    extraction_method: str = "llm+regex"

class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "processing", "done", "failed"]
    progress: float = 0.0   # 0.0 – 1.0
    result: ExtractionResult | None = None
    error: str | None = None

# ── Text extraction helpers ──────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".doc", ".docx", ".png", ".jpg", ".jpeg", ".tiff", ".tif"}

def extract_text_from_pdf(path: str) -> str:
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    return "\n\n".join(text_parts)

def extract_text_from_docx(path: str) -> str:
    doc = DocxDocument(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def extract_text_from_image(path: str) -> str:
    img = Image.open(path)
    return pytesseract.image_to_string(img, lang="eng")

def extract_text(path: str, suffix: str) -> str:
    suffix = suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    if suffix in (".doc", ".docx"):
        return extract_text_from_docx(path)
    if suffix in (".png", ".jpg", ".jpeg", ".tiff", ".tif"):
        return extract_text_from_image(path)
    raise ValueError(f"Unsupported file type: {suffix}")

# ── LLM extraction ───────────────────────────────────────────
SYSTEM_PROMPT = """You are a contract analysis engine for DocDine (www.docdine.com).
Extract structured data from contract text and return ONLY valid JSON — no markdown, no preamble.

Return this exact structure:
{
  "parties": [{"role": "client|vendor|...", "name": "...", "confidence": 0.0-1.0}],
  "effective_date": "YYYY-MM-DD or null",
  "expiry_date": "YYYY-MM-DD or null",
  "contract_value": {"amount": number_or_null, "currency": "AUD|USD|GBP|..."},
  "jurisdiction": "string or null",
  "clauses": [
    {"type": "PAYMENT_TERMS|TERMINATION|CONFIDENTIALITY|INTELLECTUAL_PROPERTY|INDEMNITY|DISPUTE_RESOLUTION|OTHER",
     "text": "concise summary of the clause",
     "confidence": 0.0-1.0}
  ],
  "flags": ["list of notable risks or unusual terms"],
  "overall_confidence": 0.0-1.0
}

Be conservative with confidence scores. Flag unusual clauses, auto-renewal provisions, penalty clauses, and one-sided terms."""

def call_llm(text: str) -> dict:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Extract contract data from this document:\n\n{text[:12000]}"}],
    )
    raw = message.content[0].text.strip()
    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    import json
    return json.loads(raw)

# ── Background processing ────────────────────────────────────
def process_job(job_id: str, file_path: str, suffix: str):
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["progress"] = 0.1

        # Step 1: extract raw text
        raw_text = extract_text(file_path, suffix)
        jobs[job_id]["progress"] = 0.4

        if not raw_text.strip():
            raise ValueError("No text could be extracted from this document.")

        # Step 2: call LLM
        jobs[job_id]["progress"] = 0.5
        extracted = call_llm(raw_text)
        jobs[job_id]["progress"] = 0.9

        # Step 3: build result
        result = ExtractionResult(
            document_id=f"CTR-{job_id[:8].upper()}",
            processed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            confidence_score=extracted.get("overall_confidence", 0.85),
            parties=[Party(**p) for p in extracted.get("parties", [])],
            effective_date=extracted.get("effective_date"),
            expiry_date=extracted.get("expiry_date"),
            contract_value=ContractValue(**extracted["contract_value"]) if extracted.get("contract_value") else None,
            jurisdiction=extracted.get("jurisdiction"),
            clauses=[Clause(**c) for c in extracted.get("clauses", [])],
            flags=extracted.get("flags", []),
            extraction_method="llm+ocr" if suffix in (".png", ".jpg", ".jpeg", ".tiff", ".tif") else "llm+parser",
        )

        jobs[job_id]["status"] = "done"
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["result"] = result.model_dump()

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        print(f"[DocDine] Job {job_id} failed: {traceback.format_exc()}")
    finally:
        # Clean up temp file
        try:
            os.unlink(file_path)
        except Exception:
            pass

# ── Routes ───────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Used by the frontend to show API online/offline status."""
    return {"status": "ok", "service": "api.docdine.com"}


@app.post("/extract")
async def extract_contract(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a contract document. Returns a job_id immediately.
    Poll GET /jobs/{job_id} to check progress and retrieve the result.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Accepted: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    # Save to temp file so background task can access it after request ends
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    content = await file.read()
    tmp.write(content)
    tmp.flush()
    tmp.close()

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "result": None,
        "error": None,
    }

    background_tasks.add_task(process_job, job_id, tmp.name, suffix)

    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}", response_model=JobStatus)
def get_job_status(job_id: str):
    """Poll this endpoint to get processing progress and the final result."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    """Clean up a finished job from memory."""
    jobs.pop(job_id, None)
    return {"deleted": job_id}


# ── Dev entrypoint ───────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
