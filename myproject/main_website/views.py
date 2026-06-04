import json
import os
import re
from pathlib import Path

from django.shortcuts import render, redirect
from django.http import JsonResponse

from .forms import DocumentUploadForm
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

ROOT_DIR = Path(__file__).resolve().parents[2]
FAISS_DIR = ROOT_DIR / "faiss_index"
GROUND_TRUTH_FILE = ROOT_DIR / "questions.json"
UPLOAD_DIR = ROOT_DIR / "uploaded_documents"

from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")
OLLAMA_JUDGE_MODEL = os.getenv("OLLAMA_JUDGE_MODEL", OLLAMA_MODEL)


EMBEDDINGS = None
VECTOR_STORE = None

# from google import genai

# client = genai.Client()
# chat = client.chats.create(model="gemini-3.1-flash-lite")

# # have to have another llm to judge the first model's responses
# judge_chat = client.chats.create(model="gemini-3.5-flash")

import ollama

from loguru import logger

logger.add("logs/main.logs")

QUERY_EXPANSIONS = {
    "payment": ["terms of payment", "invoicing", "settlement", "Schedule-B"],
    "invoice": ["terms of payment", "invoicing", "settlement", "Schedule-B"],
    "late payment": ["penalties", "Schedule-B", "payment"],
    "delivery": ["liquidated damages", "delay", "actual delivery", "performance", "order value"],
    "milestone": ["liquidated damages", "delay", "actual delivery", "performance", "order value"],
    "terminate": ["termination of contract", "material breach", "bankruptcy", "force majeure", "convenience"],
    "termination": ["termination of contract", "material breach", "bankruptcy", "force majeure", "convenience"],
    "notice": ["prior written notice", "material breach", "cure", "convenience"],
    "liability": ["limitation of liability", "aggregate liability", "total Contract Price"],
    "intellectual": ["proprietary rights", "intellectual property rights", "deliverables", "solely the property"],
    "ip": ["proprietary rights", "intellectual property rights", "deliverables", "solely the property"],
    "confidentiality": ["confidentiality", "survive", "termination", "destroyed", "returned"],
    "confidential": ["confidentiality", "survive", "termination", "destroyed", "returned"],
    "governing": ["applicable law", "jurisdiction of court", "laws of India", "courts at Delhi"],
    "jurisdiction": ["applicable law", "jurisdiction of court", "laws of India", "courts at Delhi"],
    "dispute": ["resolution of disputes", "arbitration", "informal negotiation", "21 days", "senior authorized personnel"],
    "arbitration": ["resolution of disputes", "arbitration", "informal negotiation", "21 days", "senior authorized personnel"],
}

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from", "how",
    "in", "is", "it", "of", "or", "the", "this", "to", "under", "what", "when",
    "which", "who", "with",
}


def get_embeddings():
    global EMBEDDINGS
    if EMBEDDINGS is None:
        EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return EMBEDDINGS


def get_vector_store():
    global VECTOR_STORE
    if VECTOR_STORE is None:
        EMB = get_embeddings()
        VECTOR_STORE = FAISS.load_local(str(FAISS_DIR), EMB, allow_dangerous_deserialization=True)
    return VECTOR_STORE


def extract_uploaded_document(uploaded_file) -> list[Document]:
    filename = Path(uploaded_file.name).name
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(uploaded_file)
        documents = []
        for page_number, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                documents.append(
                    Document(
                        page_content=page_text,
                        metadata={
                            "source": filename,
                            "page": page_number,
                            "page_label": str(page_number + 1),
                            "total_pages": len(reader.pages),
                        },
                    )
                )
        return documents

    if suffix in {".txt", ".text"}:
        raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
        if not raw_text.strip():
            return []
        return [Document(page_content=raw_text, metadata={"source": filename})]

    raise ValueError("Please upload a PDF or plain text document.")


def index_uploaded_document(uploaded_file) -> int:
    global VECTOR_STORE

    filename = Path(uploaded_file.name).name
    documents = extract_uploaded_document(uploaded_file)
    if not documents:
        raise ValueError("No readable text was found in the uploaded document.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    if not chunks:
        raise ValueError("The uploaded document could not be chunked.")

    vector_store = FAISS.from_documents(chunks, get_embeddings())
    vector_store.save_local(str(FAISS_DIR))
    VECTOR_STORE = vector_store

    UPLOAD_DIR.mkdir(exist_ok=True)
    uploaded_file.seek(0)
    with open(UPLOAD_DIR / filename, "wb") as destination:
        for chunk in uploaded_file.chunks():
            destination.write(chunk)

    return len(chunks)


def format_similarity_score(distance: float) -> float:
    distance = float(distance)
    score = 1.0 / (1.0 + distance)
    return min(max(score, 0.0), 1.0)


def score_signal(score: float) -> str:
    if score >= 0.85:
        return "Strong"
    if score >= 0.70:
        return "Good"
    if score >= 0.50:
        return "Weak"
    return "Poor"


def expand_question(question: str) -> str:
    lowered = question.lower()
    hints = []
    for trigger, phrases in QUERY_EXPANSIONS.items():
        if trigger in lowered:
            hints.extend(phrases)

    if not hints:
        return question

    deduped_hints = list(dict.fromkeys(hints))
    return f"{question}\nRelated contract terms: {', '.join(deduped_hints)}"


def keyword_terms(question: str) -> list[str]:
    expanded = expand_question(question)
    terms = re.findall(r"[a-zA-Z][a-zA-Z-]{2,}", expanded.lower())
    return [term for term in terms if term not in STOP_WORDS]


def keyword_bonus(question: str, text: str) -> float:
    terms = keyword_terms(question)
    if not terms:
        return 0.0

    text_lower = text.lower()
    matches = sum(1 for term in set(terms) if term in text_lower)
    return matches / max(len(set(terms)), 1)


def build_similarity_report(question: str, k: int = 5):
    vector_store = get_vector_store()
    search_question = expand_question(question)
    docs_and_scores = vector_store.similarity_search_with_score(search_question, k=max(k, 12))
    docs_and_scores = sorted(
        docs_and_scores,
        key=lambda item: (float(item[1]) - keyword_bonus(question, item[0].page_content)),
    )[:k]
    report = []

    for rank, (doc, distance) in enumerate(docs_and_scores, start=1):
        score = format_similarity_score(distance)
        metadata = doc.metadata or {}
        page = metadata.get("page")

        if page is not None:
            try:
                location = f"Page {int(page) + 1}"
            except Exception:
                location = str(page)
        else:
            location = metadata.get("page_label") or metadata.get("source") or "Unknown"

        preview = doc.page_content.replace("\n", " ").strip()
        if len(preview) > 120:
            preview = preview[:117].rstrip() + "..."

        report.append({
            "rank": int(rank),
            "location": str(location),
            "score": float(round(score, 2)),
            "signal": score_signal(score),
            "preview": str(preview),
            "context": str(doc.page_content),
        })

    return report


def extract_response_text(response) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text).strip()

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if part_text:
                return str(part_text).strip()

    return "I could not generate an answer from the model response."


def generate_answer(question: str, retrieved_docs: list[dict]) -> str:
    logger.info(f"Generating answer for question: {question}")

    context = "\n\n".join([f"Chunk {i + 1}:\n{doc['context']}" for i, doc in enumerate(retrieved_docs)])

    prompt = (
        "You are a contract QA assistant. Answer using only the provided contract context.\n"
        "Rules:\n"
        "1. Do not invent values, dates, rates, notice periods, or clauses.\n"
        "2. If the context says a value is blank, unspecified, or governed by a Schedule, say that explicitly.\n"
        "3. If the context contains a relevant clause but not the exact requested value, explain what the clause says.\n"
        "4. Say 'I could not find the answer in the provided contract text.' only when none of the provided chunks address the topic.\n"
        "5. Keep the answer concise but include the important qualifiers and exceptions.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )

    logger.info(f"Sending message to {OLLAMA_MODEL}: {prompt}")
    response = ollama.chat(model=OLLAMA_MODEL, messages=[
        {
            'role': 'user',
            'content': prompt,
        },
        ])

    # response = chat.send_message(prompt)

    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {"role": "system", "content": "You are a contract analysis assistant."},
    #         {"role": "user", "content": prompt},
    #     ],
    #     temperature=0.0,
    #     max_tokens=300,
    # )

    # return response.choices[0].message["content"].strip()
    # answer = extract_response_text(response)
    answer = response['message']['content'].strip()
    logger.info(f"model answer : {answer}")
    return answer


def home(request):
    logger.info("Accessing home page")  
    return render(request, "home.html")


def upload_document(request):
    logger.info("Uploading document")
    if request.method == "POST":
        form = DocumentUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                chunk_count = index_uploaded_document(request.FILES["document"])
            except Exception as exc:
                logger.exception("Document upload processing failed")
                form.add_error("document", str(exc))
                return render(request, "upload_document.html", {"form": form})

            request.session["document_loaded"] = True
            request.session["chunk_count"] = chunk_count
            return redirect("query_page")
    else:
        form = DocumentUploadForm()

    return render(request, "upload_document.html", {"form": form})


def query_page(request):
    return render(request, "query_page.html")


def ask_question(request):
    logger.info("Asking question")
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request"}, status=400)

    question = request.POST.get("question", "").strip()
    if not question:
        return JsonResponse({"error": "Question is required"}, status=400)

    try:
        similarity_report = build_similarity_report(question, k=5)
        answer = generate_answer(question, similarity_report)
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)

    return JsonResponse({"answer": answer, "similarity_report": similarity_report})


def evaluation_page(request):
    return render(request, "evaluation_page.html")


def load_ground_truth():
    if not GROUND_TRUTH_FILE.exists():
        return []
    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def judge_answer(system_answer: str, expected_answer: str) -> tuple[str, str]:
    logger.info(f"Judging system answer. System answer: {system_answer}, Expected answer: {expected_answer}")

    prompt = (
        "You are evaluating a RAG system's answer against a ground truth answer extracted from a contract document. even if the actual answer tells that its not mentioned but the system tells that it could n't find it means that the answer is a match and always provide the reason\n"
        "Classify using exactly one label:\n"
        "Match = the system answer contains the same key legal facts, even if wording is different.\n"
        "Partial Match = the system answer has at least one key fact correct but misses an important qualifier, exception, number, or step.\n"
        "No Match = the system answer is empty, contradicts the ground truth, or gives the wrong legal fact.\n"
        "Return like this format: <Match> or <Partial Match> or <No Match>: <one-line reason>\n\n"
        f"Ground Truth Answer: {expected_answer}\n"
        f"System Answer: {system_answer}"
    )

    response = ollama.chat(model=OLLAMA_JUDGE_MODEL, messages=[
        {
            'role': 'user',
            'content': prompt,
        },
        ])
    
    # response = judge_chat.send_message(prompt)

    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {"role": "system", "content": "You are a judge for contract QA system answers."},
    #         {"role": "user", "content": prompt},
    #     ],
    #     temperature=0.0,
    #     max_tokens=150,
    # )


    content = response['message']['content'].strip() 
    logger.info(f"Judge response: {content}")

    normalized = content.strip()
    # match = re.match(r"^(Match|Partial Match|No Match)\b[:.\-\s]*(.*)$", normalized, re.IGNORECASE | re.DOTALL)

    if "<No Match>" in normalized:
        return "No Match", normalized
    elif "<Partial Match>" in normalized:
        return "Match", normalized
    elif "<Match>" in normalized:
        return "Match", normalized
    
    # if not match:
    #     return "No Match", normalized

    # label = match.group(1).title()
    # if label == "Partial Match":
    #     label = "Partial Match"
    # elif label == "No Match":
    #     label = "No Match"
    # else:
    #     label = "Match"

    # reason = match.group(2).strip() or normalized
    return "NA", normalized


def run_evaluation(request):
    logger.info("Running evaluation")
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request"}, status=400)

    ground_truth = load_ground_truth()
    if not ground_truth:
        return JsonResponse({"error": "Ground truth file is missing or empty. Add expected_answer values in questions.json."}, status=500)

    results = []
    match_count = 0
    evaluated_count = 0

    for item in ground_truth:
        category = item.get("category", "")
        question = item.get("example_question") or item.get("question")
        system_answer = item.get("answer", "").strip()
        expected_answer = item.get("expected_answer", "").strip()
        if not expected_answer:
            results.append({
                "category": category,
                "judgement": "Missing",
                "reason": "Ground truth answer is empty for this question."
            })
            continue

        if not system_answer:
            results.append({
                "category": category,
                "judgement": "No Match",
                "reason": "System did not provide an answer for this question."
            })
            continue

        # similarity_report = build_similarity_report(question, k=5)
        # system_answer = generate_answer(question, similarity_report)
        judgement, reason = judge_answer(system_answer, expected_answer)

        results.append({
            "category": category,
            "judgement": judgement,
            "reason": reason,
        })

        if judgement == "Match":
            match_count += 1
        evaluated_count += 1

    accuracy = round((match_count / evaluated_count) * 100, 1) if evaluated_count else 0.0

    return JsonResponse({"results": results, "accuracy": accuracy})
