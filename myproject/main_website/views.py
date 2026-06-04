import json
import os
from pathlib import Path

from django.shortcuts import render, redirect
from django.http import JsonResponse

from .forms import DocumentUploadForm
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import openai

ROOT_DIR = Path(__file__).resolve().parents[2]
FAISS_DIR = ROOT_DIR / "faiss_index"
GROUND_TRUTH_FILE = ROOT_DIR / "questions.json"
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if OPENAI_API_KEY:
#     openai.api_key = OPENAI_API_KEY

from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


EMBEDDINGS = None
VECTOR_STORE = None

from google import genai

client = genai.Client()
chat = client.chats.create(model="gemini-3.1-flash-lite")

# have to have another llm to judge the first model's responses
judge_chat = client.chats.create(model="gemini-3.5-flash")


from loguru import logger

logger.add("logs/main.logs")

EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


def get_embeddings():
    return EMBEDDINGS


def get_vector_store():
    global VECTOR_STORE
    if VECTOR_STORE is None:
        EMB = get_embeddings()
        VECTOR_STORE = FAISS.load_local(str(FAISS_DIR), EMB, allow_dangerous_deserialization=True)
    return VECTOR_STORE


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


def build_similarity_report(question: str, k: int = 5):
    vector_store = get_vector_store()
    docs_and_scores = vector_store.similarity_search_with_score(question, k=k)
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
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY must be set in the environment to generate answers.")

    context = "\n\n".join([f"Chunk {i + 1}:\n{doc['context']}" for i, doc in enumerate(retrieved_docs)])

    prompt = (
        "You are a contract QA assistant. Answer the question using only the provided context from the contract. "
        "Do not hallucinate or invent details. If the answer is not in the context, say 'I could not find the answer in the provided contract text.'\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )

    logger.info(f"Sending message to Gemini API: {prompt}")

    response = chat.send_message(prompt)

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
    answer = extract_response_text(response)
    logger.info(f"gemini answer : {answer}")
    return answer


def home(request):
    logger.info("Accessing home page")  
    return render(request, "home.html")


def upload_document(request):
    logger.info("Uploading document")
    if request.method == "POST":
        form = DocumentUploadForm(request.POST, request.FILES)
        if form.is_valid():
            request.session["document_loaded"] = True
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
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY must be set in the environment to run the judge.")

    prompt = (
        "You are a judge for contract QA system answers. You are evaluating a RAG system's answer against a ground truth answer extracted from a contract document. "
        "Compare the two answers and classify the result as exactly one of: Match, Partial Match, or No Match. "
        "Then provide a single sentence explaining your classification. Do not add any other commentary.\n\n"
        f"Ground Truth Answer: {expected_answer}\n"
        f"System Answer: {system_answer}"
    )

    response = judge_chat.send_message(prompt)

    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {"role": "system", "content": "You are a judge for contract QA system answers."},
    #         {"role": "user", "content": prompt},
    #     ],
    #     temperature=0.0,
    #     max_tokens=150,
    # )

    logger.info(f"Judge response: {response}")

    content = response.text.strip()
    if ":" in content:
        judgement, reason = content.split(":", 1)
        return judgement.strip(), reason.strip()

    return "No Match", content


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
        expected_answer = item.get("expected_answer", "").strip()

        if not expected_answer:
            results.append({
                "category": category,
                "judgement": "Missing",
                "reason": "Ground truth answer is empty for this question."
            })
            continue

        similarity_report = build_similarity_report(question, k=5)
        system_answer = generate_answer(question, similarity_report)
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
