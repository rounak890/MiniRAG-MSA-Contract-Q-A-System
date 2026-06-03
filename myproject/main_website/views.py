from django.shortcuts import render, redirect
from django.http import JsonResponse
from .forms import DocumentUploadForm

# Create your views here.
def home(request):
    return render(request, 'home.html')

def upload_document(request):

    if request.method == "POST":

        form = DocumentUploadForm(
            request.POST,
            request.FILES
        )

        if form.is_valid():

            uploaded_file = request.FILES["document"]

            # Extract text
            # Chunk document
            # Create embeddings
            # Save to vector DB

            request.session["document_loaded"] = True

            return redirect("query_page")

    else:
        form = DocumentUploadForm()

    return render(
        request,
        "upload_document.html",
        {"form": form}
    )


def query_page(request):

    return render(
        request,
        "query_page.html"
    )


def ask_question(request):

    if request.method == "POST":

        question = request.POST.get("question")

        # retrieval pipeline
        # top_chunks = retrieve(question)

        similarity_report = [
            {
                "rank": 1,
                "location": "Page 6 Para 3",
                "score": 0.91,
                "signal": "Strong",
                "preview": "failure to meet deadlines..."
            }
        ]

        answer = """
        The penalty for missing a delivery
        deadline is 2% per week capped at 10%.
        """

        return JsonResponse({
            "answer": answer,
            "similarity_report": similarity_report
        })

    return JsonResponse({"error": "Invalid request"})


def evaluation_page(request):

    return render(
        request,
        "evaluation_page.html"
    )


def run_evaluation(request):

    results = [
        {
            "category": "Payment Terms",
            "judgement": "Match",
            "reason": "Correctly identified Net 30"
        }
    ]

    accuracy = 80

    return JsonResponse({
        "results": results,
        "accuracy": accuracy
    })