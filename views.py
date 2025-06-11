import os
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from transformers import pipeline
from PyPDF2 import PdfReader
from textblob import TextBlob

# Load summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        text += page.extract_text() or ''
    return text

def summarizer_view(request):
    summary = ""
    sentiment = ""
    original_text = ""
    if request.method == 'POST':
        text_input = request.POST.get('text', '').strip()
        pdf_file = request.FILES.get('pdf', None)

        if pdf_file:
            # Save the uploaded file
            fs = FileSystemStorage()
            filename = fs.save(pdf_file.name, pdf_file)
            pdf_path = fs.path(filename)

            with open(pdf_path, 'rb') as f:
                original_text = extract_text_from_pdf(f)

            os.remove(pdf_path)  # Clean up after reading
        else:
            original_text = text_input

        if original_text:
            # Summarize the text
            summary_result = summarizer(original_text, max_length=300, min_length=100, do_sample=False)
            summary = summary_result[0]['summary_text']

            # Sentiment analysis
            blob = TextBlob(summary)
            sentiment = blob.sentiment.polarity
            if sentiment > 0:
                sentiment = "Positive"
            elif sentiment < 0:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"

    return render(request, 'index.html', {
        'summary': summary,
        'sentiment': sentiment,
        'original_text': original_text
    })