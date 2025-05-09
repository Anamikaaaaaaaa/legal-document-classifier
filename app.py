from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import json

app = Flask(__name__)

# Using zero-shot classification as our demonstration custom model
custom_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_legal_document(text):
    candidate_labels = [
        "Contract", "Agreement", "Court Ruling", "Regulation", "Patent", "Legal Notice",
        "Memorandum", "Settlement", "Statute", "Judgment", "Deed", "Arbitration Award",
        "Policy Document", "License", "Compliance Report", "Tax Law", "Employment Agreement",
        "Privacy Policy", "Company Bylaws", "Intellectual Property Law"
    ]
    result = custom_classifier(text, candidate_labels, multi_label=True)
    top_indices = sorted(range(len(result["scores"])), key=lambda i: result["scores"][i], reverse=True)[:5]
    filtered_result = {
        "sequence": result["sequence"],
        "labels": [result["labels"][i] for i in top_indices],
        "scores": [round(result["scores"][i], 2) for i in top_indices]
    }
    return filtered_result

@app.route('/', methods=['GET', 'POST'])
def legal():
    result = None
    if request.method == 'POST':
        text = request.form['text']
        result = classify_legal_document(text)
    return render_template('legal.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
