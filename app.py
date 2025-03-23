# app.py
import gradio as gr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import plotly.express as px
import os
import nltk
from nltk.tokenize import sent_tokenize
import sys
import socket

# NLTK setup
print("Downloading NLTK resources...")
try:
    nltk.download('punkt', quiet=False)
    print("NLTK resources downloaded successfully.")
except Exception as e:
    print(f"Failed to download NLTK resources: {str(e)}", file=sys.stderr)
    sys.exit(1)

# Port config (Colab sets this via environment variable)
PORT = int(os.environ.get("GRADIO_SERVER_PORT", 7860))

analyzer = SentimentIntensityAnalyzer()

def get_sentiment_label(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    return "Neutral"

def process_corpus(corpus):
    results, doc_scores = [], []
    for doc_idx, doc in enumerate(corpus, start=1):
        if doc.strip():
            try:
                sentences = sent_tokenize(doc)
                doc_score = 0
                for sent_idx, sentence in enumerate(sentences, start=1):
                    scores = analyzer.polarity_scores(sentence)
                    scores = {k: round(v, 3) for k, v in scores.items()}
                    results.append({
                        "doc_ID": doc_idx,
                        "sent_ID": sent_idx,
                        "sentence": sentence,
                        "compound": scores["compound"],
                        "neg": scores["neg"],
                        "neu": scores["neu"],
                        "pos": scores["pos"],
                        "sentiment_label": get_sentiment_label(scores["compound"])
                    })
                    doc_score += scores["compound"]
                doc_scores.append({
                    "doc_ID": doc_idx,
                    "doc_senti_score": round(doc_score, 3),
                    "doc_sentiment_label": get_sentiment_label(round(doc_score, 3))
                })
            except Exception as e:
                return pd.DataFrame(), pd.DataFrame(), f"Error in doc {doc_idx}: {e}"
    return pd.DataFrame(results), pd.DataFrame(doc_scores), None

def analyze_text(user_input):
    try:
        return (*process_corpus(user_input.splitlines()), None)
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), str(e), None

def generate_plot(doc_df):
    if doc_df.empty:
        return None
    fig = px.line(doc_df, x='doc_ID', y='doc_senti_score', title="Document-Level Sentiment Scores")
    fig.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="Neutral")
    return fig

with gr.Blocks() as demo:
    gr.Markdown("## Text Sentiment Analysis")
    with gr.Tab("Text Input"):
        text_input = gr.Textbox(lines=5, label="Enter Text")
        btn = gr.Button("Analyse")
        sentence_df = gr.Dataframe(label="Sentence-Level")
        doc_df = gr.Dataframe(label="Document-Level")
        error = gr.Textbox(label="Error", interactive=False)
        plot = gr.Plot()

        def handle_text_analysis(text):
            s_df, d_df, err, p_df = analyze_text(text)
            return s_df, d_df, err, generate_plot(p_df)

        btn.click(handle_text_analysis, inputs=text_input,
                  outputs=[sentence_df, doc_df, error, plot])

# ✅ Auto-launch with environment-based port
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=PORT)

