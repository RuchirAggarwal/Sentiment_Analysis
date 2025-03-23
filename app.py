
import gradio as gr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import plotly.express as px
import os
import nltk
from nltk.tokenize import sent_tokenize

# Ensure NLTK resources are downloaded
nltk.download('punkt')

# Initialise VADER
analyzer = SentimentIntensityAnalyzer()


def get_sentiment_label(compound_score):
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def process_corpus(corpus):
    results, doc_scores = [], []
    for doc_idx, doc in enumerate(corpus, start=1):
        if doc.strip():
            try:
                sentences = sent_tokenize(doc.strip())
                doc_compound_score = 0
                for sent_idx, sentence in enumerate(sentences, start=1):
                    if sentence.strip():
                        scores = analyzer.polarity_scores(sentence)
                        scores_rounded = {k: round(v, 3) for k, v in scores.items()}
                        results.append({
                            'doc_ID': doc_idx,
                            'sent_ID': sent_idx,
                            'sentence': sentence,
                            'compound': scores_rounded['compound'],
                            'neg': scores_rounded['neg'],
                            'neu': scores_rounded['neu'],
                            'pos': scores_rounded['pos'],
                            'sentiment_label': get_sentiment_label(scores_rounded['compound'])
                        })
                        doc_compound_score += scores_rounded['compound']
                doc_scores.append({
                    'doc_ID': doc_idx,
                    'doc_senti_score': round(doc_compound_score, 3),
                    'doc_sentiment_label': get_sentiment_label(round(doc_compound_score, 3))
                })
            except Exception as e:
                return pd.DataFrame(), pd.DataFrame(), f"Error in doc {doc_idx}: {str(e)}"
    return pd.DataFrame(results), pd.DataFrame(doc_scores), None


def process_corpus_with_ids(corpus, doc_ids):
    results, doc_scores = [], []
    for doc_id, doc in zip(doc_ids, corpus):
        if doc.strip():
            try:
                sentences = sent_tokenize(doc.strip())
                doc_compound_score = 0
                for sent_idx, sentence in enumerate(sentences, start=1):
                    if sentence.strip():
                        scores = analyzer.polarity_scores(sentence)
                        scores_rounded = {k: round(v, 3) for k, v in scores.items()}
                        results.append({
                            'doc_ID': doc_id,
                            'sent_ID': sent_idx,
                            'sentence': sentence,
                            'compound': scores_rounded['compound'],
                            'neg': scores_rounded['neg'],
                            'neu': scores_rounded['neu'],
                            'pos': scores_rounded['pos'],
                            'sentiment_label': get_sentiment_label(scores_rounded['compound'])
                        })
                        doc_compound_score += scores_rounded['compound']
                doc_scores.append({
                    'doc_ID': doc_id,
                    'doc_senti_score': round(doc_compound_score, 3),
                    'doc_sentiment_label': get_sentiment_label(round(doc_compound_score, 3))
                })
            except Exception as e:
                return pd.DataFrame(), pd.DataFrame(), f"Error in doc {doc_id}: {str(e)}"
    return pd.DataFrame(results), pd.DataFrame(doc_scores), None


def analyze_text(user_input):
    sentence_df, doc_df, error = process_corpus(user_input.splitlines())
    return sentence_df, doc_df, error, doc_df if not error else None


def analyze_file(file, doc_id_col=None, text_col=None):
    try:
        file_ext = os.path.splitext(file.name)[1].lower()
        if file_ext == '.txt':
            with open(file.name, 'r', encoding='utf-8') as f:
                corpus = f.read().splitlines()
            return *process_corpus(corpus), None
        elif file_ext == '.csv':
            df = pd.read_csv(file.name)
            if doc_id_col not in df.columns or text_col not in df.columns:
                return pd.DataFrame(), pd.DataFrame(), "Invalid column names", None
            corpus = df[text_col].astype(str).tolist()
            doc_ids = df[doc_id_col].tolist()
            return *process_corpus_with_ids(corpus, doc_ids), None
        else:
            return pd.DataFrame(), pd.DataFrame(), "Unsupported file format", None
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), str(e), None


def update_file_ui(file):
    file_ext = os.path.splitext(file.name)[1].lower()
    try:
        if file_ext == '.csv':
            df = pd.read_csv(file.name)
            columns = df.columns.tolist()
            return df.head(), gr.update(choices=columns, visible=True), gr.update(choices=columns, visible=True), gr.update(visible=True)
        elif file_ext == '.txt':
            with open(file.name, 'r', encoding='utf-8') as f:
                preview = f.read().splitlines()[:5]
            return pd.DataFrame(preview, columns=["Text"]), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
    except:
        pass
    return pd.DataFrame(), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)


def generate_interactive_plot(doc_df):
    if doc_df.empty:
        return None
    fig = px.line(
        doc_df,
        x='doc_ID',
        y='doc_senti_score',
        title="Document-Level Sentiment Scores",
        labels={'doc_ID': 'Document ID', 'doc_senti_score': 'Sentiment Score'},
        markers=True
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="Neutral")
    return fig


with gr.Blocks() as demo:
    gr.Markdown("### Sentiment Analysis App with Multiple Input Options")

    with gr.Tab("Text Input"):
        text_input = gr.Textbox(label="Enter text", lines=5)
        button_text = gr.Button("Analyze Sentiment")
        sentence_df_text = gr.Dataframe(label="Sentence-Level Results")
        doc_df_text = gr.Dataframe(label="Document-Level Results")
        error_text = gr.Textbox(label="Error", interactive=False)
        plot_text = gr.Plot()

        def run_text_analysis(text):
            s_df, d_df, err, p_df = analyze_text(text)
            return s_df, d_df, err, generate_interactive_plot(p_df)

        button_text.click(run_text_analysis, inputs=text_input,
                          outputs=[sentence_df_text, doc_df_text, error_text, plot_text])

    with gr.Tab("File Upload"):
        file_input = gr.File()
        doc_col = gr.Dropdown(label="Doc ID Column", visible=False)
        text_col = gr.Dropdown(label="Text Column", visible=False)
        button_file = gr.Button("Run Analysis", visible=False)
        sentence_df_file = gr.Dataframe(label="Sentence-Level Results")
        doc_df_file = gr.Dataframe(label="Document-Level Results")
        error_file = gr.Textbox(label="Error", interactive=False)
        plot_file = gr.Plot()

        file_input.change(update_file_ui, inputs=file_input,
                          outputs=[gr.Dataframe(visible=False), doc_col, text_col, button_file])

        def run_file_analysis(file, doc_col, text_col):
            s_df, d_df, err, p_df = analyze_file(file, doc_col, text_col)
            return s_df, d_df, err, generate_interactive_plot(p_df)

        button_file.click(run_file_analysis, inputs=[file_input, doc_col, text_col],
                          outputs=[sentence_df_file, doc_df_file, error_file, plot_file])

if __name__ == "__main__":
    demo.launch()
