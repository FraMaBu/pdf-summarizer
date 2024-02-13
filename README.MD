# PDF Summary Generator - Streamlit App 📚🚀

## Overview 🌟

Welcome to the Generative AI powered **PDF Summary Generator**! This `Streamlit` application is your go-to tool for generating quick and concise summaries from PDF documents. Whether you're a student 🎓, researcher 🔬, or a busy professional 🏢, the app makes it a breeze to extract the essence of lengthy documents. Powered by the custom [genaipy Python library](https://github.com/FraMaBu/genaipy-lib), get ready to transform how you process information!

## How It Works 🔍

The app leverages a [Map-Reduce](https://js.langchain.com/docs/modules/chains/document/map_reduce) approach to distill text from PDFs, comprising:

1. 🗺️ **Map Step**: Extracts text from each PDF page, creating individual page summaries.
2. 📉 **Reduce Step**: Merges these page summaries into a final, comprehensive document summary.

To use the app, just follow these steps:

1. 📤 **Upload PDF**: Upload any PDF file.
2. 📄 **Select Pages**: Select a specific page range from the document for targeted summarization.
3. ⚙️ **Customize Settings (Optional)**: Tweak settings like the summary mode and word counts for summaries.
4. 🌈 **Generate Summary**: Click the `Generate Summary` button to start the summarization process.
5. 💾 **Download Summary**: Grab your final summary in `.txt` or `.md` formats.

## Summary Modes 🔄

The app offers the user two distinct summarization modes catering to different use cases and document types:

- **General Mode**: Desgined for creating general, concise summaries for a wide range of documents.

- **Research Mode**: Designed for detailed research summaries of academic articles. This mode generates research summaries answering the five C's (`Category`, `Context`, `Correctness`, `Contributions`, `Clarity`) as described [in this article](http://ccr.sigcomm.org/online/files/p83-keshavA.pdf).

### How to Select a Mode 🛠️

In the Streamlit app sidebar named `Settings`, you'll find the option to select between `General` (default) and `Research` modes. Choose the one that best fits your document type and summarization needs. Upon selection, the app will automatically adjust the summarization prompts to align with the chosen mode.

## Getting Started 🚀

### Prerequisites 📋

- An OpenAI API key 🔑, set as an environment variable named `OPENAI_API_KEY`.

### Setup Instructions 🛠️

1. **Clone the Repository**:

```
git clone https://github.com/FraMaBu/pdf-summarizer
cd pdf-summarizer
```

2. **Create and Activate a Virtual Environment**:

```
python -m venv .venv
.venv\Scripts\activate
```

3. **Install Dependencies**:

Install all the necessary dependencies listed in `requirements.txt`:

```
pip install -r requirements.txt
```

4. **Run the App**:

```
streamlit run app.py
```

Then, simply open your browser and head to the local URL provided by Streamlit. Enjoy summarizing effortlessly! 🎉

---

For any questions or feedback, feel free to get in touch! I'd love to hear from you! 🤝✉️
