# PDF Summary Generator - Streamlit App 📚🚀

## Overview 🌟

Welcome to the Generative AI powered **PDF Summary Generator**! This `Streamlit` application is your go-to tool for generating quick and concise summaries from PDF documents. Whether you're a student 🎓, researcher 🔬, or a busy professional 🏢, the app makes it a breeze to extract the essence of lengthy documents. Powered by the custom `genaipy` [Python library](https://github.com/FraMaBu/genaipy-lib), get ready to transform how you process information!

## How It Works 🔍

The app leverages a [Map-Reduce](https://js.langchain.com/docs/modules/chains/document/map_reduce). approach to distill text from PDFs, comprising:

1. 🗺️ **Map Step**: Extracts text from each PDF page, creating individual intermediate summaries.
2. 📉 **Reduce Step**: Merges these summaries into a final, comprehensive document summary.

## Features 🌈

- 📤 **PDF Upload**: Upload any PDF file effortlessly.
- 📄 **Page Selection**: Select specific page ranges for targeted summarization.
- ⚙️ **Customizable Summarization**: Tweak settings like summary word count and system messages.
- 💾 **Downloadable Summaries**: Grab your summary in `.txt` or `.md` formats.

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
