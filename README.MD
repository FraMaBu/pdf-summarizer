# PDF Summarizer

A `Streamlit` app for generating AI summaries with the [map-reduce technique](https://python.langchain.com/docs/modules/chains/document/map_reduce). The app is powered by the `genaipy` Python library.

# PDF Summary Generator - Streamlit App

## Overview

Welcome to the PDF Summary Generator! This `Streamlit` application simplifies generating concise summaries from PDF documents, ideal for students, researchers, and professionals looking to quickly distill knowledge embedded in lengthy documents. The app is powered by the `genaipy` Python library.

## How It Works

The app employs a [Map-Reduce](<(https://python.langchain.com/docs/modules/chains/document/map_reduce)>) methodology to process and summarize text from your PDFs, involving:

1. **Map Step**: Extracting the text from each PDF page and generating individual intermediate summaries.
2. **Reduce Step**: Combining these summaries into a final, comprehensive document summary.

## Features

- **PDF Upload**: Upload any PDF file.
- **Page Selection**: Choose a specific page range for summarization.
- **Customizable Summarization**: Adjust settings like customizing the system message and maximum word count of summaries.
- **Downloadable Summaries**: Download the summary in `.txt` or `.md` formats.

## Getting Started

### Prerequisites

- Access to an OpenAI API key, which is set as an environemnt variable `OPENAI_API_KEY`.

### Setup Instructions

1. **Clone the Repository**:

```

git clone [Your Repository URL]
cd [Your Repository Name]

```

2. **Create a Conda Environment**:

```

conda create --name your_env_name python=3.11
conda activate your_env_name

```

3. **Install Dependencies**:

- Install Streamlit:
  ```
  pip install streamlit
  ```
- Install `genaipy` directly from GitHub:
  ```
  pip install git+[Your genaipy GitHub Repo URL]
  ```

4. **Run the App**:

```

streamlit run app.py

```

Open your browser and navigate to the local URL provided by Streamlit to start using the PDF Summary Generator.

## Acknowledgements

Powered by `genaipy`, a custom AI library for efficient data processing. Find `genaipy` in its dedicated repository under the same GitHub account.

---

Enjoy summarizing with ease!
