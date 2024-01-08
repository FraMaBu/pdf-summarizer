""" Streamlit App for generating Map-Reduce Summaries from PDFs."""

import logging
import streamlit as st
from genaipy.extractors.pdf import extract_pages_text
from genaipy.openai_apis.chat import get_chat_response
from genaipy.prompts.build_prompt import build_prompt
from genaipy.prompts.generate_summaries import (
    DEFAULT_SYS_MESSAGE,
    SUMMARY_PROMPT_TPL,
    REDUCE_SUMMARY_PROMPT_TPL,
)
from genaipy.utils import validate_api_key

# Logger configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Authentication (use your method to validate or input API key)
OPENAI_API_KEY = validate_api_key(env_var_name="OPENAI_API_KEY")


# Streamlit UI components
st.title("PDF Summary Generator")

# Initialize session state for summary
if "final_summary" not in st.session_state:
    st.session_state["final_summary"] = None

# Sidebar settings
st.sidebar.title("Settings")

st.sidebar.header("System Message")
system_message_choice = st.sidebar.selectbox(
    "Choose system message option:", ["Default", "Custom"], index=0
)
if system_message_choice == "Custom":
    user_system_message = st.sidebar.text_input("Enter your custom system message:")
else:
    user_system_message = DEFAULT_SYS_MESSAGE


st.sidebar.header("Map Step")
user_map_llm = st.sidebar.selectbox(
    "Map step model:", ["gpt-3.5-turbo", "gpt-4-1106-preview"], index=0
)
user_map_words = st.sidebar.number_input(
    "Maximum words for each map summary:", min_value=50, max_value=300, value=150
)

st.sidebar.header("Reduce Step")
user_reduce_llm = st.sidebar.selectbox(
    "Reduce step model:", ["gpt-3.5-turbo", "gpt-4-1106-preview"], index=1
)
user_reduce_words = st.sidebar.number_input(
    "Maximum words for final reduce summary:", min_value=100, max_value=500, value=300
)


# Functions
def process_pdf(full_path, start_page, end_page):
    """Extracts text from specified page range of a PDF file."""
    try:
        pages = extract_pages_text(
            pdf_path=full_path, start_page=start_page, end_page=end_page
        )
        logging.info("Successfully loaded text from %d PDF pages.", len(pages))
        return pages
    except Exception as e:
        logging.error("An error occurred while processing PDF: %s", e)
        raise


def generate_map_summaries(pages, progress_bar):
    """Generates summaries for each page."""
    map_summaries = []
    total_pages = len(pages)
    for i, (page, content) in enumerate(pages.items()):
        try:
            map_prompt = build_prompt(
                template=SUMMARY_PROMPT_TPL,
                text=content,
                max_words=user_map_words,
            )
            summary = get_chat_response(
                map_prompt, sys_message=user_system_message, model=user_map_llm
            )
            map_summaries.append(summary)
            logging.info("Map Summary #%d: %s", page, summary)

            # Update progress bar and status
            progress_bar.progress((i + 1) / total_pages)

        except Exception as e:
            logging.error("An error occurred while generating summary #%d: %s", page, e)
            raise
    return map_summaries


def generate_reduce_summary(map_summaries):
    """Generates a final summary from map summaries using reduce summarization."""
    text = "\n".join(map_summaries).replace("\n\n", "")
    try:
        reduce_prompt = build_prompt(
            template=REDUCE_SUMMARY_PROMPT_TPL, text=text, max_words=user_reduce_words
        )
        final_summary = get_chat_response(
            prompt=reduce_prompt,
            sys_message=user_system_message,
            model=user_reduce_llm,
            max_tokens=1024,
        )
        logging.info("Completed final reduce summary!")
        return final_summary
    except Exception as e:
        logging.error("An error occurred while generating final summary: %s", e)
        raise


# Main page layout
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
user_start_page = st.number_input("Start Page", min_value=1, value=1)
user_end_page = st.number_input("End Page", min_value=1, value=10)

# Button to generate summary
if st.button("Generate Summary") and uploaded_file is not None:
    TEMP_PDF_PATH = "temp.pdf"  # Saves uploaded file temporarily
    with open(TEMP_PDF_PATH, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing..."):
        temp_progress_bar = st.progress(0)
        status_text = st.empty()
        try:
            temp_pages = process_pdf(TEMP_PDF_PATH, user_start_page, user_end_page)
            status_text.info("Step 1: Generating map summaries...")
            temp_map_summaries = generate_map_summaries(temp_pages, temp_progress_bar)
            status_text.info("Step 2: Generating final summary...")
            st.session_state["final_summary"] = generate_reduce_summary(
                temp_map_summaries
            )
            status_text.success("Summary generation completed successfully!")
        except Exception as err:
            status_text.error(f"An error occurred: {err}")

# Displaying output and download options if summary has been generated
if st.session_state["final_summary"]:
    st.subheader("Final Summary")
    st.markdown(st.session_state["final_summary"])

    # Provide options for file formats when downloading
    file_format = st.selectbox(
        "Choose a file format for download:",
        ["Text File (.txt)", "Markdown File (.md)"],
    )
    if file_format == "Text File (.txt)":
        st.download_button(
            label="Download Summary",
            data=st.session_state["final_summary"],
            file_name="summary.txt",
            mime="text/plain",
        )
    elif file_format == "Markdown File (.md)":
        st.download_button(
            label="Download Summary",
            data=st.session_state["final_summary"],
            file_name="summary.md",
            mime="text/markdown",
        )
