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
from genaipy.utilities import validate_api_key
from prompts import SYS_MESSAGE_RESEARCH, MAP_PROMPT_RESEARCH, REDUCE_PROMPT_RESEARCH

# Logger configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Authentication (use your method to validate or input API key)
OPENAI_API_KEY = validate_api_key(env_var_name="OPENAI_API_KEY")

# Define mode-specific prompts
mode_settings = {
    "General": {
        "sys_message": DEFAULT_SYS_MESSAGE,
        "map_prompt_tpl": SUMMARY_PROMPT_TPL,
        "reduce_prompt_tpl": REDUCE_SUMMARY_PROMPT_TPL,
    },
    "Research": {
        "sys_message": SYS_MESSAGE_RESEARCH,
        "map_prompt_tpl": MAP_PROMPT_RESEARCH,
        "reduce_prompt_tpl": REDUCE_PROMPT_RESEARCH,
    },
}

# Streamlit UI components
st.title("PDF Summary Generator")

# Initialize session state for summary
if "final_summary" not in st.session_state:
    st.session_state["final_summary"] = None

# Sidebar settings
st.sidebar.title("Settings")

# Mode selection
st.sidebar.header("Summary Mode")
selected_mode = st.sidebar.radio(
    "Select a mode for your summary:",
    ["General", "Research"],
    index=0,
    help="`General` is suited for a wide range of documents. `Research` provides detailed summaries of academic articles.",
)

user_system_message = mode_settings[selected_mode]["sys_message"]
map_prompt_template = mode_settings[selected_mode]["map_prompt_tpl"]
reduce_prompt_template = mode_settings[selected_mode]["reduce_prompt_tpl"]

st.sidebar.info(f"Mode selected: **{selected_mode}**")
logging.info("%s mode-specific prompts have been initialized.", selected_mode)

# Max words selection
st.sidebar.header("Summary Configuration")
user_map_words = st.sidebar.number_input(
    "Maximum words for each map summary:",
    min_value=50,
    max_value=300,
    value=150,
    help="Adjust the maximum word count for individual page summaries.",
)
user_reduce_words = st.sidebar.number_input(
    "Maximum words for final reduce summary:",
    min_value=100,
    max_value=500,
    value=300,
    help="Set the maximum word count for the final summary.",
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
                template=MAP_PROMPT_RESEARCH,
                text=content,
                max_words=user_map_words,
            )
            summary = get_chat_response(
                map_prompt, sys_message=user_system_message, model="gpt-3.5-turbo"
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
            template=REDUCE_PROMPT_RESEARCH, text=text, max_words=user_reduce_words
        )
        final_summary = get_chat_response(
            prompt=reduce_prompt,
            sys_message=user_system_message,
            model="gpt-4-1106-preview",
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
