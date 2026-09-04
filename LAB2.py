import streamlit as st
from openai import OpenAI
from pypdf import PdfReader


# reading a users PDF file into plain text
def read_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:  # we ignore all the images in the pages and focus on the text only
            text += page_text + "\n"
    return text


st.title("📄 Lab 2 - Document Summarizer")
st.write(
    "Upload a document and choose how you'd like it summarized. "
    "Use the options in the sidebar to control the summary type, language, and model."
)

# Part B: getting the API key from Streamlit secrets, NO  text box 
# The key lives in .streamlit/secrets.toml locally, and in the app's
openai_api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)

#  Part C: sidebar controls 

# 1) Summary type: three options.
summary_type = st.sidebar.selectbox(
    "Type of summary:",
    (
        "Summarize in 100 words",
        "Summarize in 2 connecting paragraphs",
        "Summarize in 5 bullet points",
    ),
)

# 2) Output language.
language = st.sidebar.selectbox(
    "Summary language:",
    ("English", "French", "Spanish"),
)

# 3) Model choice: checkbox picks the advanced model.
use_advanced = st.sidebar.checkbox("Use advanced model")
# nano is the cheaper/faster default; mini is the advanced option.
model = "gpt-5-mini" if use_advanced else "gpt-5-nano"
st.sidebar.caption(f"Using model: {model}")

# --- File upload (no typed question anymore — summary only) ---
# now accepting .pdf and .txt files
uploaded_file = st.file_uploader(
    "Upload a document (.pdf or .txt)", type=("pdf", "txt", )
)

if uploaded_file:
    # routing the file to the right reader based on its extension
    document = None
    file_extension = uploaded_file.name.split('.')[-1]
    if file_extension == 'txt':
        document = uploaded_file.read().decode()
    elif file_extension == 'pdf':
        document = read_pdf(uploaded_file)
    else:
        st.error("Unsupported file type.")

    # only summarize if we successfully read the document
    if document:
        # Turn the chosen summary type into a clear instruction for the model.
        if summary_type == "Summarize in 100 words":
            instruction = "Summarize the document in about 100 words."
        elif summary_type == "Summarize in 2 connecting paragraphs":
            instruction = "Summarize the document in two connecting paragraphs."
        else:
            instruction = "Summarize the document in exactly 5 bullet points."

        # Add the language requirement to the instruction.
        instruction += f" Write the summary in {language}."

        messages = [
            {
                "role": "user",
                "content": f"Here's a document:\n\n{document}\n\n---\n\n{instruction}",
            }
        ]

        # Generate and stream the summary.
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        st.write_stream(stream)