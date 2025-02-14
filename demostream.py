import streamlit as st
import os
from nova import SpaceAI  # Import your class
from tempfile import NamedTemporaryFile
import asyncio 

# Streamlit UI setup
st.set_page_config(page_title="Demo AI", page_icon="🤖", layout="wide")

st.title("🤖 Demo AI - Your Assistant")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    data_directory = "./data"
    os.makedirs(data_directory, exist_ok=True)  # Ensure the directory exists

    # Model selection (optional)
    model_choice = st.selectbox("Select LLM Model", ["mistralai/Mistral-7B-Instruct-v0.3"])
    include_web = st.checkbox("Include Web Search", False)

    st.markdown("---")
    uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])

    if uploaded_file:
        with NamedTemporaryFile(delete=False, dir=data_directory) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name
            st.session_state.uploaded_file_path = temp_path  # Store the path in session state

st.markdown("---")

# Initialize session state for conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Type your message here...")
# User input
if query:
    user_id = "user_123"  # Replace with an actual user ID system
    ai = SpaceAI(data_directory, query, user_id, include_web, model_choice)

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            response, urls = asyncio.run(ai.handle_user_message())

            # Handle different response types
            if isinstance(response, str):
                st.markdown(response)
            elif isinstance(response, dict):
                st.markdown("### Response:")
                st.json(response)  # Display structured responses as JSON
            elif isinstance(response, list):
                st.markdown("### Response List:")
                for item in response:
                    st.markdown(f"- {item}")
            else:
                st.error("Unexpected response format.")

            # Display sources if available
            if urls and isinstance(urls, list):
                st.markdown("### Sources:")
                for url in urls:
                    if isinstance(url, str):
                        st.markdown(f"- [{url}]({url})")
                    else:
                        st.markdown(f"- {url}")  # Handle non-string URLs safely

        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            st.error(error_message)
            response = error_message  # Store the error message in session history

    # Store conversation history
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "assistant", "content": response})


