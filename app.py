import streamlit as st
from rag_app import rag_chain


st.set_page_config(
    page_title="My RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("AI TechCorp Employee Assistant")
st.write("Ask questions or doubts related to our company related things.")

question = st.chat_input("Ask your questions and doubts....")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching...."):
            try:
                response = rag_chain.invoke(question)
            except Exception:
                st.error(
                    "The AI service couldn't answer now."
                    "Please try again later"
                )

            if isinstance(response.content, list):
                answer = response.content[0]['text']
            else:
                answer = response.content

            st.write(answer)