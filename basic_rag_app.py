import os
import streamlit as st

from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS


DB_FAISS_PATH = "vectorstore/db_faiss"

# Step 1: Load FAISS vectorstore
@st.cache_resource
def get_vectorstore():
    embedding_model = OllamaEmbeddings(model="all-minilm")  # local embeddings
    db = FAISS.load_local(
        DB_FAISS_PATH, 
        embedding_model, 
        allow_dangerous_deserialization=True
    )
    return db


# Step 2: Custom prompt
def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt


# Step 3: Load local Ollama LLM
def load_llm(model_name="mistral"):
    llm = OllamaLLM(model=model_name)  # local mistral via Ollama
    return llm


def main():
    st.title("Ask Chatbot (Local Ollama + FAISS)")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Ask a question about your PDFs...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Don't provide anything outside of the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("❌ Failed to load the vector store")

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm("mistral"),  # or "llama2", depending on which you pulled
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})

            result = response["result"]
            source_documents = response["source_documents"]

            result_to_show = result + "\n\n📚 Source Docs:\n"
            for i, doc in enumerate(source_documents, 1):
                result_to_show += f"\n[{i}] {doc.page_content[:300]}..."

            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()

