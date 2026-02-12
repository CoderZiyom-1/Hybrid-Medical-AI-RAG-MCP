import os
import asyncio
import streamlit as st

from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# MCP session
from mcp.client import session


DB_FAISS_PATH = "vectorstore/db_faiss"


# ========== RAG PART ==========

@st.cache_resource
def get_vectorstore():
    embedding_model = OllamaEmbeddings(model="all-minilm")  # local embeddings
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return db


def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )


def load_llm(model_name="mistral"):
    return OllamaLLM(model=model_name)


# ========== MCP PART ==========

async def query_mcp(query: str):
    """
    Calls MCP tool (example: 'symptom-checker') with the user query.
    """
    try:
        async with session("symptom-checker") as s:   # <-- replace with your real MCP tool name
            result = await s.call_tool("search_symptoms", {"query": query})
            return result.get("content", "No answer from MCP")
    except Exception as e:
        return f"(MCP error: {str(e)})"


def query_mcp_sync(query: str):
    """
    Makes MCP call usable inside normal (non-async) code like Streamlit.
    """
    return asyncio.run(query_mcp(query))


# ========== STREAMLIT APP ==========

def main():
    st.title("🩺 DOC SAABH – Hybrid RAG + MCP")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Ask DOC SAABH")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        You are a strict medical assistant. 
        Only use the provided context to answer the user's question. 

        Rules:
        - If the context does not contain an answer, reply: "I don’t know how to help with that."
        - Do not guess, do not add outside knowledge, do not hallucinate.
        - Stick to the medical PDFs only. 

        Context:
        {context}

        Question:
        {question}

        Answer:
        """

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("❌ Failed to load the vector store")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm("mistral"),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            # 🔹 RAG response
            rag_response = qa_chain.invoke({'query': prompt})["result"]

            # 🔹 MCP response
            mcp_response = query_mcp_sync(prompt)

            # 🔹 Final merged answer
            final_answer = f"""
**📘 RAG (from PDFs):**
{rag_response}

---

**⚡ MCP (external tool):**
{mcp_response}
"""

            st.chat_message('assistant').markdown(final_answer)
            st.session_state.messages.append({'role': 'assistant', 'content': final_answer})

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
