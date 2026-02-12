import os
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from mcp.client import MCPClient  

# ---------------------------
# Step 1: Setup local LLM
# ---------------------------
def load_llm(model_name="mistral"):
    return Ollama(model=model_name)


# ---------------------------
# Step 2: Custom Prompt Template
# ---------------------------
CUSTOM_PROMPT_TEMPLATE = """
You are a medical assistant. 
Use ONLY the provided context and approved MCP tools to answer.
Do not hallucinate or guess.
If unclear, say "I don't know."

Context: {context}
Question: {question}

Start answer directly, no small talk.
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(
        template=custom_prompt_template, 
        input_variables=["context", "question"]
    )


# ---------------------------
# Step 3: Load FAISS Vectorstore
# ---------------------------
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = OllamaEmbeddings(model="all-minilm")

db = FAISS.load_local(
    DB_FAISS_PATH, 
    embedding_model, 
    allow_dangerous_deserialization=True
)


# ---------------------------
# Step 4: MCP Setup
# ---------------------------
def get_mcp_client():
    # Example: local MCP server (adjust as needed)
    return MCPClient("http://localhost:8000")

def query_mcp(client, tool_name, query):
    """Call MCP tool for extra context."""
    try:
        response = client.call_tool(tool_name, {"query": query})
        return response.get("result", "No result from MCP")
    except Exception as e:
        return f" MCP Error: {e}"


# ---------------------------
# Step 5: Hybrid QA
# ---------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm("mistral"),  
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)


def hybrid_query(user_query):
    # 1. RAG result
    rag_response = qa_chain.invoke({'query': user_query})
    rag_result = rag_response["result"]
    sources = rag_response["source_documents"]

    # 2. MCP result
    client = get_mcp_client()
    if "drug" in user_query.lower():
        mcp_result = query_mcp(client, "drug_info", user_query)
    elif "lab" in user_query.lower():
        mcp_result = query_mcp(client, "lab_values", user_query)
    elif "guideline" in user_query.lower():
        mcp_result = query_mcp(client, "guidelines", user_query)
    else:
        mcp_result = "ℹ No MCP tool used."

    # 3. Combine
    result_to_show = f"🧾 **RAG Answer:**\n{rag_result}\n\n📚 **Sources:**\n"
    for i, doc in enumerate(sources, 1):
        result_to_show += f"\n[{i}] {doc.page_content[:200]}..."

    result_to_show += f"\n\n🔗 **MCP Lookup:**\n{mcp_result}"
    return result_to_show


# ---------------------------
# Step 6: Run
# ---------------------------
if __name__ == "__main__":
    user_query = input("Write Query Here: ")
    final_answer = hybrid_query(user_query)
    print("\nFINAL ANSWER:\n", final_answer)
