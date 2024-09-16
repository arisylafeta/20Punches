from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from state import State
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY").strip("__")



embeddings = OpenAIEmbeddings()

vector_store = FAISS.load_local(
    "knowledge_base", embeddings, allow_dangerous_deserialization=True
)
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})

#------------------------------
# Prompt for Generating Answer from HyDE. Keep {context} and {question} to not break it.
template = """Answer the following question based on this context:

{context}

Question: {question}
"""
#------------------------------

universal_prompt = ChatPromptTemplate.from_template(template)

# HyDE

#------------------------------
# Prompt for generating HyDE paragraph
template = """You are Warren Buffet. Answer this question with a passage using your principles: {question}
Passage:"""
#------------------------------


prompt_hyde = ChatPromptTemplate.from_template(template)

# Generate hypothetical document
generate_hyde_passage = (
    prompt_hyde | ChatOpenAI(temperature=0) | StrOutputParser()
)

# Retrieve documents based on the generated passage
hyde_retriever = generate_hyde_passage | retriever

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Complete HyDE RAG chain
hyde_rag_chain = (
    {"question": RunnablePassthrough()}
    | {"hyde_passage": generate_hyde_passage, "question": RunnablePassthrough()}
    | {"context": hyde_retriever | format_docs, "question": lambda x: x["question"]}
    | universal_prompt
    | ChatOpenAI(temperature=0)
)

def retrieve_docs(state: State) -> State:
    last_message = state["messages"][-1].content if state["messages"] else ""
    retrieved_docs = hyde_rag_chain.invoke({"question": last_message})
    
    # Create a ChatPromptTemplate for summarization
    summarization_template = """Given the following information and the user's question, provide a 400 word summary that is relevant to answering the question, refer to the context:

    User's question: {question}

    Information:
    {docs}

    Relevant summary:"""
    summarization_prompt = ChatPromptTemplate.from_template(summarization_template)
    
    # Create the summarization chain
    summarization_chain = (
        summarization_prompt 
        | ChatOpenAI(streaming=True) 
        | StrOutputParser()
    )
    
    # Invoke the summarization chain
    summary = summarization_chain.invoke({"docs": retrieved_docs, "question": last_message})
    
    state['summarized_docs'] = summary
    return state