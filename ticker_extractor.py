from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from state import State
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY").strip("__")



# Data model for ticker extraction
class TickerExtraction(BaseModel):
    """Extract the relevant stock tickers from a user query."""

    tickers: List[str] = Field(
        default_factory=list,
        description="The stock tickers mentioned or implied in the query. Use an empty list if no specific companies are mentioned.",
    )

# LLM with function call
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
structured_llm_ticker_extractor = llm.with_structured_output(TickerExtraction)

# Prompt
system = """You are an expert at identifying company stock tickers in financial queries.
If specific companies are mentioned or strongly implied, provide their stock tickers as a list.
If no specific companies are mentioned or implied, return an empty list.
Always provide a brief explanation for your decision."""

ticker_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

ticker_extractor = ticker_prompt | structured_llm_ticker_extractor

def extract_ticker(state: State) -> State:
    last_message = state["messages"][-1].content if state["messages"] else ""
    result = ticker_extractor.invoke({"question": last_message})
    
    state['tickers'] = result.tickers
    
    return state