from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from state import State
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY").strip("__")


# Data model
class RouteQuery(BaseModel):
    """Route a user query to the appropriate action."""

    action: Literal["company_specific", "general_knowledge"] = Field(
        ...,
        description="Given a user question, choose whether it's company-specific or general knowledge.",
    )

# LLM with function call
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

# Prompt
system = """You are an expert at determining whether a query is about a specific company or financial data, or if it's a general knowledge question.
For queries about specific companies, financial metrics, stock performance, or company-specific data, use 'company_specific'.
For general knowledge questions, including explanations of financial concepts not tied to a specific company, use 'general_knowledge'."""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router

def router(state: State) -> State:
    last_message = state["messages"][-1].content if state["messages"] else ""
    result = question_router.invoke({"question": last_message})
    
    # Update the state with the routing decision
    state['routing_decision'] = result.action
    
    return state