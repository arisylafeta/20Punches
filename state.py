from langgraph.graph.message import AnyMessage, add_messages
from typing import List, Optional, TypedDict, Annotated, Sequence
import operator

class State(TypedDict):
    messages: Annotated[Sequence[AnyMessage], operator.add]
    tickers: Optional[List[str]]
    financial_summary: Optional[str]
    summarized_docs: Optional[str]
    routing_decision: Optional[str]  