import ssl
from urllib.request import urlopen
import certifi
import json
from langchain_core.tools import tool
import os

#--------------------------------------------------------------------------------------------------
# TOOLS
#--------------------------------------------------------------------------------------------------

# Function to clean the ticker input
def clean_ticker(ticker: str) -> str:
    return ticker.strip().upper()

# Function 1
@tool
def get_financial_key_metrics(ticker: str) -> str:
    """
    Retrieve all financial key metrics for a specified company.

    This function fetches financial data from the Financial Modeling Prep API
    and returns all the data as a formatted string.

    Parameters:
    ticker (str): The stock symbol of the company (e.g., "AAPL" for Apple Inc.).

    Returns:
    str: A formatted string containing all key metrics for the company.
    """
    ticker = clean_ticker(ticker)
    api_key = os.getenv('FINANCIAL_MODELING_PREP_API_KEY')
    url = f"https://financialmodelingprep.com/api/v3/key-metrics/{ticker}?apikey={api_key}"
    
    context = ssl.create_default_context(cafile=certifi.where())
    
    with urlopen(url, context=context) as response:
        data = response.read().decode("utf-8")
    
    json_data = json.loads(data)
    
    result = ""
    for item in json_data:
        result += "========================\n"
        for key, value in item.items():
            result += f"{key}: {value}\n"
    
    return result

# Function 2
@tool
def get_financial_ratios(ticker: str) -> str:
    """
    Retrieve all financial ratios for a specified company.

    This function fetches financial ratio data from the Financial Modeling Prep API
    and returns all the data as a formatted string.

    Parameters:
    ticker (str): The stock symbol of the company (e.g., "AAPL" for Apple Inc.).

    Returns:
    str: A formatted string containing all financial ratios for the company.
    """
    ticker = clean_ticker(ticker)
    api_key = os.getenv('FINANCIAL_MODELING_PREP_API_KEY')
    url = f"https://financialmodelingprep.com/api/v3/ratios/{ticker}?&apikey={api_key}"
    
    context = ssl.create_default_context(cafile=certifi.where())
    
    with urlopen(url, context=context) as response:
        data = response.read().decode("utf-8")
    
    json_data = json.loads(data)
    
    result = ""
    for item in json_data:
        result += "========================\n"
        for key, value in item.items():
            result += f"{key}: {value}\n"
    
    return result

# Function 3
# Will add more functions as needed

tools = [get_financial_key_metrics, get_financial_ratios]

#--------------------------------------------------------------------------------------------------
# Financial Analyst Agent
#--------------------------------------------------------------------------------------------------

from state import State
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY").strip("__")


model = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
prompt = ChatPromptTemplate.from_template("""
    You are a financial analysis assistant powered by Warren Buffett's investment philosophy. Your job is to gather data relevant to the question as best as you can and prepare a good summary.
    You have access to the following tools:

    {tools}

    Use the following format:

    Question: The question you must provide relevant information on.
    Ticker: The company you should research

    Thought: Reflect on what information you need to answer the question and which tool should provide you with the best information. 

    Action: Choose one of [{tool_names}]
    Action Input: The input to the tool call. This must be ONLY the ticker symbol, with no additional characters, spaces, or formatting. For example: AAPL

    Observation: The result of the action
    ... (this Thought/Action/Action Input/Observation can be repeated as necessary)

    Thought: I now have sufficient information to provide a comprehensive summary

    Final Answer: Provide a detailed summary of the relevant financial metrics/ratios and their significance in answering the question, incorporating Warren Buffett's investment principles where applicable.

    Begin your analysis:

    Question: {input}
    Ticker: {ticker}
    Thought:{agent_scratchpad}""")

agent = create_react_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

def execute_financial_analysis(state: State) -> State:
    last_message = state['messages'][-1].content
    tickers = state['tickers']
    
    result = agent_executor.invoke(
        {
            "input": last_message,
            "ticker": tickers
        }
    )
    
    state['financial_summary'] = result['output']
 
    return state