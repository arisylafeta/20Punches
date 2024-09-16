from state import State
from router import router
from docs_summarizer import retrieve_docs
from ticker_extractor import extract_ticker
from financial_analyst import execute_financial_analysis
from buffet_bot import buffet_agent
from langchain_core.runnables import Runnable, RunnableConfig
import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import asyncio
from langchain_core.messages import AIMessage, HumanMessage
import time
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import base64
import random
import colorsys

# Graph
from langgraph.graph import StateGraph, END, START

st.set_page_config(page_title="20 Punches Investment", layout="wide")

# Brand colors
BRAND_COLORS = ['#6290C3', '#C2E7DA', '#F1FFE7', '#1A1B41', '#BAFF29']

graph = StateGraph(State)

graph.add_node("Router", router)
graph.add_node("DocRetriever", retrieve_docs)
graph.add_node("TickerExtractor", extract_ticker)
graph.add_node("FinancialAnalyst", execute_financial_analysis)
graph.add_node("BuffetAgent", buffet_agent)

graph.add_edge(START, "Router")
graph.add_conditional_edges(
    "Router",
    lambda x: x['routing_decision'],
    {
        "general_knowledge": "DocRetriever",
        "company_specific": "TickerExtractor"
    }
)

graph.add_edge("DocRetriever", "BuffetAgent")
graph.add_edge("TickerExtractor", "FinancialAnalyst")
graph.add_edge("FinancialAnalyst", "DocRetriever")
graph.add_edge("BuffetAgent", END)

multi_agent_graph = graph.compile()

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
    st.session_state.state = State(
        messages=[],
        tickers=None,
        financial_summary=None,
        summarized_docs=None,
        routing_decision=None
    )
if 'investments' not in st.session_state:
    st.session_state.investments = []

@st.cache_data
def get_company_suggestions(search_term):
    try:
        companies = pd.read_csv('https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv')
        name_col = 'Name' if 'Name' in companies.columns else 'Security'
        symbol_col = 'Symbol' if 'Symbol' in companies.columns else 'Symbol'
        
        return companies[
            companies[name_col].str.contains(search_term, case=False) | 
            companies[symbol_col].str.contains(search_term, case=False) |
            companies[name_col].str.lower().str.contains(search_term.lower())
        ].head(10)  # Increased to 10 results
    except Exception as e:
        st.error(f"Error fetching company data: {str(e)}")
        return pd.DataFrame()

def add_investment_section(key_suffix=""):
    st.header("Add Investment")
    search_term = st.text_input("Search for company or enter stock ticker:", key=f"search_input_{key_suffix}")
    
    if search_term:
        suggestions = get_company_suggestions(search_term)
        if not suggestions.empty:
            name_col = 'Name' if 'Name' in suggestions.columns else 'Security'
            symbol_col = 'Symbol' if 'Symbol' in suggestions.columns else 'Symbol'
            selected_company = st.selectbox(
                "Select a company:", 
                suggestions[name_col] + " (" + suggestions[symbol_col] + ")",
                key=f"company_select_{key_suffix}"
            )
            if selected_company:
                ticker = selected_company.split("(")[-1].split(")")[0].strip()
                if st.button(f"Add {ticker} to investments", key=f"add_button_{key_suffix}"):
                    add_investment(ticker)
                    # Clear the search input and reset the selectbox
                    st.session_state[f"search_input_{key_suffix}"] = ""
                    st.session_state[f"company_select_{key_suffix}"] = None
        else:
            st.warning("No matching companies found. Try a different search term.")

def add_investment(ticker):
    investments = st.session_state.get('investments', [])
    if len(investments) < 20:
        if any(inv['ticker'] == ticker.upper() for inv in investments):
            st.error(f"{ticker.upper()} is already in your investments.")
            return
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            investment = {
                'ticker': ticker.upper(),
                'name': info.get('longName', 'N/A'),
                'price': info.get('currentPrice', 'N/A'),
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'dividend_yield': info.get('dividendYield', 'N/A')
            }
            investments.append(investment)
            st.session_state.investments = investments
            st.success(f"Added {ticker.upper()} to your investments.")
            st.rerun()  # Use st.rerun() instead of st.experimental_rerun()
        except Exception as e:
            st.error(f"Couldn't fetch data for {ticker}. Error: {str(e)}")
    else:
        st.error("You've reached the maximum of 20 investments.")

def set_custom_style():
    with open("styles.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def display_punchcard():
    st.markdown("<h2 class='punchcard-title'>Your Investment Punchcard</h2>", unsafe_allow_html=True)
    st.markdown("<p class='punchcard-description'>Fill all 20 slots with your best investment ideas. Each punch represents a carefully chosen stock in your portfolio.</p>", unsafe_allow_html=True)
    
    investments = st.session_state.get('investments', [])
    
    punchcard_html = '<div class="punchcard">'
    for i in range(20):
        if i < len(investments):
            color = BRAND_COLORS[i % len(BRAND_COLORS)]
            punchcard_html += f'<div class="punchcard-button filled" style="background-color: {color};">{investments[i]["ticker"]}</div>'
        else:
            punchcard_html += f'<div class="punchcard-button empty"><span class="add-icon">+</span></div>'
    punchcard_html += '</div>'
    
    progress = len(investments)
    st.markdown(f'<div class="progress-indicator">Progress: {progress}/20 slots filled</div>', unsafe_allow_html=True)
    
    st.markdown(punchcard_html, unsafe_allow_html=True)

def display_portfolio_metrics():
    investments = st.session_state.get('investments', [])
    if investments:
        # Calculate portfolio-wide metrics
        total_value = sum(inv.get('price', 0) for inv in investments if isinstance(inv.get('price'), (int, float)))
        total_market_cap = sum(inv.get('market_cap', 0) for inv in investments if isinstance(inv.get('market_cap'), (int, float)))
        
        # Calculate weighted average P/E ratio
        total_weight = sum(inv.get('price', 0) for inv in investments if isinstance(inv.get('price'), (int, float)) and isinstance(inv.get('pe_ratio'), (int, float)))
        weighted_pe = sum(inv.get('price', 0) * inv.get('pe_ratio', 0) for inv in investments if isinstance(inv.get('price'), (int, float)) and isinstance(inv.get('pe_ratio'), (int, float))) / total_weight if total_weight else 0
        
        # Calculate average dividend yield
        div_yields = [inv.get('dividend_yield', 0) for inv in investments if isinstance(inv.get('dividend_yield'), (int, float))]
        avg_dividend_yield = sum(div_yields) / len(div_yields) if div_yields else 0
        
        # Count number of investments
        num_investments = len(investments)
        
        metrics_html = f'''
        <div class="metrics-container">
            <div class="metric-card">
                <h4>Total Portfolio Value</h4>
                <p>${total_value:,.2f}</p>
            </div>
            <div class="metric-card">
                <h4>Total Market Cap</h4>
                <p>${total_market_cap:,.0f}</p>
            </div>
            <div class="metric-card">
                <h4>Weighted Avg P/E Ratio</h4>
                <p>{weighted_pe:.2f}</p>
            </div>
            <div class="metric-card">
                <h4>Avg Dividend Yield</h4>
                <p>{avg_dividend_yield:.2%}</p>
            </div>
            <div class="metric-card">
                <h4>Number of Investments</h4>
                <p>{num_investments}</p>
            </div>
        </div>
        '''
        st.markdown(metrics_html, unsafe_allow_html=True)
    else:
        st.info("Add investments to see portfolio metrics.")

def display_investments():
    st.subheader("Your Investments")
    investments = st.session_state.get('investments', [])
    if investments:
        df = pd.DataFrame(investments)
        
        # Add Buffett-style metrics
        for i, investment in enumerate(investments):
            ticker = investment['ticker']
            stock = yf.Ticker(ticker)
            info = stock.info
            
            df.at[i, 'roe'] = info.get('returnOnEquity', 'N/A')
            df.at[i, 'debt_to_equity'] = info.get('debtToEquity', 'N/A')
            df.at[i, 'free_cash_flow'] = info.get('freeCashflow', 'N/A')
            df.at[i, 'profit_margins'] = info.get('profitMargins', 'N/A')
            df.at[i, 'earnings_growth'] = info.get('earningsQuarterlyGrowth', 'N/A')
        
        # Reorder and rename columns
        df = df[['ticker', 'name', 'price', 'pe_ratio', 'market_cap', 'dividend_yield', 'roe', 'debt_to_equity', 'free_cash_flow', 'profit_margins', 'earnings_growth']]
        df.columns = ['Ticker', 'Company', 'Price', 'P/E Ratio', 'Market Cap', 'Dividend Yield', 'ROE', 'Debt/Equity', 'Free Cash Flow', 'Profit Margin', 'Earnings Growth']
        
        # Format the dataframe
        df['Price'] = df['Price'].apply(lambda x: f"${x:.2f}" if x != 'N/A' else x)
        df['P/E Ratio'] = df['P/E Ratio'].apply(lambda x: f"{x:.2f}" if x != 'N/A' else x)
        df['Market Cap'] = df['Market Cap'].apply(lambda x: f"${x:,.0f}" if x != 'N/A' else x)
        df['Dividend Yield'] = df['Dividend Yield'].apply(lambda x: f"{x:.2%}" if x != 'N/A' else x)
        df['ROE'] = df['ROE'].apply(lambda x: f"{x:.2%}" if x != 'N/A' else x)
        df['Debt/Equity'] = df['Debt/Equity'].apply(lambda x: f"{x:.2f}" if x != 'N/A' else x)
        df['Free Cash Flow'] = df['Free Cash Flow'].apply(lambda x: f"${x:,.0f}" if x != 'N/A' else x)
        df['Profit Margin'] = df['Profit Margin'].apply(lambda x: f"{x:.2%}" if x != 'N/A' else x)
        df['Earnings Growth'] = df['Earnings Growth'].apply(lambda x: f"{x:.2%}" if x != 'N/A' else x)
        
        # Display the styled dataframe
        st.dataframe(df.style.set_properties(**{
            'background-color': '#1A1B41',
            'color': '#F1FFE7',
            'border-color': '#4a4a4a'
        }).set_table_styles([{
            'selector': 'th',
            'props': [('background-color', '#2c3e50'), ('color', '#BAFF29')]
        }]))
    else:
        st.info("You haven't added any investments yet.")

def punchcard_view():
    display_portfolio_metrics()
    display_punchcard()
    display_investments()

def chat_view():
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("I'm Warren Buffet. Ask me anything."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add user message to state
        st.session_state.state["messages"].append(HumanMessage(content=prompt))
        state = st.session_state.state

        # Generate the assistant's response
        with st.chat_message("assistant"):
            # Set up a placeholder for the streaming response
            message_placeholder = st.empty()
            full_response = ""

            # Use the graph to process the state
            with st.spinner("Time is a friend of a wonderful company, the enemy of mediocre..."):
                response = multi_agent_graph.invoke(state)

                ai_response = response["messages"][-1]["content"]
                # Use the graph to process the state
                for line in ai_response.split('\n'):
                    full_response += line + "\n"
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")

            # Display the full response
            message_placeholder.markdown(full_response)

        # Add the assistant's response to the chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Update the session state
        st.session_state.state = state

def sidebar_content():
    # Add logo image
    st.image("logo.jpeg")
    
    # Add title and subtitle
    st.title("20Punches")
    st.markdown("*Stand on the shoulder of investment giants*")
    
    st.markdown("<hr>", unsafe_allow_html=True)  # Add a separator
    
    # Move the view selection below the title and subtitle
    view = st.radio("Select View", ["Punchcard", "Chat"], key="view_toggle", horizontal=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)  # Add another separator
    
    if view == "Punchcard":
        add_investment_section(key_suffix="sidebar")
    else:
        st.write("Chat view selected. Add chat-specific sidebar content here if needed.")
    
    return view

def get_text_color(bg_color):
    # Convert hex to RGB
    bg_color = bg_color.lstrip('#')
    rgb = tuple(int(bg_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Calculate luminance
    luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
    
    # Return white for dark backgrounds, black for light backgrounds
    return '#ffffff' if luminance < 0.5 else '#000000'

# Add the main execution logic
def main():
    set_custom_style()
    
    # Create a sidebar
    with st.sidebar:
        view = sidebar_content()
    
    # Main content area
    if view == "Punchcard":
        punchcard_view()
    else:
        chat_view()

if __name__ == "__main__":
    main()
