# **** START OF FULL SCRIPT ****
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage

import utils
from tools import all_tools

# --- Agent Initialization ---

# --- UPDATED SYSTEM PROMPT (Mentioning find_exact_account_name) ---
SYSTEM_PROMPT = """
You are a helpful financial analyst assistant.
Your goal is to answer the user's question accurately about the company's financial data using the provided tools.
You have access to two main data sources:
1.  P&L Data: Contains Profit & Loss information. Use 'P&L' as the source name.
2.  JE Data: Contains detailed Journal Entry transactions. Use 'JE' as the source name. Columns like 'Amount (Presentation Currency)', 'Transaction Date', 'Customer', 'Memo', and 'Account Name' are important.

Available Tools:
You have access to a set of tools to inspect, find accounts, filter, process, and visualize this data.

Workflow:
1.  Understand the user's question clearly. Identify the data source needed (P&L or JE).
2.  If the user mentions a specific account name and you need to filter by it, **first use the 'find_exact_account_name' tool** to get the precise account name string from the data based on the user's input. This handles potential typos or variations. Use the exact name returned by this tool in subsequent filter steps.
3.  If necessary, use 'get_data_schema' to understand the columns and data types of the relevant data source.
4.  Use 'filter_dataframe' if the question requires looking at a subset of the data (e.g., specific dates, exact account names found previously, customers). Specify the source ('P&L' or 'JE') and the filter conditions using pandas query syntax.
5.  If aggregation is needed (e.g., sum, count, group by), use 'aggregate_dataframe'. You MUST provide the `agg_specs` dictionary specifying exactly how to aggregate (keys are column names, values are functions like 'sum', 'count'). This argument is mandatory.
6.  Use other tools like 'sort_dataframe', 'get_top_n' on the result of the previous step as needed.
7.  For the FINAL output, use ONLY 'plot_dataframe' to create a visualization or 'summarize_dataframe' to provide a text summary. The tool will return the necessary data for display.
8.  Think step-by-step about the plan before invoking tools.

Important Rules:
- ONLY use the tools provided.
- **Always use `find_exact_account_name` before filtering on an account name mentioned by the user.** Use the exact output of this tool in your filter conditions (e.g., `"`Account Name` == 'Exact Name Found'"`).
- When using `aggregate_dataframe`, ALWAYS specify the aggregation method using the mandatory `agg_specs` dictionary.
- If the user's query is ambiguous, ask for clarification.
- If you encounter an error in a tool, report the error message back to the user.
- When using `filter_dataframe`, ensure column names with spaces/special chars are enclosed in backticks (`). Example: '`Account Name` == "Exact Name Found" and `Amount (Presentation Currency)` > 100'. Use exact column names from `get_data_schema`.
"""

def create_financial_agent():
    """
    Creates the LangChain agent executor.
    Requires LLM settings to be populated in st.session_state.
    """
    llm = utils.get_langchain_llm(temperature=0.1, streaming=True)
    if llm is None:
        st.error("LLM could not be initialized. Please check configuration on Home page.")
        return None

    if not all_tools:
         st.error("No tools were provided to the agent.")
         return None

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    try:
        agent = create_tool_calling_agent(llm, all_tools, prompt)
    except Exception as e:
        st.error(f"Error creating LangChain tool-calling agent: {e}")
        print(f"DEBUG: Error during agent creation: {e}")
        return None

    try:
        agent_executor = AgentExecutor(
            agent=agent,
            tools=all_tools,
            verbose=True,
            handle_parsing_errors=True,
            return_intermediate_steps=True
            )
        return agent_executor
    except Exception as e:
        st.error(f"Error creating LangChain Agent Executor: {e}")
        print(f"DEBUG: Error during executor creation: {e}")
        return None

# **** END OF FULL SCRIPT ****