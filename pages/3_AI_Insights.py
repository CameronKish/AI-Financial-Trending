# pages/3_AI_Insights.py
# **** START OF FULL SCRIPT ****
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import time
from langchain_core.agents import AgentAction, AgentFinish # Import AgentAction

import utils # Includes ensure_settings_loaded
from agent import create_financial_agent # Import the function to create the agent executor
from tools import INTERMEDIATE_DF_KEY # Import key used by tools

# --- Page Setup ---
st.set_page_config(layout="wide", page_title="AI Insights")
st.markdown(f"<style> h1 {{ color: {utils.EY_DARK_BLUE_GREY}; }} </style>", unsafe_allow_html=True)
st.markdown(f"<h1>🤖 AI Insights (Chat with Data)</h1>", unsafe_allow_html=True)
st.caption("Ask questions about your P&L or Journal Entry data. Use the sidebar to add specific data context (optional).")

# --- Ensure Settings & Data are Loaded ---
utils.ensure_settings_loaded()

if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.error("Data not loaded. Please go back to the Home page and ensure data is loaded correctly.")
    st.stop()

if 'pl_flat_df' not in st.session_state or 'je_detail_df' not in st.session_state:
     st.error("Financial dataframes not found in session state. Please reload data.")
     st.stop()

# --- Initialize State Keys ---
if "ai_insights_messages" not in st.session_state:
    st.session_state.ai_insights_messages = []
if "sidebar_source" not in st.session_state:
    st.session_state.sidebar_source = "None"
if "sidebar_columns" not in st.session_state:
    st.session_state.sidebar_columns = []
if "sidebar_filters" not in st.session_state:
    st.session_state.sidebar_filters = {}

# --- Helper Callbacks for Sidebar ---
def clear_downstream_sidebar_filters():
    st.session_state.sidebar_columns = []
    st.session_state.sidebar_filters = {}
def clear_value_sidebar_filters():
    st.session_state.sidebar_filters = {}
def clear_all_sidebar_filters():
    st.session_state.sidebar_source = "None"
    st.session_state.sidebar_columns = []
    st.session_state.sidebar_filters = {}
    st.rerun()

# --- Sidebar Context Filters ---
st.sidebar.header("Data Context Filters (Optional)")
st.sidebar.caption("Define context here to help the AI focus its analysis.")
source_options = ["None", "P&L", "JE"]
selected_source = st.sidebar.selectbox(
    "1. Select Data Source:", options=source_options, key="sidebar_source",
    on_change=clear_downstream_sidebar_filters
)
source_df, source_columns = None, []
if selected_source == "P&L":
    source_df = st.session_state.pl_flat_df; source_columns = sorted(source_df.columns.tolist())
elif selected_source == "JE":
    source_df = st.session_state.je_detail_df; source_columns = sorted(source_df.columns.tolist())

selected_cols = []
if selected_source != "None" and source_columns:
    selected_cols = st.sidebar.multiselect(
        "2. Select Columns (Optional):", options=source_columns, key="sidebar_columns",
        on_change=clear_value_sidebar_filters
    )

current_value_filters = {}
if selected_cols and source_df is not None:
    st.sidebar.markdown("3. Filter by Values (Optional):")
    filters_in_state = st.session_state.sidebar_filters.copy()
    for col in selected_cols:
        if col in source_df.columns:
            try:
                unique_vals = sorted(source_df[col].dropna().unique().astype(str))
                if len(unique_vals) > 0:
                    if len(unique_vals) > 1000: st.sidebar.caption(f"Warning: Column '{col}' has >1000 unique values.")
                    default_selection = filters_in_state.get(col, [])
                    selected_values = st.sidebar.multiselect(
                        f"'{col}' values:", options=unique_vals, default=default_selection,
                        key=f"sidebar_filter_{col}"
                    )
                    if selected_values: current_value_filters[col] = selected_values
                else: st.sidebar.caption(f"Column '{col}' has no filterable unique values.")
            except Exception as e: st.sidebar.warning(f"Could not get values for '{col}': {e}")
        else: st.sidebar.warning(f"Column '{col}' not found in {selected_source} data.")
    if current_value_filters != st.session_state.sidebar_filters:
         st.session_state.sidebar_filters = current_value_filters

st.sidebar.button("Clear All Context Filters", on_click=clear_all_sidebar_filters, use_container_width=True)

# --- Display Chat History ---
# Use enumerate to get an index for unique keys
for idx, message in enumerate(st.session_state.ai_insights_messages):
    with st.chat_message(message["role"]):
        if "text" in message["content"] and message["content"]["text"]:
            st.markdown(message["content"]["text"])
        if "plot_json" in message["content"] and message["content"]["plot_json"]:
            try:
                fig = go.Figure(json.loads(message["content"]["plot_json"]))
                # --- ADDED KEY ---
                st.plotly_chart(fig, use_container_width=True, key=f"history_plot_{idx}")
            except Exception as e:
                st.error(f"Error displaying previous plot (idx: {idx}): {e}")
        if "audit_dataframe" in message["content"] and message["content"]["audit_dataframe"]:
            try:
                audit_df = pd.DataFrame(message["content"]["audit_dataframe"])
                expander_title = "Show Data Used for Response"
                if "plot_json" in message["content"] and message["content"]["plot_json"]: expander_title = "Show Data Used for Plot"
                elif "text" in message["content"] and isinstance(message["content"].get("text"), str) and message["content"]["text"].startswith("Summary of"): expander_title = "Show Data Used for Summary"
                with st.expander(expander_title):
                    # --- ADDED KEY ---
                    st.dataframe(audit_df, height=200, key=f"history_audit_{idx}")
            except Exception as e:
                st.error(f"Error displaying previous audit data (idx: {idx}): {e}")


# --- Handle Chat Input ---
if prompt := st.chat_input("Ask a question (or use sidebar context)"):
    st.session_state.ai_insights_messages.append({"role": "user", "content": {"text": prompt}})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate unique key prefix for elements related to this specific response
    current_response_key_prefix = f"response_{len(st.session_state.ai_insights_messages)}"

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response_content = {"text": None, "plot_json": None, "audit_dataframe": None}

        try:
            if INTERMEDIATE_DF_KEY in st.session_state: del st.session_state[INTERMEDIATE_DF_KEY]
            agent_executor = create_financial_agent()
            if agent_executor:
                with st.status("Thinking and processing...", expanded=False) as status:
                    st.write("Initializing agent...")
                    context_prefix = ""
                    sidebar_context_parts = []
                    if st.session_state.sidebar_source and st.session_state.sidebar_source != "None":
                        sidebar_context_parts.append(f"Use {st.session_state.sidebar_source} data.")
                    if st.session_state.sidebar_filters:
                        filter_parts = []
                        for col, values in st.session_state.sidebar_filters.items():
                            formatted_values = [f'"{v}"' if isinstance(v, str) else str(v) for v in values]
                            col_formatted = f"`{col}`" if (' ' in col or '(' in col or '/' in col) else col
                            filter_parts.append(f"{col_formatted} in [{', '.join(formatted_values)}]")
                        if filter_parts: sidebar_context_parts.append(f"Apply filters: {' AND '.join(filter_parts)}.")
                    if sidebar_context_parts: context_prefix = "Context: " + " ".join(sidebar_context_parts) + "\n\nUser Query: "
                    agent_input_text = f"{context_prefix}{prompt}"
                    agent_input = {"input": agent_input_text}
                    st.write(f"Invoking agent...") # Simplified log
                    response = agent_executor.invoke(agent_input)
                    st.write("Agent finished.")
                    status.update(label="Processing complete!", state="complete")

                # --- Process Agent Response ---
                final_content_to_display = {}
                agent_final_text = response.get('output', "Sorry, I couldn't retrieve a final answer.")
                final_content_to_display["text"] = agent_final_text

                plot_found, audit_data_found = None, None
                if "intermediate_steps" in response and response["intermediate_steps"]:
                    last_step = response["intermediate_steps"][-1]
                    if isinstance(last_step, tuple) and len(last_step) == 2:
                        observation = last_step[1]
                        if isinstance(observation, dict):
                            if "plot_json" in observation: plot_found = observation["plot_json"]
                            if "audit_dataframe" in observation: audit_data_found = observation["audit_dataframe"]

                if plot_found: final_content_to_display["plot_json"] = plot_found
                if audit_data_found: final_content_to_display["audit_dataframe"] = audit_data_found

                # --- Display final response ---
                if final_content_to_display.get("text"): message_placeholder.markdown(final_content_to_display["text"])
                else: message_placeholder.markdown("*(No text response generated)*")

                if final_content_to_display.get("plot_json"):
                    try:
                        fig = go.Figure(json.loads(final_content_to_display["plot_json"]))
                        # --- ADDED KEY ---
                        st.plotly_chart(fig, use_container_width=True, key=f"{current_response_key_prefix}_plot")
                    except Exception as e: st.error(f"Error displaying plot: {e}")

                if final_content_to_display.get("audit_dataframe"):
                    try:
                        audit_df = pd.DataFrame(final_content_to_display["audit_dataframe"])
                        expander_title = "Show Data Used for Response"
                        if final_content_to_display.get("plot_json"): expander_title = "Show Data Used for Plot"
                        elif isinstance(final_content_to_display.get("text"), str) and final_content_to_display["text"].startswith("Summary of"): expander_title = "Show Data Used for Summary"
                        with st.expander(expander_title):
                             # --- ADDED KEY ---
                            st.dataframe(audit_df, height=200, key=f"{current_response_key_prefix}_audit")
                    except Exception as e: st.error(f"Error displaying audit data: {e}")

                full_response_content = final_content_to_display

            else:
                 message_placeholder.error("Agent could not be initialized. Check configuration.")
                 full_response_content["text"] = "Error: Agent initialization failed."

        except Exception as e:
            st.exception(e)
            message_placeholder.error(f"An error occurred during agent execution: {e}")
            full_response_content["text"] = f"Error during agent execution: {e}"

        finally:
             if INTERMEDIATE_DF_KEY in st.session_state: del st.session_state[INTERMEDIATE_DF_KEY]

    st.session_state.ai_insights_messages.append({"role": "assistant", "content": full_response_content})
    # Rerun necessary after adding message to history to display it and clear input box
    st.rerun() # Moved rerun here to ensure it happens after state update

# **** END OF FULL SCRIPT ****