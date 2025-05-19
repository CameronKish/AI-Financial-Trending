# pages/3_AI_Insights.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import time
from langchain_core.agents import AgentAction, AgentFinish 
from langchain_core.messages import HumanMessage, AIMessage 
import utils 
from agent import create_financial_agent 
from tools import INTERMEDIATE_DF_KEY, _get_dataframe 
from datetime import date, timedelta


st.set_page_config(layout="wide", page_title="AI Insights")
st.markdown(f"<style> h1 {{ color: {utils.EY_DARK_BLUE_GREY}; }} </style>", unsafe_allow_html=True)
st.markdown(f"<h1>🤖 AI Insights (Chat with Data)</h1>", unsafe_allow_html=True)
st.caption("Ask questions about your P&L or Journal Entry data. Use the sidebars to apply filters and add contextual hints from JE data.")

utils.ensure_settings_loaded() 

data_is_available_for_agent = st.session_state.get('uploaded_data_ready', False) or \
                              st.session_state.get('data_loaded', False)

if not data_is_available_for_agent:
    st.error("Data not loaded. Please go to the Home page (for sample data) or 'Data Upload & Validation' page to upload your data.")
    st.stop()

# --- Initialize State Keys for this page ---
if "ai_insights_messages" not in st.session_state: st.session_state.ai_insights_messages = []
# Direct Data Filters
if 'ai_je_scope_filter' not in st.session_state: st.session_state.ai_je_scope_filter = "All JEs"
if 'ai_filter_start_date' not in st.session_state: st.session_state.ai_filter_start_date = None
if 'ai_filter_end_date' not in st.session_state: st.session_state.ai_filter_end_date = None
# Contextual Hints (Now always from JE data)
if 'ai_hint_columns_selected' not in st.session_state: st.session_state.ai_hint_columns_selected = []
if 'ai_hint_values_selected' not in st.session_state: st.session_state.ai_hint_values_selected = {}


# --- Sidebar Content ---
st.sidebar.header("AI Insights Filters & Context")

# --- 1. JE Data Scope Filter (Direct Data Filter) ---
st.sidebar.subheader("JE Data Scope Filter")
je_scope_options = ["All JEs", "P&L JEs Only", "BS JEs Only"]
# Ensure current state is valid, default if not
current_je_scope_state = st.session_state.ai_je_scope_filter
if current_je_scope_state not in je_scope_options:
    current_je_scope_state = je_scope_options[0]
    st.session_state.ai_je_scope_filter = current_je_scope_state

selected_je_scope = st.sidebar.radio(
    "Apply to Journal Entry Data:", options=je_scope_options,
    index=je_scope_options.index(current_je_scope_state), # Use validated state
    key="ai_je_scope_radio_v3_final" 
)
if selected_je_scope != st.session_state.ai_je_scope_filter:
    st.session_state.ai_je_scope_filter = selected_je_scope
    st.session_state.ai_hint_columns_selected = [] 
    st.session_state.ai_hint_values_selected = {}
    st.rerun()

# --- 2. Date Range Filter for Agent Data (Direct Data Filter) ---
st.sidebar.subheader("Date Filter for Agent Data")
min_ai_date_dt_overall, max_ai_date_dt_overall = (date.today() - timedelta(days=365*3)), date.today() 
temp_ai_start_peek = st.session_state.ai_filter_start_date
temp_ai_end_peek = st.session_state.ai_filter_end_date
temp_je_scope_peek_for_date = st.session_state.ai_je_scope_filter 
st.session_state.ai_filter_start_date = None 
st.session_state.ai_filter_end_date = None
# For peeking the widest possible date range, temporarily set JE scope to "All JEs"
st.session_state.ai_je_scope_filter = "All JEs" 
all_agent_dates_for_range_calc = []
try:
    df_peek_pl_sidebar = _get_dataframe("P&L") 
    if not df_peek_pl_sidebar.empty and 'Date' in df_peek_pl_sidebar.columns and pd.api.types.is_datetime64_any_dtype(df_peek_pl_sidebar['Date']):
        all_agent_dates_for_range_calc.extend(df_peek_pl_sidebar['Date'].dropna())
    df_peek_je_sidebar = _get_dataframe("JE") 
    if not df_peek_je_sidebar.empty and 'Transaction Date' in df_peek_je_sidebar.columns and pd.api.types.is_datetime64_any_dtype(df_peek_je_sidebar['Transaction Date']):
        all_agent_dates_for_range_calc.extend(df_peek_je_sidebar['Transaction Date'].dropna())
    if all_agent_dates_for_range_calc:
        min_val_from_data = min(all_agent_dates_for_range_calc).date()
        max_val_from_data = max(all_agent_dates_for_range_calc).date()
        if not pd.isna(min_val_from_data): min_ai_date_dt_overall = min_val_from_data
        if not pd.isna(max_val_from_data): max_ai_date_dt_overall = max_val_from_data
except Exception: pass 
st.session_state.ai_filter_start_date = temp_ai_start_peek 
st.session_state.ai_filter_end_date = temp_ai_end_peek
st.session_state.ai_je_scope_filter = temp_je_scope_peek_for_date # Restore actual JE scope

start_value_candidate_sidebar = st.session_state.ai_filter_start_date if st.session_state.ai_filter_start_date else min_ai_date_dt_overall
end_value_candidate_sidebar = st.session_state.ai_filter_end_date if st.session_state.ai_filter_end_date else max_ai_date_dt_overall
final_start_value_sidebar = max(min_ai_date_dt_overall, min(pd.to_datetime(start_value_candidate_sidebar).date(), max_ai_date_dt_overall))
final_end_value_sidebar = min(max_ai_date_dt_overall, max(pd.to_datetime(end_value_candidate_sidebar).date(), min_ai_date_dt_overall))
if final_start_value_sidebar > final_end_value_sidebar: final_start_value_sidebar = final_end_value_sidebar 

ai_col_start_sidebar, ai_col_end_sidebar = st.sidebar.columns(2)
ai_new_start_val_sidebar = ai_col_start_sidebar.date_input("Start Date", value=final_start_value_sidebar,
                                       min_value=min_ai_date_dt_overall, max_value=max_ai_date_dt_overall, 
                                       key="ai_start_date_widget_final_v3")
ai_new_end_val_sidebar = ai_col_end_sidebar.date_input("End Date", value=final_end_value_sidebar,
                                     min_value=ai_new_start_val_sidebar, max_value=max_ai_date_dt_overall, 
                                     key="ai_end_date_widget_final_v3")
if st.sidebar.button("Reset Date Range", key="ai_reset_date_button_final_v3"):
    st.session_state.ai_filter_start_date = min_ai_date_dt_overall 
    st.session_state.ai_filter_end_date = max_ai_date_dt_overall; st.rerun()
if ai_new_start_val_sidebar != st.session_state.get('ai_filter_start_date') or \
   ai_new_end_val_sidebar != st.session_state.get('ai_filter_end_date'):
    st.session_state.ai_filter_start_date = ai_new_start_val_sidebar
    st.session_state.ai_filter_end_date = ai_new_end_val_sidebar; st.rerun()
st.sidebar.markdown("---")

# --- 3. Simple Contextual Hints for Prompt (Always from JE Data) ---
st.sidebar.subheader("Add Textual Hints from JE Data to Prompt")
st.sidebar.caption("Hints are based on JE data after JE Scope and Date filters are applied.")

# Get the currently filtered JE data to populate hint options
df_for_je_hints = _get_dataframe("JE") # This respects JE Scope and Date filters

available_hint_cols_for_select = []
if df_for_je_hints is not None and not df_for_je_hints.empty:
    for col in df_for_je_hints.columns:
        if df_for_je_hints[col].dtype == 'object' or pd.api.types.is_string_dtype(df_for_je_hints[col]):
            nunique = df_for_je_hints[col].nunique(dropna=True)
            # Adjust heuristic: allow more unique values for hints, as user will pick specific ones
            if 1 < nunique < 500: # Increased upper limit
                available_hint_cols_for_select.append(col)
    available_hint_cols_for_select = sorted(available_hint_cols_for_select)

if not available_hint_cols_for_select:
    st.sidebar.caption("No suitable columns found in current JE data for hinting.")
else:
    current_hint_cols_selected = st.session_state.get('ai_hint_columns_selected', [])
    valid_hint_cols_selected = [col for col in current_hint_cols_selected if col in available_hint_cols_for_select]
    
    new_hint_cols_selected = st.sidebar.multiselect(
        "Select JE Columns for Hint Text:", options=available_hint_cols_for_select,
        default=valid_hint_cols_selected,
        key="ai_hint_columns_multiselect_v3_final" # Unique key
    )
    if new_hint_cols_selected != st.session_state.ai_hint_columns_selected:
        st.session_state.ai_hint_columns_selected = new_hint_cols_selected
        st.session_state.ai_hint_values_selected = {
            k: v for k, v in st.session_state.ai_hint_values_selected.items() if k in new_hint_cols_selected
        }
        st.rerun()

    if st.session_state.ai_hint_columns_selected and df_for_je_hints is not None and not df_for_je_hints.empty:
        st.sidebar.caption("Filter by specific values for hints (optional):")
        new_hint_values_selected_dict = {}
        for col_to_filter_hint in st.session_state.ai_hint_columns_selected:
            if col_to_filter_hint in df_for_je_hints.columns:
                unique_vals_hint = sorted(df_for_je_hints[col_to_filter_hint].dropna().astype(str).unique().tolist())
                if unique_vals_hint: # Only show multiselect if there are options
                    current_val_sel_hint = st.session_state.ai_hint_values_selected.get(col_to_filter_hint, [])
                    valid_val_sel_hint = [val for val in current_val_sel_hint if val in unique_vals_hint]
                    
                    selected_values_hint = st.sidebar.multiselect(
                        f"Values for '{col_to_filter_hint}':", options=unique_vals_hint, default=valid_val_sel_hint,
                        key=f"ai_hint_val_filter_{col_to_filter_hint.replace(' ', '_')}_v3_final" 
                    )
                    if selected_values_hint: # Only store if something is selected
                        new_hint_values_selected_dict[col_to_filter_hint] = selected_values_hint
        
        if new_hint_values_selected_dict != st.session_state.ai_hint_values_selected:
            st.session_state.ai_hint_values_selected = new_hint_values_selected_dict
            st.rerun()
st.sidebar.markdown("---")


# --- Display Chat History --- 
for idx, message in enumerate(st.session_state.ai_insights_messages):
    with st.chat_message(message["role"]): 
        if "text" in message["content"] and message["content"]["text"]: st.markdown(message["content"]["text"])
        if "plot_json" in message["content"] and message["content"]["plot_json"]:
            try: st.plotly_chart(go.Figure(json.loads(message["content"]["plot_json"])),use_container_width=True,key=f"hist_plot_ai_{idx}_v7")
            except Exception as e: st.error(f"Err display plot: {e}")
        if "audit_dataframe" in message["content"] and message["content"]["audit_dataframe"]:
            try:
                audit_df = pd.DataFrame(message["content"]["audit_dataframe"]); exp_title = "Show Data Used"
                if "plot_json" in message["content"]: exp_title = "Data for Plot"
                elif "text" in message["content"] and str(message["content"].get("text","")).startswith("Summary"): exp_title = "Data for Summary"
                with st.expander(exp_title): st.dataframe(audit_df, height=200, key=f"hist_audit_ai_{idx}_v7")
            except Exception as e: st.error(f"Err display audit: {e}")


# --- Handle Chat Input ---
if prompt := st.chat_input("Ask a question about P&L or JE data..."):
    st.session_state.ai_insights_messages.append({"role": "user", "content": {"text": prompt}})
    with st.chat_message("user"): st.markdown(prompt)

    current_response_key_prefix = f"response_ai_{len(st.session_state.ai_insights_messages)}"
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response_content = {"text": None, "plot_json": None, "audit_dataframe": None}
        try:
            if INTERMEDIATE_DF_KEY in st.session_state: del st.session_state[INTERMEDIATE_DF_KEY]
            agent_executor = create_financial_agent()
            if agent_executor:
                with st.status("AI is thinking and analyzing data...", expanded=True) as status:
                    st.write("Initializing agent...")
                    context_parts_for_prompt = []
                    current_je_scope_prompt = st.session_state.get('ai_je_scope_filter', "All JEs")
                    if current_je_scope_prompt != "All JEs":
                        context_parts_for_prompt.append(f"JE Data Scope: Analyzing '{current_je_scope_prompt}'.")
                    s_date_ai_prompt = st.session_state.get('ai_filter_start_date')
                    e_date_ai_prompt = st.session_state.get('ai_filter_end_date')
                    if s_date_ai_prompt or e_date_ai_prompt: 
                        s_date_str_ai_prompt = s_date_ai_prompt.strftime('%Y-%m-%d') if s_date_ai_prompt else "any date"
                        e_date_str_ai_prompt = e_date_ai_prompt.strftime('%Y-%m-%d') if e_date_ai_prompt else "any date"
                        context_parts_for_prompt.append(f"Date Range for Analysis: From {s_date_str_ai_prompt} to {e_date_str_ai_prompt}.")
                    
                    hint_values_selected_prompt = st.session_state.get('ai_hint_values_selected', {})
                    if hint_values_selected_prompt:
                        hint_clauses = []
                        for col, vals in hint_values_selected_prompt.items():
                            if vals: hint_clauses.append(f"'{col}' is one of [{', '.join(map(str, vals))}]")
                        if hint_clauses:
                            context_parts_for_prompt.append(f"Additional Hints from JE Data: Focus on entries where " + " AND ".join(hint_clauses) + ".")
                    
                    final_context_str = ""
                    if context_parts_for_prompt:
                        final_context_str = "[ACTIVE FILTERS & HINTS FOR YOUR ANALYSIS:\n" + "\n".join(context_parts_for_prompt) + "\n]\n\n"
                    final_prompt_for_agent = f"{final_context_str}USER QUERY: {prompt}"
                    status.update(label=f"Agent processing query...")
                                        
                    langchain_chat_history = []
                    for msg_hist_item in st.session_state.ai_insights_messages[:-1]:
                        role = msg_hist_item.get("role"); content_dict = msg_hist_item.get("content", {}); text_content = content_dict.get("text", "")
                        if role == "user": langchain_chat_history.append(HumanMessage(content=text_content))
                        elif role == "assistant": langchain_chat_history.append(AIMessage(content=text_content))
                    
                    response = agent_executor.invoke({"input": final_prompt_for_agent, "chat_history": langchain_chat_history})
                    status.update(label="Processing complete!", state="complete")

                final_content_to_display = {} 
                agent_final_text = response.get('output', "Sorry, I couldn't retrieve a final answer.")
                final_content_to_display["text"] = agent_final_text; plot_found, audit_data_found = None, None
                if "intermediate_steps" in response and response["intermediate_steps"]:
                    for step in response["intermediate_steps"]:
                        if isinstance(step, tuple) and len(step) == 2:
                            observation = step[1]
                            if isinstance(observation, dict):
                                if "plot_json" in observation: plot_found = observation["plot_json"]
                                if "audit_dataframe" in observation: audit_data_found = observation["audit_dataframe"]
                if plot_found: final_content_to_display["plot_json"] = plot_found
                if audit_data_found: final_content_to_display["audit_dataframe"] = audit_data_found
                if final_content_to_display.get("text"): message_placeholder.markdown(final_content_to_display["text"])
                else: message_placeholder.markdown("*(No text response generated)*")
                if final_content_to_display.get("plot_json"):
                    try: st.plotly_chart(go.Figure(json.loads(final_content_to_display["plot_json"])), use_container_width=True, key=f"{current_response_key_prefix}_plot_ai_v7")
                    except Exception as e: st.error(f"Error displaying plot: {e}")
                if final_content_to_display.get("audit_dataframe"):
                    try:
                        audit_df = pd.DataFrame(final_content_to_display["audit_dataframe"]); exp_title = "Show Data Used"
                        if "plot_json" in final_content_to_display: exp_title = "Data for Plot"
                        elif "text" in final_content_to_display and str(final_content_to_display.get("text","")).startswith("Summary"): exp_title = "Data for Summary"
                        with st.expander(exp_title): st.dataframe(audit_df, height=200, key=f"{current_response_key_prefix}_audit_ai_v7")
                    except Exception as e: st.error(f"Error displaying audit data: {e}")
                full_response_content = final_content_to_display
            else:
                 message_placeholder.error("Agent could not be initialized. Check LLM configuration on Home page.")
                 full_response_content["text"] = "Error: Agent initialization failed."
        except Exception as e:
            st.exception(e); message_placeholder.error(f"An error occurred: {e}"); full_response_content["text"] = f"Error: {e}"
        finally:
             if INTERMEDIATE_DF_KEY in st.session_state: del st.session_state[INTERMEDIATE_DF_KEY]
    st.session_state.ai_insights_messages.append({"role": "assistant", "content": full_response_content})
    st.rerun()