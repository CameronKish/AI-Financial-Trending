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
st.caption("Ask questions about your P&L or Journal Entry data.")

# --- Ensure Settings & Data are Loaded ---
utils.ensure_settings_loaded()

if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.error("Data not loaded. Please go back to the Home page and ensure data is loaded correctly.")
    st.stop()

if 'pl_flat_df' not in st.session_state or 'je_detail_df' not in st.session_state:
     st.error("Financial dataframes not found in session state. Please reload data.")
     st.stop()

# --- Initialize Chat History ---
if "ai_insights_messages" not in st.session_state:
    st.session_state.ai_insights_messages = []

# --- Display Prior Chat Messages ---
for message in st.session_state.ai_insights_messages:
    with st.chat_message(message["role"]):
        # Render message content (text, plots, dataframes)
        if "text" in message["content"] and message["content"]["text"]:
            st.markdown(message["content"]["text"])
        if "plot_json" in message["content"] and message["content"]["plot_json"]:
            try:
                fig = go.Figure(json.loads(message["content"]["plot_json"]))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying previous plot: {e}")
        # Display audit data if it exists (even if only text was generated)
        if "audit_dataframe" in message["content"] and message["content"]["audit_dataframe"]:
            try:
                audit_df = pd.DataFrame(message["content"]["audit_dataframe"])
                expander_title = "Show Data Used for Response"
                if "plot_json" in message["content"] and message["content"]["plot_json"]:
                    expander_title = "Show Data Used for Plot"
                elif "text" in message["content"] and isinstance(message["content"].get("text"), str) and message["content"]["text"].startswith("Summary of"):
                     expander_title = "Show Data Used for Summary" # Keep this check for historical display

                with st.expander(expander_title):
                    st.dataframe(audit_df, height=200)
            except Exception as e:
                st.error(f"Error displaying previous audit data: {e}")


# --- Handle Chat Input ---
if prompt := st.chat_input("Ask a question (e.g., 'What were the top 5 customers by JE amount in 2023-01?', 'Plot P&L for Total Net Sales')"):
    st.session_state.ai_insights_messages.append({"role": "user", "content": {"text": prompt}})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        # Initialize structure for the response content to be stored in history
        full_response_content = {"text": None, "plot_json": None, "audit_dataframe": None}

        try:
            if INTERMEDIATE_DF_KEY in st.session_state:
                del st.session_state[INTERMEDIATE_DF_KEY]

            agent_executor = create_financial_agent()

            if agent_executor:
                with st.status("Thinking and processing...", expanded=False) as status:
                    st.write("Initializing agent...")
                    agent_input = {"input": prompt}
                    st.write(f"Invoking agent with input: {prompt[:100]}...")
                    response = agent_executor.invoke(agent_input)
                    st.write("Agent finished.")
                    status.update(label="Processing complete!", state="complete")

                # --- Process Agent Response - REVISED LOGIC ---
                final_content_to_display = {}

                # 1. Always get the primary text response from the agent's final output
                agent_final_text = response.get('output', "Sorry, I couldn't retrieve a final answer.")
                final_content_to_display["text"] = agent_final_text

                # 2. Check intermediate steps ONLY for supplementary plot or audit data
                plot_found = None
                audit_data_found = None
                if "intermediate_steps" in response and response["intermediate_steps"]:
                    last_step = response["intermediate_steps"][-1]
                    if isinstance(last_step, tuple) and len(last_step) == 2:
                        observation = last_step[1] # Get the result of the last tool call
                        if isinstance(observation, dict):
                            # Check specifically for plot_json
                            if "plot_json" in observation:
                                plot_found = observation["plot_json"]
                            # Always grab audit data if the tool provided it
                            if "audit_dataframe" in observation:
                                audit_data_found = observation["audit_dataframe"]
                            # If the last tool reported an error, append it to the main text (optional)
                            # if "error" in observation:
                            #     final_content_to_display["text"] += f"\n\nNote: The last tool reported an error: {observation['error']}"


                # Add supplementary content if found
                if plot_found:
                    final_content_to_display["plot_json"] = plot_found
                if audit_data_found:
                     final_content_to_display["audit_dataframe"] = audit_data_found

                # --- Display final response ---
                # Display text first
                if final_content_to_display.get("text"):
                    message_placeholder.markdown(final_content_to_display["text"])
                else:
                     message_placeholder.markdown("*(No text response generated)*") # Should not happen often

                # Display plot if it exists
                if final_content_to_display.get("plot_json"):
                    try:
                        fig = go.Figure(json.loads(final_content_to_display["plot_json"]))
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error displaying plot: {e}")

                # Display audit data if it exists
                if final_content_to_display.get("audit_dataframe"):
                    try:
                        audit_df = pd.DataFrame(final_content_to_display["audit_dataframe"])
                        # Determine title based on what was generated primarily
                        expander_title = "Show Data Used for Response"
                        if final_content_to_display.get("plot_json"):
                             expander_title = "Show Data Used for Plot"
                        # Maybe check if the final text suggests a summary was the goal? Less reliable.
                        # elif isinstance(final_content_to_display.get("text"), str) and "summary of" in final_content_to_display["text"].lower():
                        #      expander_title = "Show Data Used for Summary"

                        with st.expander(expander_title):
                            st.dataframe(audit_df, height=200)
                    except Exception as e:
                        st.error(f"Error displaying audit data: {e}")

                # Store whatever was displayed into history
                full_response_content = final_content_to_display

            else:
                 message_placeholder.error("Agent could not be initialized. Check configuration.")
                 full_response_content["text"] = "Error: Agent initialization failed."

        except Exception as e:
            st.exception(e)
            message_placeholder.error(f"An error occurred during agent execution: {e}")
            full_response_content["text"] = f"Error during agent execution: {e}"

        finally:
             if INTERMEDIATE_DF_KEY in st.session_state:
                 del st.session_state[INTERMEDIATE_DF_KEY]

    st.session_state.ai_insights_messages.append({"role": "assistant", "content": full_response_content})

# **** END OF FULL SCRIPT ****