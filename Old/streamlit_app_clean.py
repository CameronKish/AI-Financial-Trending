# pages/1_P&L_Analysis_&_Drilldown.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Import shared utilities and data processing functions
import utils
import data_processor
from prompts import get_pnl_analysis_prompt

# --- Page Setup ---
st.set_page_config(layout="wide", page_title="P&L Analysis")
st.markdown(f"<style> h1 {{ color: {utils.EY_DARK_BLUE_GREY}; }} </style>", unsafe_allow_html=True)

# --- Check if data is loaded ---
if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.error("Data not loaded. Please go back to the main page.")
    st.stop()

# --- Retrieve data from Session State ---
try:
    pl_flat_df = st.session_state.pl_flat_df; je_detail_df = st.session_state.je_detail_df
    col_config = st.session_state.column_config
    PL_ID_COLUMN = col_config["PL_ID"]; PL_MAP_COLUMN = col_config["PL_MAP_DISPLAY"]
    JE_ID_COLUMN = col_config["JE_ID"]; JE_DATE_COLUMN = col_config["JE_DATE"]
    JE_AMOUNT_COLUMN = col_config["JE_AMOUNT"]; JE_ACCOUNT_NAME_COL = col_config["JE_ACCOUNT_NAME"]
except KeyError as e: st.error(f"Missing data/config: {e}. Reload app."); st.stop()

# --- Initialize/Ensure necessary session state keys ---
if 'llm_analyses' not in st.session_state: st.session_state.llm_analyses = {}
if 'llm_streaming_key' not in st.session_state: st.session_state.llm_streaming_key = None
if 'selected_account_id' not in st.session_state: st.session_state.selected_account_id = None
if 'selected_account_name' not in st.session_state: st.session_state.selected_account_name = None
if 'selected_period' not in st.session_state: st.session_state.selected_period = None
if 'related_jes_df' not in st.session_state: st.session_state.related_jes_df = pd.DataFrame(columns=col_config.get("JE_DETAILS_BASE", []))
# Ensure prev_ keys exist
if 'prev_selected_account_id' not in st.session_state: st.session_state.prev_selected_account_id = None
if 'prev_selected_period' not in st.session_state: st.session_state.prev_selected_period = None


# --- Prepare P&L Data (Pivot, Diff, StdDev, Options) ---
try:
    # Ensure Period columns are correct for pivoting
    if 'Period_dt' not in pl_flat_df.columns or pl_flat_df['Period_dt'].isnull().any():
         pl_flat_df['Period_dt'] = pd.to_datetime(pl_flat_df['Period'], errors='coerce')
         # Use apply with error handling for string conversion
         pl_flat_df['Period_Str'] = pl_flat_df.apply(lambda r: r['Period_dt'].strftime('%Y-%m') if pd.notna(r['Period_dt']) else str(r['Period']), axis=1)
         period_col_for_pivot = 'Period_Str'
    else:
         pl_flat_df['Period_Str'] = pl_flat_df['Period_dt'].dt.strftime('%Y-%m'); period_col_for_pivot = 'Period_Str'

    pnl_wide_view_df = pl_flat_df.pivot_table(index=[PL_ID_COLUMN, PL_MAP_COLUMN], columns=period_col_for_pivot, values='Amount').fillna(0)
    pnl_wide_view_df = pnl_wide_view_df.sort_index(axis=1) # Sort columns (periods) chronologically

    # Calculate differences and standard deviations for highlighting
    diff_df = pnl_wide_view_df.diff(axis=1); row_std_diff = pnl_wide_view_df.diff(axis=1).std(axis=1, skipna=True).fillna(0)

    # Prepare for table display and selections
    pnl_wide_view_df_reset = pnl_wide_view_df.reset_index() # Keep a resettable version for lookup if needed

    # Get options for widgets/mapping
    account_options = sorted(pnl_wide_view_df.index.get_level_values(PL_MAP_COLUMN).unique().tolist())
    unique_indices = pnl_wide_view_df.index.unique(); # Get unique multi-index tuples (ID, Name)
    # Robust mapping creation handling potential non-hashable types (though unlikely with IDs/Names)
    account_name_to_id_map = {n: i for i, n in unique_indices if isinstance(i,(str,int,float)) and isinstance(n,(str,int,float))}
    account_id_to_name_map = {i: n for i, n in unique_indices if isinstance(i,(str,int,float)) and isinstance(n,(str,int,float))}
    period_options = pnl_wide_view_df.columns.tolist()
except Exception as e: st.error(f"Error preparing P&L data: {e}"); st.exception(e); st.stop()


# --- Sidebar Elements ---

# LLM Analysis History
st.sidebar.header("LLM Analysis History")
if st.sidebar.button("Clear History", key='clear_llm_hist_btn'):
    st.session_state.llm_analyses = {}; st.session_state.llm_streaming_key = None; st.rerun()
if 'llm_analyses' in st.session_state and st.session_state.llm_analyses:
    analysis_keys = list(st.session_state.llm_analyses.keys()); MAX_HISTORY = 10
    st.sidebar.caption(f"Showing latest {min(len(analysis_keys), MAX_HISTORY)}:")
    for key in reversed(analysis_keys[-MAX_HISTORY:]):
         try:
             acc_id, acc_name, period = key; result = st.session_state.llm_analyses[key]
             is_streaming = (st.session_state.llm_streaming_key == key)
             expander_label = f"{acc_name} - {period}" + (" (Processing...)" if is_streaming else "")
             with st.sidebar.expander(expander_label):
                 if is_streaming: st.info("🔄 Processing...")
                 st.markdown(result if result else "_Analysis initiated..._")
         except Exception as e_hist: st.sidebar.warning(f"Err display hist {key}: {e_hist}")
else: st.sidebar.caption("No analyses run yet.")
st.sidebar.markdown("---")

# Outlier Sensitivity Slider
st.sidebar.header("P&L Controls")
threshold_std_dev = st.sidebar.slider("Outlier Sensitivity", 1.0, 4.0, st.session_state.get('outlier_threshold', 2.0), 0.1, key='outlier_threshold')


# --- Page Content Area ---

st.markdown(f"<h1>P&L Analysis & JE Drilldown</h1>", unsafe_allow_html=True)

# --- Filters moved to main area ---
st.markdown("#### Select Account & Period")
col1, col2 = st.columns(2)

# Use session state to determine initial index for filters
current_account_name = st.session_state.get('selected_account_name')
current_period = st.session_state.get('selected_period')
account_index = utils.get_index(account_options, current_account_name)
period_index = utils.get_index(period_options, current_period)

with col1:
    widget_account_name = st.selectbox(
        "GL Account:", options=account_options, index=account_index, key="widget_account_select"
    )
with col2:
    widget_period = st.selectbox(
        "Period:", options=period_options, index=period_index, key="widget_period_select"
    )

# --- Update Central State Based on Widgets (if changed by user) ---
widget_account_id = account_name_to_id_map.get(widget_account_name)
# Check if widgets caused a change compared to the central state
if (widget_account_id != st.session_state.get('selected_account_id') or
    widget_period != st.session_state.get('selected_period')):
    st.session_state.selected_account_id = widget_account_id
    st.session_state.selected_account_name = widget_account_name
    st.session_state.selected_period = widget_period
    # Reset dependent states that need re-fetching or clearing
    st.session_state.related_jes_df = pd.DataFrame(columns=col_config.get("JE_DETAILS_BASE", [])) # Clear JEs
    st.session_state.dup_col = None; st.session_state.dup_val = None; st.session_state.dup_search_triggered = False
    st.session_state.llm_streaming_key = None
    # No explicit rerun needed here, Streamlit reruns automatically on widget change


# --- Process Table Selection --- (MODIFIED: ADDED st.rerun())
table_selection_state = st.session_state.get("pnl_select_df", {"rows": [], "columns": []})
selected_rows_indices = table_selection_state.get('rows', [])
selected_cols_names = table_selection_state.get('columns', [])

if selected_rows_indices and selected_cols_names:
    try:
        selected_row_pos_index = selected_rows_indices[0] # Get positional index from selection state
        selected_row_data = pnl_wide_view_df_reset.iloc[selected_row_pos_index] # Use iloc on reset df

        table_account_id = str(selected_row_data[PL_ID_COLUMN]);
        table_account_name = selected_row_data[PL_MAP_COLUMN];
        table_period = selected_cols_names[0] # Column name is the period

        # Check if table click represents a NEW selection compared to current central state
        if (table_account_id != st.session_state.get('selected_account_id') or
            table_period != st.session_state.get('selected_period')):
            # If it's new, update the central state
            st.session_state.selected_account_id = table_account_id
            st.session_state.selected_account_name = table_account_name
            st.session_state.selected_period = table_period
            # Reset dependent states
            st.session_state.related_jes_df = pd.DataFrame(columns=col_config.get("JE_DETAILS_BASE", []))
            st.session_state.dup_col = None; st.session_state.dup_val = None; st.session_state.dup_search_triggered = False
            st.session_state.llm_streaming_key = None

            # **** ADDED EXPLICIT RERUN ****
            st.rerun() # Force an immediate rerun after state update, mimicking old script

    except IndexError:
        st.warning("Could not process table selection (Index Error). Please ensure P&L data is loaded correctly.")
    except KeyError as e_proc_key:
        st.warning(f"Could not process table selection (Key Error: {e_proc_key}). Check column names.")
    except Exception as e_proc: st.warning(f"Table selection processing error: {e_proc}")


# --- Fetch Journal Entries Based on Comparison to Previous State --- (MODIFIED: RESTORED ORIGINAL LOGIC)
should_fetch = False
sel_id = st.session_state.get('selected_account_id')
sel_period = st.session_state.get('selected_period')
prev_id = st.session_state.get('prev_selected_account_id') # Read previous state
prev_period = st.session_state.get('prev_selected_period') # Read previous state

# ORIGINAL LOGIC: Fetch if selection is valid AND differs from the previous state recorded
if sel_id and sel_period:
    if (sel_id != prev_id or sel_period != prev_period):
         should_fetch = True

if should_fetch:
    # Optional Debug log: Uncomment to see when fetching is attempted
    # st.sidebar.info(f"DEBUG: Fetch Triggered (Original Logic) for {sel_id}, {sel_period}")
    if sel_id and sel_period: # Double check selection is still valid before fetching
        try:
            fetched_jes = data_processor.get_journal_entries(sel_id, sel_period, je_detail_df)
            st.session_state.related_jes_df = fetched_jes # Update JE data state

            # **CRUCIAL**: Update prev state *after* successful fetch
            st.session_state.prev_selected_account_id = sel_id
            st.session_state.prev_selected_period = sel_period

        except Exception as e_fetch:
            st.error(f"Error fetching JEs: {e_fetch}")
            st.session_state.related_jes_df = pd.DataFrame(columns=col_config.get("JE_DETAILS_BASE", [])) # Reset on error
            # Do not update prev state on error
    # else: # Optional: Handle case where should_fetch was True but selection became invalid
          # st.sidebar.warning("DEBUG: Fetch skipped, selection invalid.")


# --- P&L Overview Table Display ---
st.markdown("---")
st.markdown("#### P&L Data Overview")
st.caption(f"Highlighting indicates MoM change > {st.session_state.outlier_threshold:.1f} Std Dev. Click row+col for JE/LLM.")
try: # Apply styling and render dataframe
    # Calculate thresholds based on current slider value
    outlier_threshold_values = st.session_state.outlier_threshold * row_std_diff
    # Apply styling (ensure helper functions are robust)
    styled_df = pnl_wide_view_df.style.apply(
        utils.highlight_outliers_pandas,
        axis=1, # Apply row-wise
        diffs_df=diff_df,
        thresholds_series=outlier_threshold_values,
        color=utils.EY_YELLOW,
        text_color=utils.EY_TEXT_ON_YELLOW
    ).format("{:,.0f}") # Apply number formatting AFTER styling

    # Display the dataframe with selection enabled
    st.dataframe(
        styled_df,
        use_container_width=True,
        key="pnl_select_df", # Key used to access selection state
        on_select="rerun", # Trigger script rerun on selection
        selection_mode=("single-row", "single-column") # Allow selecting one row and one column
    )
except Exception as e_style: st.error(f"Error styling P&L: {e_style}"); st.exception(e_style); st.dataframe(pnl_wide_view_df.style.format("{:,.0f}"), use_container_width=True, key="pnl_select_df", on_select="rerun", selection_mode=("single-row", "single-column")) # Fallback


# --- Journal Entry Details Display ---
st.markdown(f"<hr><h2 style='color: {utils.EY_DARK_BLUE_GREY};'>Journal Entry Details</h2>", unsafe_allow_html=True)
# Read the data that *should* have been updated by the fetch logic above
related_jes_to_display = st.session_state.get('related_jes_df', pd.DataFrame())
# Read selection state again for display consistency (this is the state AFTER potential updates)
selected_id_disp = st.session_state.get('selected_account_id')
selected_name_disp = st.session_state.get('selected_account_name')
selected_period_disp = st.session_state.get('selected_period')

if selected_id_disp and selected_period_disp:
    st.write(f"Showing JEs for: **{selected_name_disp} ({selected_id_disp})** | Period: **{selected_period_disp}**")
    if isinstance(related_jes_to_display, pd.DataFrame) and not related_jes_to_display.empty:
         # Prepare JE table for display (formatting)
         je_display_df = related_jes_to_display.copy()
         # Identify amount columns for formatting
         je_amount_cols = [col for col in je_display_df.columns if 'Amount' in col or col == JE_AMOUNT_COLUMN]
         for col in je_amount_cols: je_display_df[col] = je_display_df[col].apply(utils.format_amount_safely)
         # Configure columns (especially dates)
         je_col_config = {}
         date_cols_to_format = [col for col in je_display_df.columns if 'Date' in col or col == JE_DATE_COLUMN]
         for col in date_cols_to_format:
             je_col_config[col] = st.column_config.DateColumn(label=col, format="YYYY-MM-DD") # Ensure consistent date format

         # Display JE dataframe
         st.dataframe(je_display_df, use_container_width=True, column_config=je_col_config, hide_index=True)

    elif isinstance(related_jes_to_display, pd.DataFrame) and related_jes_to_display.empty:
         # Check if the fetch was successful for this selection but yielded no results
         if (selected_id_disp == st.session_state.get('prev_selected_account_id') and
             selected_period_disp == st.session_state.get('prev_selected_period')):
             # If the current selection matches the last successful fetch state, and the table is empty,
             # it means the fetch ran successfully and found nothing.
            st.info(f"No JEs found for {selected_name_disp} in {selected_period_disp}.")
         # else: # If selection doesn't match previous, fetch might not have run or failed, don't show "No JEs" yet.
              # This case might be less likely now with the explicit rerun ensuring fetch happens.
              # Consider adding a message here if needed for debugging.
              # st.warning("DEBUG: JE display might be pending update...")
    else:
         # related_jes_to_display is not a DataFrame (e.g., None), indicates an error upstream
         st.warning("Journal Entry data is currently unavailable.")
else:
    st.info("Select Account/Period using the filters above or by clicking the P&L table.")


# --- LLM Analysis Section ---
st.markdown(f"<hr><h2 style='color: {utils.EY_DARK_BLUE_GREY};'>LLM Period Analysis</h2>", unsafe_allow_html=True)

# Define state vars for clarity
analysis_key = None
# Check against the JE data we intend to display
je_data_available = isinstance(related_jes_to_display, pd.DataFrame) and not related_jes_to_display.empty
# Read model directly from session state (set on Home page)
llm_model_currently_selected = st.session_state.get('selected_llm_model')

# Conditions to show the button: valid selection, model chosen, and JEs present for the selection
show_analysis_button = (selected_id_disp and selected_name_disp and selected_period_disp and
                        llm_model_currently_selected and je_data_available)

analysis_button_pressed = False # Reset flag each run

if show_analysis_button:
    analysis_key = (selected_id_disp, selected_name_disp, selected_period_disp) # Use display values for consistency
    button_label = f"🤖 Analyze Period Activity with {llm_model_currently_selected}"
    is_streaming_this = (st.session_state.llm_streaming_key == analysis_key)
    # Change button label if analysis exists and we are not currently streaming it
    if analysis_key in st.session_state.llm_analyses and not is_streaming_this:
        button_label = f"🔄 Re-analyze Period Activity with {llm_model_currently_selected}"

    # Disable button while streaming this specific analysis
    if st.button(button_label, key=f"llm_btn_{analysis_key}", disabled=is_streaming_this):
        analysis_button_pressed = True
        st.session_state.llm_streaming_key = analysis_key # Set flag to indicate streaming start
        # No explicit rerun needed here, button click causes rerun automatically

elif selected_id_disp and selected_period_disp: # Give feedback if button isn't shown but selection is made
    if not llm_model_currently_selected: st.warning("Select an LLM model on the Home page first.")
    elif not je_data_available: st.warning("No JE data available for this selection to analyze.")
    # If both model and JEs were available, the button would have shown.
else: st.info("Select Account/Period with JEs to enable LLM analysis.")

# --- Perform Analysis and Stream Output ---
# This block runs if the button was just pressed OR if already streaming for the current selection
current_streaming_key = st.session_state.get('llm_streaming_key')
# Check if the key matches the currently displayed selection
should_be_streaming_this = current_streaming_key and current_streaming_key == (selected_id_disp, selected_name_disp, selected_period_disp)

# Define placeholder for streaming output - always define it so it can be cleared if needed
analysis_output_placeholder = st.empty()

# Only proceed if we should be streaming for the currently selected/displayed item
if should_be_streaming_this:
    model_to_use = st.session_state.get('selected_llm_model') # Read fresh again inside block
    analysis_key_to_stream = current_streaming_key # This is the key = (id, name, period)

    if not model_to_use: # Critical check: Did the model selection get lost?
         analysis_output_placeholder.error("LLM Model configuration lost. Please re-select on the Home page.")
         st.session_state.llm_streaming_key = None # Stop streaming state
         # Consider adding st.rerun() if you want this error to clear immediately or update sidebar state
    else:
        # Prepare context for LLM (ensure using consistent selected_id_disp etc.)
        try:
            # Show initial status within the placeholder
            with analysis_output_placeholder.container(): st.info(f"🤖 Contacting {model_to_use} for analysis...")

            # --- Context Gathering ---
            # Use the selection state that triggered this stream
            account_id=analysis_key_to_stream[0]
            account_name=analysis_key_to_stream[1]
            current_period=analysis_key_to_stream[2]

            # Retrieve P&L amounts (handle potential errors)
            try: current_amount = pnl_wide_view_df.loc[(account_id, account_name), current_period]
            except KeyError: current_amount = None # Account/Period might not exist in P&L wide view? Should be unlikely.

            current_period_index = period_options.index(current_period) if current_period in period_options else -1
            previous_period = period_options[current_period_index - 1] if current_period_index > 0 else None
            previous_amount = None
            if previous_period:
                 try: previous_amount = pnl_wide_view_df.loc[(account_id, account_name), previous_period]
                 except KeyError: previous_amount = None # Previous period might not exist for this account

            # Use the JE data currently stored in session state (should be the correct one now)
            je_data_to_analyze = st.session_state.related_jes_df
            if not isinstance(je_data_to_analyze, pd.DataFrame) or je_data_to_analyze.empty:
                raise ValueError("JE data for analysis is missing or empty.")

            current_amount_str = utils.format_amount_safely(current_amount) if current_amount is not None else "N/A"
            previous_amount_str = utils.format_amount_safely(previous_amount) if previous_amount is not None else "N/A"

            # Format JE data for the prompt
            formatted_je_data = utils.format_je_for_llm(je_data_to_analyze, JE_AMOUNT_COLUMN)

            # Add outlier context if applicable
            is_outlier_context = "";
            try: # Optional outlier context - wrap in try/except for robustness
                 diff_val = diff_df.loc[(account_id, account_name), current_period];
                 std_dev_val = row_std_diff.loc[(account_id, account_name)]
                 threshold_mult = st.session_state.outlier_threshold;
                 threshold_val = threshold_mult * std_dev_val
                 # Check for NaN/infinite values and ensure std dev is meaningful
                 if pd.notna(diff_val) and pd.notna(std_dev_val) and np.isfinite(diff_val) and std_dev_val > 1e-6 and abs(diff_val) > threshold_val:
                      is_outlier_context = f"\nNote: This period's change from prior month might be considered an outlier based on historical volatility (Sensitivity: {threshold_mult:.1f} Std Dev)."
            except KeyError: pass # Ignore if account/period not found in diff/std dev data
            except Exception as e_outlier: st.warning(f"Outlier context error: {e_outlier}") # Log other errors

            # Generate the prompt
            prompt = get_pnl_analysis_prompt(
                account_name=account_name, account_id=account_id,
                current_period=current_period, current_amount_str=current_amount_str,
                previous_period=previous_period, previous_amount_str=previous_amount_str,
                formatted_je_data=formatted_je_data,
                is_outlier_context=is_outlier_context
            )

            # --- Stream response and store full result ---
            final_response = "" # Initialize empty string to accumulate response
            with analysis_output_placeholder.container():
                st.markdown("---") # Visual separator in the output area
                # Call the streaming function from utils
                ollama_generator = utils.call_ollama_stream(prompt, model_to_use)
                try:
                     # Use st.write_stream to display the generator output directly
                     streamed_content = st.write_stream(ollama_generator)
                     # Store the complete result (streamed_content holds the full string after completion)
                     final_response = streamed_content if streamed_content else "*(No content received)*"

                except Exception as stream_err:
                     st.error(f"LLM Streaming Error: {stream_err}")
                     final_response = f"**Error during analysis streaming:** {str(stream_err)}"

            # Store the complete result (or error message) in history dict
            st.session_state.llm_analyses[analysis_key_to_stream] = final_response

        except ValueError as ve: # Catch specific error from JE data check
            st.error(f"LLM analysis preparation error: {ve}")
            if analysis_key_to_stream: st.session_state.llm_analyses[analysis_key_to_stream] = f"Analysis setup error: {str(ve)}"
        except Exception as e: # Catch other general errors during prep/call
            st.error(f"LLM analysis preparation error: {e}"); st.exception(e)
            if analysis_key_to_stream: st.session_state.llm_analyses[analysis_key_to_stream] = f"Analysis setup error: {str(e)}"
        finally:
            # Clear the streaming key indicator *after* processing attempt (success or failure)
            st.session_state.llm_streaming_key = None
            # Rerun to update UI (e.g., disable button turns back to enabled, sidebar updates)
            st.rerun()


# --- Duplicate JE Finder ---
st.markdown(f"<hr><h2 style='color: {utils.EY_DARK_BLUE_GREY};'>Duplicate Value Lookup (Across All JEs)</h2>", unsafe_allow_html=True)
# Use JE data currently displayed for selecting the value to search for
je_data_for_dup_check = st.session_state.get('related_jes_df', pd.DataFrame())

if isinstance(je_data_for_dup_check, pd.DataFrame) and not je_data_for_dup_check.empty:
    # Define potential columns for duplicate checking (usually identifiers or text fields)
    potential_dup_cols = ['Customer', 'Memo', 'Transaction Id', JE_AMOUNT_COLUMN];
    # Check which of these columns exist in the *full* JE dataset (je_detail_df)
    available_dup_cols = [col for col in potential_dup_cols if col in je_detail_df.columns]

    if available_dup_cols:
        col1, col2 = st.columns(2);
        with col1:
            last_dup_col_index = utils.get_index(available_dup_cols, st.session_state.get('dup_col'))
            selected_dup_col = st.selectbox("Check Column:", available_dup_cols, index=last_dup_col_index, key='dup_col_select')
        with col2:
            value_options = []; last_dup_val = st.session_state.get('dup_val'); dup_val_index = 0
            # Get unique values from the *currently displayed* JEs for the selected column
            if selected_dup_col and selected_dup_col in je_data_for_dup_check.columns:
                 # Ensure options are hashable and sorted for consistent display
                 try: value_options = sorted(list(je_data_for_dup_check[selected_dup_col].dropna().unique()))
                 except TypeError: value_options = list(je_data_for_dup_check[selected_dup_col].dropna().unique()) # Fallback if sorting fails

            # Find index of last selected value safely
            try: dup_val_index = utils.get_index(value_options, last_dup_val) if value_options else 0
            except Exception: dup_val_index = 0 # Default to first if error

            selected_dup_val = st.selectbox(f"Value from Current JEs:", value_options, index=dup_val_index, key='dup_val_select', disabled=(not value_options))

        # Button to trigger the search across the full dataset
        find_duplicates_button = st.button("Find All Occurrences", key='find_dup_btn')

        if find_duplicates_button and selected_dup_col and selected_dup_val is not None:
            # Store selection and set trigger flag on button press
            st.session_state.dup_col = selected_dup_col; st.session_state.dup_val = selected_dup_val; st.session_state.dup_search_triggered = True;
            st.rerun() # Rerun immediately to perform the search in the next cycle

        # Search logic executes if trigger flag is set
        if st.session_state.get('dup_search_triggered'):
            col_to_check = st.session_state.dup_col; val_to_find = st.session_state.dup_val;
            st.write(f"Finding all JEs where **{col_to_check}** is **'{val_to_find}'**...")
            if col_to_check and val_to_find is not None:
                try: # --- Robust Duplicate Filtering Logic ---
                    # Use the full je_detail_df for searching
                    target_col = je_detail_df[col_to_check]
                    target_dtype = target_col.dtype

                    # Handle NaN comparison correctly
                    if pd.isna(val_to_find):
                        duplicate_jes_df = je_detail_df[target_col.isna()]
                    # Handle numeric comparison (including floats with tolerance)
                    elif pd.api.types.is_numeric_dtype(target_dtype):
                        try: val_num = pd.to_numeric(val_to_find) # Convert search value to numeric
                        except (ValueError, TypeError): raise ValueError(f"Value '{val_to_find}' is not compatible with numeric column '{col_to_check}'.")

                        if pd.api.types.is_float_dtype(target_dtype) or isinstance(val_num, float):
                            # Use tolerance for float comparison
                            duplicate_jes_df = je_detail_df[np.isclose(target_col.astype(float).fillna(np.nan), float(val_num), atol=1e-6, equal_nan=True)]
                        else:
                            # Use direct comparison for integers (after ensuring types match)
                             duplicate_jes_df = je_detail_df[target_col.fillna(pd.NA).astype(pd.Int64Dtype()) == int(val_num)]
                    # Handle datetime comparison (compare dates only by default)
                    elif pd.api.types.is_datetime64_any_dtype(target_dtype):
                         val_dt = pd.to_datetime(val_to_find, errors='coerce')
                         if pd.notna(val_dt):
                            duplicate_jes_df = je_detail_df[pd.to_datetime(target_col).dt.normalize() == val_dt.normalize()]
                         else: # If conversion fails, try string comparison as fallback
                            duplicate_jes_df = je_detail_df[target_col.astype(str).str.strip() == str(val_to_find).strip()]
                    # Default to string comparison (case-sensitive, strips whitespace)
                    else:
                         duplicate_jes_df = je_detail_df[target_col.astype(str).str.strip() == str(val_to_find).strip()]

                    # --- Display results ---
                    if not duplicate_jes_df.empty:
                        st.write(f"Found {len(duplicate_jes_df)} matching entries across all JE data:")
                        # Format results for display
                        dup_df_display = duplicate_jes_df.copy()
                        dup_amount_cols = [c for c in dup_df_display.columns if 'Amount' in c or c == JE_AMOUNT_COLUMN]
                        dup_date_cols = [c for c in dup_df_display.columns if 'Date' in c or c == JE_DATE_COLUMN]
                        for c in dup_amount_cols: dup_df_display[c] = dup_df_display[c].apply(utils.format_amount_safely)
                        dup_col_config = {c: st.column_config.DateColumn(format="YYYY-MM-DD") for c in dup_date_cols}
                        # Display the results table
                        st.dataframe(dup_df_display, use_container_width=True, column_config=dup_col_config, hide_index=True)
                    else:
                        st.info(f"No other JEs found where '{col_to_check}' is '{val_to_find}' in the full dataset.")

                except KeyError: st.error(f"Duplicate Check Error: Column '{col_to_check}' not found in the main JE dataset.")
                except ValueError as ve: st.error(f"Duplicate Check Lookup Error: {ve}") # Catch specific conversion/comparison errors
                except Exception as e: st.error(f"Unexpected error during duplicate lookup: {e}"); st.exception(e)

            # Reset trigger *after* search attempt (whether successful or not)
            st.session_state.dup_search_triggered = False
            # Rerun ONLY if search was triggered to clear the "Finding..." message and results
            st.rerun()

    else: st.warning("No suitable columns identified for duplicate value checking in the JE data.")
else: st.info("Select an Account/Period with Journal Entries displayed to enable the duplicate value lookup feature.")