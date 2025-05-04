# streamlit_app.py
import streamlit as st
import pandas as pd
from datetime import datetime
import time # For simulating load time if needed

# Import functions/constants from our modules
import data_processor
import utils

# --- Page Config ---
st.set_page_config(layout="wide", page_title="P&L Analyzer")

# --- Styling & Constants ---
# (Using constants from utils)

# --- Data Loading Function with Caching ---
@st.cache_data(ttl=3600) # Cache data for 1 hour
def load_data_cached():
    """Loads and processes data using data_processor and caches the result."""
    try:
        start_time = time.time()
        logging.info("Attempting to load and process data...")
        pl_flat, je_detail = data_processor.load_and_process_data()
        end_time = time.time()
        logging.info(f"Data loaded and processed in {end_time - start_time:.2f} seconds.")
        if pl_flat is None or je_detail is None:
             st.error("Critical error during data loading. Check logs.")
             return None, None, None # Return None if loading failed
        # Make Account Name available easily if it exists
        je_account_name_col = 'Account Name' if 'Account Name' in je_detail.columns else None
        return pl_flat, je_detail, je_account_name_col
    except Exception as e:
        st.error(f"Failed to load or process data: {e}")
        logging.error(f"Exception during data loading: {e}", exc_info=True)
        return None, None, None

# --- Load Data and Store in Session State ---
# We store the dataframes in session state so pages can access them without reloading
if 'data_loaded' not in st.session_state:
    pl_flat_df, je_detail_df, je_account_name_col = load_data_cached()
    if pl_flat_df is not None and je_detail_df is not None:
        st.session_state.pl_flat_df = pl_flat_df
        st.session_state.je_detail_df = je_detail_df
        st.session_state.data_loaded = True
        st.session_state.column_config = { # Store column names in session state
            "PL_ID": data_processor.PL_ID_COLUMN,
            "PL_MAP1": data_processor.PL_MAP1_COLUMN,
            "PL_MAP2": data_processor.PL_MAP2_COLUMN,
            "PL_MAP_DISPLAY": data_processor.PL_MAP_COLUMN, # Lowest level mapping
            "JE_ID": data_processor.JE_ID_COLUMN,
            "JE_DATE": data_processor.JE_DATE_COLUMN,
            "JE_AMOUNT": data_processor.JE_AMOUNT_COLUMN,
            "JE_ACCOUNT_NAME": je_account_name_col,
            "JE_DETAILS_BASE": data_processor.JE_DETAIL_COLUMNS_BASE # Base list
        }
        st.success("Data loaded successfully!")
        # Give a moment for the success message to show before potential page switch
        # time.sleep(1) # Optional: uncomment if needed
    else:
        st.session_state.data_loaded = False
        st.error("Data loading failed. Cannot proceed.")
        st.stop() # Stop execution if data didn't load

# --- Initialize Session State (Only if data loaded successfully) ---
# Ensure this runs *after* data might be loaded
if st.session_state.data_loaded:
    # Initialize selections only if not already set
    if 'selected_account_id' not in st.session_state: st.session_state.selected_account_id = None
    if 'selected_account_name' not in st.session_state: st.session_state.selected_account_name = None
    if 'selected_period' not in st.session_state: st.session_state.selected_period = None
    if 'prev_selected_account_id' not in st.session_state: st.session_state.prev_selected_account_id = None
    if 'prev_selected_period' not in st.session_state: st.session_state.prev_selected_period = None
    # Use an empty DataFrame as default for related_jes_df to avoid errors
    if 'related_jes_df' not in st.session_state: st.session_state.related_jes_df = pd.DataFrame(columns=st.session_state.column_config["JE_DETAILS_BASE"])
    # Duplicate finder state
    if 'dup_col' not in st.session_state: st.session_state.dup_col = None
    if 'dup_val' not in st.session_state: st.session_state.dup_val = None
    if 'dup_search_triggered' not in st.session_state: st.session_state.dup_search_triggered = False
    # Chart selection state (initialize based on loaded data)
    if 'chart_accounts_selection' not in st.session_state:
        if 'pl_flat_df' in st.session_state:
             temp_account_options = sorted(st.session_state.pl_flat_df[st.session_state.column_config["PL_MAP_DISPLAY"]].unique().tolist())
             default_chart_selection = []
             potential_defaults = ["Total Net Sales", "Total COGS/COS", "Total Operating Expenses"] # Customize as needed
             for acc in potential_defaults:
                 if acc in temp_account_options: default_chart_selection.append(acc)
             # Fallback if specific defaults aren't found
             if not default_chart_selection and temp_account_options:
                 default_chart_selection = temp_account_options[:min(3, len(temp_account_options))]
             st.session_state.chart_accounts_selection = default_chart_selection
        else:
             st.session_state.chart_accounts_selection = [] # Empty list if data not ready


# --- Global Sidebar Elements ---
st.sidebar.header("App Navigation")
# Timestamp/Location can stay here to be global
current_time = datetime.now()
# Consider using a timezone-aware library if accuracy across zones is critical
st.sidebar.write(f"Timestamp: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
st.sidebar.write(f"Location Context: Denver, CO, USA") # Placeholder

# --- Main Page Content ---
st.title("Welcome to the P&L Analyzer")
st.markdown(f"""
<div style="background-color:{utils.EY_DARK_BLUE_GREY}; padding: 15px; border-radius: 5px; color: white;">
Use the navigation sidebar on the left ( < ) to switch between the analysis pages:
<ul>
    <li><b>📊 P&L Analysis & Drilldown:</b> View the P&L, identify outliers, and drill down into Journal Entries.</li>
    <li><b>📈 Visualizations:</b> Explore trends and patterns in your P&L and JE data.</li>
</ul>
Data is loaded from <code>{data_processor.EXCEL_FILE_PATH}</code>.
</div>
""", unsafe_allow_html=True)

st.info("Select a page from the sidebar to begin analysis.")

# Add any other introductory text or summaries here if desired.