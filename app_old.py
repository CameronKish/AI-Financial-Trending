import streamlit as st
import pandas as pd
import numpy as np # For std deviation calculation

# --- Import data and functions from your processing script ---
# Option 1: Run the processing script directly (simpler for standalone app)
# This assumes the script is in the same directory and handles file paths correctly.
try:
    # Execute the script - this will load data, clean it, and define functions/variables
    exec(open("data_processor.py").read(), globals())
    # Check if necessary variables were loaded
    if 'pl_flat_df' not in globals() or 'je_detail_df' not in globals() or 'get_journal_entries' not in globals():
        st.error("Could not load necessary data or functions from data_processor.py. Please ensure it runs correctly and defines 'pl_flat_df', 'je_detail_df', and 'get_journal_entries'.")
        st.stop()
    # Make sure key column names are accessible (they should be global if defined outside functions in data_processor.py)
    if 'PL_ID_COLUMN' not in globals() or 'PL_MAP_COLUMN' not in globals():
         st.error("Column name variables (e.g., PL_ID_COLUMN) not found. Ensure they are defined globally in data_processor.py.")
         st.stop()

except FileNotFoundError:
    st.error("Error: `data_processor.py` not found in the current directory.")
    st.stop()
except Exception as e:
    st.error(f"Error running `data_processor.py`: {e}")
    st.stop()

# Option 2: Import specific objects (requires data_processor.py to be structured for import)
# from data_processor import pl_flat_df, je_detail_df, get_journal_entries, PL_ID_COLUMN, PL_MAP_COLUMN # etc.

# --- Streamlit App Layout ---
st.set_page_config(layout="wide") # Use wide layout for better table display
st.title("P&L Analyzer with Journal Entry Drilldown")

# --- Prepare P&L Wide View ---
st.header("Profit & Loss Overview (Monthly)")

try:
    # Convert 'Period' to datetime for proper sorting if it's not already
    # The melt function might have created datetime objects if the original column headers were parsable dates
    pl_flat_df['Period_dt'] = pd.to_datetime(pl_flat_df['Period'], errors='coerce')

    # Handle potential parsing errors
    if pl_flat_df['Period_dt'].isnull().any():
        st.warning("Could not parse all period headers into dates. Sorting might be incorrect. Using original Period string.")
        period_col_for_pivot = 'Period'
    else:
        # Create a sortable/display string if needed, e.g., 'YYYY-MM'
        pl_flat_df['Period_Str'] = pl_flat_df['Period_dt'].dt.strftime('%Y-%m')
        period_col_for_pivot = 'Period_Str'


    # Pivot the table
    pnl_wide_view_df = pl_flat_df.pivot_table(
        index=[PL_ID_COLUMN, PL_MAP_COLUMN], # Use both ID and Mapping for index
        columns=period_col_for_pivot,       # Use the sortable period string
        values='Amount'
    ).fillna(0) # Fill NaNs with 0 for calculations

    # Sort columns chronologically (important!)
    pnl_wide_view_df = pnl_wide_view_df.sort_index(axis=1)

except Exception as e:
    st.error(f"Error creating pivoted P&L view: {e}")
    st.stop()


# --- Outlier Highlighting Function ---
def highlight_outliers(row, threshold_std_dev=2.0):
    """
    Calculates MoM difference and highlights values where the change
    from the previous month exceeds N standard deviations for that row.
    """
    # Calculate Month-over-Month differences for the row
    diffs = row.diff()
    # Calculate mean and std dev of the *differences*, excluding NaN (first diff)
    mean_diff = diffs.dropna().mean()
    std_diff = diffs.dropna().std()

    # Define threshold for outlier detection
    # Avoid division by zero or issues if std_dev is very small/zero
    outlier_threshold_value = threshold_std_dev * std_diff if std_diff is not None and std_diff > 1e-6 else np.inf

    # Create style list for the row
    styles = []
    for i, diff in enumerate(diffs):
        style = ''
        # Check if the *absolute* difference is an outlier
        if pd.notna(diff) and abs(diff) > outlier_threshold_value:
            # Highlight the cell for the month *where the difference occurred*
            style = 'background-color: yellow; color: black;' # Add black color for readability
        styles.append(style)

    return styles

# --- Display Styled P&L Table ---
st.dataframe(
    pnl_wide_view_df.style.apply(highlight_outliers, axis=1).format("{:,.2f}"), # Apply styling and formatting
    use_container_width=True # Adjust table width to container
)
st.caption("Cells highlighted yellow indicate a month-over-month change greater than 2 standard deviations for that specific account.")


# --- User Selection for JE Drilldown ---
st.sidebar.header("Select Data for JE Lookup")

# Create list of account mappings for the selectbox
# The index is a MultiIndex ([PL_ID_COLUMN, PL_MAP_COLUMN])
account_options = pnl_wide_view_df.index.get_level_values(PL_MAP_COLUMN).unique().tolist()
# Create a mapping from display name back to Account ID
account_name_to_id = dict(pnl_wide_view_df.index.tolist()) # Maps (ID, Name) -> Row Index, need Name -> ID
account_name_to_id_map = {name: id_ for id_, name in pnl_wide_view_df.index}


selected_account_name = st.sidebar.selectbox(
    "Select GL Account:",
    options=account_options,
    index=0 # Default to the first account
)

# Get the corresponding Account ID
selected_account_id = account_name_to_id_map.get(selected_account_name)

# Get period options from the columns of the wide table
period_options = pnl_wide_view_df.columns.tolist()
selected_period = st.sidebar.selectbox(
    "Select Period:",
    options=period_options,
    index=len(period_options) - 1 # Default to the last period
)

# --- Fetch and Display Journal Entries ---
st.header("Journal Entry Details")

if selected_account_id and selected_period:
    st.write(f"Showing Journal Entries for: **{selected_account_name} ({selected_account_id})** | Period: **{selected_period}**")

    # Call the function defined in data_processor.py
    try:
        related_jes = get_journal_entries(selected_account_id, selected_period, je_detail_df)

        if not related_jes.empty:
            # Display the JE details, potentially formatting numbers
            st.dataframe(related_jes.style.format(subset=[col for col in related_jes.columns if 'Amount' in col or 'Debit' in col or 'Credit' in col], formatter="{:,.2f}"), use_container_width=True)
        else:
            st.info(f"No Journal Entries found for '{selected_account_name}' in the period '{selected_period}'.")

    except Exception as e:
        st.error(f"An error occurred while fetching Journal Entries: {e}")

else:
    st.info("Select a GL Account and Period from the sidebar to view related Journal Entries.")