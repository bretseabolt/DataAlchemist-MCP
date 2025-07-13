import streamlit as st
import polars as pl
import os
from dotenv import load_dotenv

load_dotenv()

MCP_FILESYSTEM_DIR = os.environ.get("MCP_FILESYSTEM_DIR", "./data")

st.title("Current DataFrame")

def display_dataframe(title: str, parquet_path: str, is_series: bool = False):
    if os.path.exists(parquet_path):
        try:
            df = pl.read_parquet(parquet_path)
            if is_series:
                df = df.to_series(0).to_frame("target")

            st.subheader(title)
            show_full = st.checkbox(f"Show full {title.lower()}", key=f"full_{title.replace(' ', '_')}")
            if not show_full:
                n_rows = st.number_input(f"Number of rows to show for {title.lower()}", min_value=1, value=10, key=f"rows_{title.replace(' ', '_')}")
                display_df = df.head(n_rows)
            else:
                display_df = df
            st.dataframe(display_df, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading data: {e}")
    else:
        st.info("No data loaded yet.")

data_parquet = os.path.join(MCP_FILESYSTEM_DIR, "data_session_data.parquet")
display_dataframe("Data", data_parquet)

x_train_parquet = os.path.join(MCP_FILESYSTEM_DIR, "data_session_x_train.parquet")
display_dataframe("Training Features (X_train)", x_train_parquet)

y_train_parquet = os.path.join(MCP_FILESYSTEM_DIR, "data_session_y_train.parquet")
display_dataframe("Training Target (y_train)", y_train_parquet, is_series=True)

x_test_parquet = os.path.join(MCP_FILESYSTEM_DIR, "data_session_x_test.parquet")
display_dataframe("Testing Features (X_test)", x_test_parquet)

y_test_parquet = os.path.join(MCP_FILESYSTEM_DIR, "data_session_y_test.parquet")
display_dataframe("Testing Target (y_test)", y_test_parquet, is_series=True)



