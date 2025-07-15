import streamlit as st
import os
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
MCP_FILESYSTEM_DIR = os.environ.get("MCP_FILESYSTEM_DIR", "./data")

st.title("Visualizations")

st.info("Generated plots from the Data Alchemist agent appear here. Use the chat to request new ones.")

for file in os.listdir(MCP_FILESYSTEM_DIR):
    if file.endswith(".png"):
        plot_path = os.path.join(MCP_FILESYSTEM_DIR, file)
        st.subheader(file.replace(".png", "").replace("_", " ").title())
        image = Image.open(plot_path)
        st.image(image, use_column_width=True)

        with open(plot_path, "rb") as f:
            st.download_button("Download PNG", f, file_name=file)