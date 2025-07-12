import streamlit as st
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import signal
import sys
from typing import AsyncGenerator
import nest_asyncio

# Apply nest_asyncio for easier async integration
nest_asyncio.apply()

# Import your project modules
from langchain_mcp_adapters.client import MultiServerMCPClient
from src.graph import build_agent_graph, AgentState  # Adjust path if needed
from src.mcp_servers.config import mcp_config  # Adjust path if needed
from langchain_core.messages import HumanMessage

# Import the stream_graph_response from client.py
from src.client import stream_graph_response  # Adjust path if needed; assumes Streamlit file is in src/

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Define the working directory for data
MCP_FILESYSTEM_DIR = os.environ.get("MCP_FILESYSTEM_DIR", "./data")
os.makedirs(MCP_FILESYSTEM_DIR, exist_ok=True)


# Function to run async generator in a separate thread
def run_async_gen_in_thread(async_gen: AsyncGenerator[str, None]) -> str:
    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = ""
        try:
            while True:
                try:
                    chunk = loop.run_until_complete(anext(async_gen))
                    response += chunk
                except StopAsyncIteration:
                    break
        finally:
            loop.close()
        return response

    with ThreadPoolExecutor() as executor:
        future = executor.submit(_run)
        return future.result()


# Streamlit app
st.title("Data Alchemist")

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mcp_process" not in st.session_state:
    st.session_state.mcp_process = None
if "graph" not in st.session_state:
    st.session_state.graph = None
if "graph_config" not in st.session_state:
    st.session_state.graph_config = {
        "configurable": {
            "thread_id": "1"
        }
    }

# Start MCP server if not running (using subprocess for background)
if st.session_state.mcp_process is None:
    mcp_server_path = "/Users/brets/EDU/DataAlchemist-MCP/src/mcp_servers/data_alchemist.py"  # Adjust to your actual path
    st.session_state.mcp_process = subprocess.Popen(
        ["python", mcp_server_path],
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    st.info("Started MCP server in background.")

# Initialize client and graph (do this once)
if st.session_state.graph is None:
    def get_tools_sync():
        async def get_tools_async():
            client = MultiServerMCPClient(connections=mcp_config)
            return await client.get_tools()

        return asyncio.run(get_tools_async())


    with ThreadPoolExecutor() as executor:
        future = executor.submit(get_tools_sync)
        tools = future.result()
    st.session_state.graph = build_agent_graph(tools=tools)

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    # Save the uploaded file to the working directory
    file_path = Path(MCP_FILESYSTEM_DIR) / uploaded_file.name
    if not file_path.exists():
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File saved to {file_path}")

        # Automatically load the data via the agent (optional initial message)
        initial_message = f"Load the data from {uploaded_file.name}"
        st.session_state.messages.append({"role": "user", "content": initial_message})

        # Run the agent
        graph_input = AgentState(messages=[HumanMessage(content=initial_message)])
        async_gen = stream_graph_response(
            graph_input=graph_input,
            graph=st.session_state.graph,
            config=st.session_state.graph_config
        )
        response = run_async_gen_in_thread(async_gen)
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.warning("File already exists. Please choose a different file or reset session.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about your data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Run the agent stream
        graph_input = AgentState(messages=[HumanMessage(content=prompt)])
        async_gen = stream_graph_response(
            graph_input=graph_input,
            graph=st.session_state.graph,
            config=st.session_state.graph_config
        )


        # Stream the response in thread
        def stream_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                while True:
                    try:
                        chunk = loop.run_until_complete(anext(async_gen))
                        yield chunk
                    except StopAsyncIteration:
                        break
            finally:
                loop.close()


        with ThreadPoolExecutor() as executor:
            future = executor.submit(stream_in_thread)
            gen = future.result()  # Wait for generator, but since it's yielding, better to handle differently

        # Adjusted streaming: since generator is async, run full in thread and update progressively if possible
        # For simplicity, collect full response in thread
        full_response = run_async_gen_in_thread(async_gen)
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})


# Cleanup on app close (optional, but good practice)
def cleanup():
    if st.session_state.mcp_process:
        os.kill(st.session_state.mcp_process.pid, signal.SIGTERM)
        st.session_state.mcp_process = None

# Note: Streamlit doesn't have a direct on_close, but you can use this in production setups