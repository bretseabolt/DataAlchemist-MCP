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

nest_asyncio.apply()

from langchain_mcp_adapters.client import MultiServerMCPClient
from src.graph import build_agent_graph, AgentState  # Adjust path if needed
from src.mcp_servers.config import mcp_config  # Adjust path if needed
from langchain_core.messages import HumanMessage

from src.client import stream_graph_response  # Adjust path if needed; assumes Streamlit file is in src/

from dotenv import load_dotenv

load_dotenv()

MCP_FILESYSTEM_DIR = os.environ.get("MCP_FILESYSTEM_DIR", "./data")
os.makedirs(MCP_FILESYSTEM_DIR, exist_ok=True)


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


st.title("Data Alchemist")

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

if st.session_state.mcp_process is None:
    mcp_server_path = "/Users/brets/EDU/DataAlchemist-MCP/src/mcp_servers/data_alchemist.py"  # Adjust to your actual path
    st.session_state.mcp_process = subprocess.Popen(
        ["python", mcp_server_path],
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    st.info("Started MCP server in background.")

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

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    file_path = Path(MCP_FILESYSTEM_DIR) / uploaded_file.name
    if not file_path.exists():
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File saved to {file_path}")

        initial_message = f"Load the data from {uploaded_file.name}"
        st.session_state.messages.append({"role": "user", "content": initial_message})

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

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        graph_input = AgentState(messages=[HumanMessage(content=prompt)])
        async_gen = stream_graph_response(
            graph_input=graph_input,
            graph=st.session_state.graph,
            config=st.session_state.graph_config
        )


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
            gen = future.result()


        full_response = run_async_gen_in_thread(async_gen)
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})


def cleanup():
    if st.session_state.mcp_process:
        os.kill(st.session_state.mcp_process.pid, signal.SIGTERM)
        st.session_state.mcp_process = None

