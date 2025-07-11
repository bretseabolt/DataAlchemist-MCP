import pandas as pd
from mcp.server.fastmcp import FastMCP
from typing import Optional
import os
import pickle
from dotenv import load_dotenv
import io

load_dotenv()

mcp = FastMCP("data_alchemy")


class DataAlchemy:

    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.working_dir = os.environ.get("MCP_FILESYSTEM_DIR")
        self.session_file = os.path.join(self.working_dir, "session_data.pkl") if self.working_dir else None

    def _save_data(self) -> None:
        if self.data is not None and self.session_file:
            with open(self.session_file, "wb") as f:
                pickle.dump(self.data, f)

    def _load_data_from_session(self) -> bool:
        if self.session_file and os.path.exists(self.session_file):
            try:
                with open(self.session_file, "rb") as f:
                    self.data = pickle.load(f)
                return True
            except Exception:
                return False
        return False

    async def load_data(self, file_path: str) -> str:
        if not self.working_dir:
            return "ERROR: MCP_FILESYSTEM_DIR environment variable not set."

        full_path = os.path.join(self.working_dir, file_path)


        try:
            self.data = pd.read_csv(full_path)
            self._save_data()
            return f"Data loaded from {full_path}"
        except Exception as e:
            return f"Error loading data from {full_path}: {e}"

    async def reset_session(self) -> str:
        """
        Reset the current session by deleting the presisted data file and clearing in-memory data
        """
        if self.session_file and os.path.exists(self.session_file):
            try:
                os.remove(self.session_file)
            except Exception as e:
                return f"Error resetting session: {e}"
        self.data = None
        return "Session reset successfully. You can now load new data."

session = DataAlchemy()

@mcp.tool()
async def alchemy_load_data(file_path: str) -> str:
    """
    Load data from a file into the session

    Args:
        file_path: Path of the file to load
    """
    return await session.load_data(file_path)

@mcp.tool()
async def alchemy_data_inspection() -> str:
    """
    Performs initial data inspection on the loaded DataFrame to understand the data.
    Includes DataFrame info, data types of columns, descriptive statistics,
    missing value counts, and duplicate rows.
    """
    if session.data is None:
        if not session._load_data_from_session():
            return "Error: No data loaded. Please load data first using 'alchemy_load_data'."

    buffer = io.StringIO()
    session.data.info(buf=buffer)
    info_str = buffer.getvalue()

    dtypes_str = session.data.dtypes.to_string()

    null_values = session.data.isnull().sum().to_dict()

    duplicate_rows = session.data.duplicated().sum()

    analysis_report = f"""
    ✅ DataFrame Analysis Report

    --- INFO ---
    {info_str}

    --- DATA TYPES ---
    {dtypes_str}

    --- MISSING VALUES ---
    {null_values}

    --- DUPLICATE ROWS ---
    Found {duplicate_rows} duplicate rows.

    """
    return analysis_report.strip()

@mcp.tool()
async def alchemy_reset_session() -> str:
    """


    :return:
    """
if __name__ == "__main__":
    mcp.run(transport='stdio')