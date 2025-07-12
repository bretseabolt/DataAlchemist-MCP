import pandas as pd
import numpy as np
from mcp.server.fastmcp import FastMCP
from typing import Optional, Dict
import os
import pickle
from dotenv import load_dotenv
import io

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler

load_dotenv()

mcp = FastMCP("data_inspection_and_cleaning")


class DataAlchemist:

    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.working_dir = os.environ.get("MCP_FILESYSTEM_DIR")
        if self.working_dir:
            os.makedirs(self.working_dir, exist_ok=True) # Ensure directory exists
        self.session_file = os.path.join(self.working_dir, "session_data.pkl") if self.working_dir else None

    def _save_data(self) -> None:
        if self.data is not None and self.session_file:
            try:
                with open(self.session_file, "wb") as f:
                    pickle.dump(self.data, f)
            except Exception as e:
                print(f"Warning: Failed to save session data: {e}")

    def _load_data_from_session(self) -> bool:
        if self.session_file and os.path.exists(self.session_file):
            try:
                with open(self.session_file, "rb") as f:
                    self.data = pickle.load(f)
                return True
            except Exception as e:
                print(f"Warning: Failed to load session data: {e}")
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

    async def save_to_csv(self, file_path: str) -> str:
        """
        Saves the current state of the DataFrame to a .csv file in the working directory.

        Args:
            file_path: The name for the output .csv file.
        """
        if self.data is None:
            # Try to load from session as a fallback
            if not self._load_data_from_session():
                return "Error: No data in memory to save. Please load data first."

        if not self.working_dir:
            return "ERROR: MCP_FILESYSTEM_DIR environment variable not set."

        # Ensure the filename ends with .csv
        if not file_path.lower().endswith('.csv'):
            file_path += '.csv'

        full_path = os.path.join(self.working_dir, file_path)

        try:
            # Save the dataframe to a .csv file, excluding the pandas index
            self.data.to_csv(full_path, index=False)
            return f"Dataframe successfully saved to {full_path}"
        except Exception as e:
            return f"Error: Failed to save data to {full_path}: {e}"

    async def inspect_data(self) -> str:
        """
        Initial data inspection
        """
        if self.data is None:
            if not self._load_data_from_session():
                return "Error: No data loaded. Please load data first using 'load_data'."

        df_head = self.data.head().to_dict()

        buffer = io.StringIO()
        session.data.info(buf=buffer)
        info_str = buffer.getvalue()

        dtypes_str = session.data.dtypes.to_string()

        null_values = session.data.isnull().sum().to_dict()

        duplicate_rows = session.data.duplicated().sum()

        unique_columns = session.data.nunique().to_dict()

        analysis_report = f"""
        DataFrame Analysis Report
        --- FIRST 5 ROWS ---
        {df_head}
        
        --- INFO ---
        {info_str}

        --- DATA TYPES ---
        {dtypes_str}

        --- MISSING VALUES ---
        {null_values}

        --- DUPLICATE ROWS ---
        Found {duplicate_rows} duplicate rows.
        
        --- UNIQUE VALUES OF EVERY COLUMN ---
        {unique_columns}
        
        """
        return analysis_report.strip()

    async def impute_missing_values(self, imputation_map: Dict[str, str]) -> str:
        """
        Replaces missing values in a column by using a descriptive statistic (mean, median, or mode).

        Args:
            imputation_map: A dictionary that maps column names with imputation strategy.
        """
        if self.data is None:
            if not self._load_data_from_session():
                return "Error: No data loaded. Please load data first using 'load_data'."

        if not imputation_map:
            return "No imputation map provided; no changes made."


        allowed_strategies = ["mean", "median", "mode"]
        successful_imputations = []
        errors = []

        for column_name, imputation_strat in imputation_map.items():

            if column_name not in self.data.columns:
                errors.append(f"Error: Column '{column_name}' not found in the dataset.")
                continue

            if imputation_strat not in allowed_strategies:
                errors.append(f"Invalid strategy '{imputation_strat}' for column '{column_name}'. Valid strategies are {allowed_strategies}")
                continue

            try:
                impute_strat = "most_frequent" if imputation_strat == "mode" else imputation_strat

                imputer = SimpleImputer(missing_values=np.nan, strategy=impute_strat)
                self.data[[column_name]] = imputer.fit_transform(self.data[[column_name]])

                successful_imputations.append(f"Successfully imputed column '{column_name}' with strategy '{imputation_strat}'.")

            except Exception as e:
                errors.append(f"Failed to impute column '{column_name}' with strategy '{imputation_strat}': {e}.")


        self._save_data()

        if len(successful_imputations) == len(imputation_map) and len(errors) == 0:
            imputation_report = f"""
            Successfully performed imputation for missing values in columns with no errors: {list(imputation_map.keys())}.
            
            --- SUCCESSFUL IMPUTATIONS ---
            {'\n'.join(successful_imputations)}
            
            """
        else:
            imputation_report = f"""
            Unsuccessfully performed imputation for missing values in columns: {list(imputation_map.keys())}.
            
            --- UNSUCCESSFUL IMPUTATIONS ---
            {'\n'.join(errors)}
            
            --- SUCCESSFUL IMPUTATIONS ---
            {'\n'.join(successful_imputations)}

            """

        return imputation_report.strip()

session = DataAlchemist()

@mcp.tool()
async def alchemy_load_data(file_path: str) -> str:
    """
    Load data from a file into the session

    Args:
        file_path: Path of the file to load
    """
    return await session.load_data(file_path)

@mcp.tool()
async def alchemy_reset_session() -> str:
    """
    Reset the session by deleting the persisted data and clearing the current state.
    Use this to start fresh without previous modifications.
    """
    return await session.reset_session()

@mcp.tool()
async def alchemy_save_to_csv(file_path: str) -> str:
    """
    Saves the current state of the data (after cleaning/transformations) to a new .csv file.

    Args:
        file_path: The desired name for the output file (e.g., 'processed_data.csv').
    """
    return await session.save_to_csv(file_path)

@mcp.tool()
async def alchemy_inspect_data() -> str:
    """
    Performs initial data inspection on the loaded DataFrame to understand the data.
    Includes first five rows of the data, DataFrame info, data types of columns, descriptive statistics,
    missing value counts, and duplicate rows.
    """
    return await session.inspect_data()

@mcp.tool()
async def alchemy_impute_missing_values(imputation_map: Dict[str, str]) -> str:
    """
    Performs imputation on missing values in a column by using a descriptive statistic.
    Takes in a dictionary that maps column names with imputation strategy.
    The column MUST be present in the data frame and the imputation strategy MUST be mean, median, or mode.

    Args:
        imputation_map: Dictionary mapping column names with imputation strategy.

    """

    return await session.impute_missing_values(imputation_map)


if __name__ == "__main__":
    mcp.run(transport='stdio')