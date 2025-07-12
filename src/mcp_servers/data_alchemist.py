import pandas as pd
import numpy as np
from mcp.server.fastmcp import FastMCP
from typing import Optional, Dict, Union, List, Any
import os
import pickle
from dotenv import load_dotenv
import io


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

load_dotenv()

mcp = FastMCP("data_alchemist")


class DataAlchemist:

    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.encoders: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}

        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.DataFrame] = None

        self.model: Optional[Any] = None

        self.working_dir = os.environ.get("MCP_FILESYSTEM_DIR")
        if self.working_dir:
            os.makedirs(self.working_dir, exist_ok=True) # Ensure directory exists
        self.session_file = os.path.join(self.working_dir, "data_session.pkl") if self.working_dir else None

    def _save_data(self) -> None:
        if self.data is not None and self.session_file:
            try:
                with open(self.session_file, "wb") as f:
                    pickle.dump({'data': self.data, 'encoders': self.encoders, 'scalers': self.scalers,
                                 'X_train': self.X_train, 'X_test': self.X_test,
                                 'y_train': self.y_train, 'y_test': self.y_test,'model': self.model}, f)
            except Exception as e:
                print(f"Warning: Failed to save session data: {e}")

    def _load_data_from_session(self) -> bool:
        if self.session_file and os.path.exists(self.session_file):
            try:
                with open(self.session_file, "rb") as f:
                    state = pickle.load(f)
                    self.data = state.get('data')
                    self.encoders = state.get('encoders', {})
                    self.scalers = state.get('scalers', {})
                    self.X_train = state.get('X_train')
                    self.X_test = state.get('X_test')
                    self.y_train = state.get('y_train')
                    self.y_test = state.get('y_test')
                    self.model = state.get('model')
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
        self.encoders = {}
        self.scalers = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
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
        if os.path.exists(full_path):
            return f"Error: File already exists at {full_path}. Choose a different name to avoid overwriting."

        try:
            # Save the dataframe to a .csv file, excluding the pandas index
            self.data.to_csv(full_path, index=False, encoding='utf-8')
            return f"Dataframe successfully saved to {full_path}. The task is complete. Please ask the user for the next action."
        except Exception as e:
            return f"Error: Failed to save data to {full_path}: {e}"


    async def inspect_data(self, n_rows: int=5) -> str:
        """
        Initial data inspection. Includes first n rows of data, DataFrame info,
        data types of each column, missing values per column, duplicate rows,
        number of unique values per column.
        """
        if self.data is None:
            if not self._load_data_from_session():
                return "Error: No data loaded. Please load data first using 'load_data'."

        df_head = self.data.head(n_rows).to_dict()

        buffer = io.StringIO()
        self.data.info(buf=buffer)
        info_str = buffer.getvalue()

        dtypes_str = self.data.dtypes.to_string()

        null_values = self.data.isnull().sum().to_dict()

        duplicate_rows = self.data.duplicated().sum()

        unique_columns = self.data.nunique().to_dict()

        analysis_report = f"""
        DataFrame Analysis Report
        --- FIRST N ROWS ---
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

    async def encode_categorical_features(self, encode_map: Dict[str, str], ordinal_map: Optional[Dict[str, List[str]]] = None) -> str:
        """
        Uses encoders to encode categorical columns (One-hot or ordinal)

        Args:
            encode_map: A dictionary that maps column names with encoding strategy.
            ordinal_map (Optional): For ordinal encoding only. A dictionary mapping the column
            name to a list containing the desired category order. If none provided, order is inferred
            alphabetically.
        """
        if self.data is None:
            if not self._load_data_from_session():
                return "Error: No data loaded. Please load data first using 'load_data'."

        if not encode_map:
            return "No encoder map provided; no changes made."

        allowed_strategies = ["one_hot", "ordinal"]
        successful_encodes = []
        errors = []

        for column_name, encode_strat in encode_map.items():

            if column_name not in self.data.columns:
                errors.append(f"Error: Column '{column_name}' not found in the dataset.")
                continue

            if encode_strat not in allowed_strategies:
                errors.append(
                    f"Invalid strategy '{encode_strat}' for column '{column_name}'. Valid strategies are {allowed_strategies}")
                continue

            try:
                if encode_strat == "one_hot":

                    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                    encoded_data = encoder.fit_transform(self.data[[column_name]])
                    encoded_df = pd.DataFrame(data=encoded_data, columns=encoder.get_feature_names_out([column_name]))

                    self.data = self.data.drop(column_name, axis=1)
                    self.data = pd.concat([self.data.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

                    self.encoders[column_name] = encoder

                    successful_encodes.append(f"Successfully encoded column '{column_name}'.")

                elif encode_strat == "ordinal":
                    custom_order = ordinal_map.get(column_name)

                    if custom_order:
                        encoder = OrdinalEncoder(
                            categories=[custom_order],
                            handle_unknown="use_encoded_value",
                            unknown_value=-1
                        )
                    else:
                        encoder = OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1
                        )


                    self.data[[column_name]] = encoder.fit_transform(self.data[[column_name]])
                    self.encoders[column_name] = encoder

                    successful_encodes.append(f"Successfully encoded column '{column_name}'.")


            except Exception as e:
                errors.append(f"Failed to encode column '{column_name}' with strategy '{encode_strat}': {e}.")

        self._save_data()

        if len(successful_encodes) == len(encode_map) and len(errors) == 0:
            encode_report = f"""
            Successfully performed encoding with no errors in: {list(encode_map.keys())}.

            --- SUCCESSFUL ENCODES ---
            {'\n'.join(successful_encodes)}

            """
        else:
            encode_report = f"""
            Unsuccessfully performed encoding in columns: {list(encode_map.keys())}.

            --- UNSUCCESSFUL ENCODES ---
            {'\n'.join(errors)}

            --- SUCCESSFUL ENCODES ---
            {'\n'.join(successful_encodes)}

            """

        return encode_report.strip()

    async def drop_columns(self, column: Union[str, List[str]]) -> str:
        """
        Drops columns in a DataFrame.

        Args:
            column: Column name or list of column names to drop.
        """
        if self.data is None:
            if not self._load_data_from_session():
                return "Error: No data loaded. Please load data first using 'load_data'."
        try:
            self.data = self.data.drop(columns=column, axis=1)
            self._save_data()
            return f"Successfully dropped column(s) '{column}'."
        except Exception as e:
            return f"Error dropping column(s) '{column}': {str(e)}"


    async def split_data(self, target_column: str, test_size: float=0.2, random_state: int = 42) -> str:
        """
        Splits the data into training and testing sets.

        Args:
            target_column: The name of the column to be used as the target (y).
            test_size: The proportion of the dataset to include in the test split.
            random_state: Seed for the random number generator for reproducibility.
        """
        if self.data is None:
            if not self._load_data_from_session():
                return "Error: No data loaded. Please load data first."
        if target_column not in self.data.columns:
            return f"Error: target column {target_column} is not present in the dataset."

        try:
            X = self.data.drop(target_column, axis=1)
            y = self.data[target_column]

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            self._save_data()

            return (f"Data successfully split into training and testing sets.\n"
                    f"Training set shape: {self.X_train.shape}\n"
                    f"Testing set shape: {self.X_test.shape}")
        except Exception as e:
            return f"Error occurred during data splitting: {e}"

    async def train_linear_regression(self) -> str:
        """
        Trains a linear regression model and evaluates its performance on the test data.
        """
        if self.X_train is None or self.y_train is None:
            # Try to reload the session state as a fallback
            if not self._load_data_from_session() or self.X_train is None or self.y_train is None:
                return "Error: Data has not been split. Please split the data first."

        try:
            self.model = LinearRegression()
            self.model.fit(self.X_train, self.y_train)

            y_pred = self.model.predict(self.X_test)

            r2 = r2_score(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = mse ** 0.5

            lr_model_report = f"""
            Linear Regression model trained successfully.

            --- MODEL PERFORMANCE ON TEST SET ---
            R-Squared: {round(r2, 4)}
            Mean Absolute Error (MAE): {round(mae, 4)}
            Mean Squared Error (MSE): {round(mse, 4)}
            Root Mean Squared Error (RMSE): {round(rmse, 4)}
            """

            return lr_model_report.strip()

        except Exception as e:
            return f"Error occurred during model training: {e}"

session = DataAlchemist()

@mcp.tool()
async def alchemy_load_data(file_path: str) -> str:
    """
    Load data from a file into the session for data preprocessing.

    Args:
        file_path: Path of the file to load
    """
    return await session.load_data(file_path)

@mcp.tool()
async def alchemy_reset_session() -> str:
    """
    Reset the session for preprocessing by deleting the persisted data and clearing the current state.
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
async def alchemy_inspect_data(n_rows: int) -> str:
    """
    Performs initial data inspection on the loaded DataFrame to understand the data.
    Includes n rows of the data, DataFrame info, data types of columns, descriptive statistics,
    missing value counts, and duplicate rows.

    Args:
        n_rows: the amount of rows to show (.head())

    Returns:
        Report in the form of a formatted string that summarizes the inspection.
    """
    return await session.inspect_data(n_rows)

@mcp.tool()
async def alchemy_impute_missing_values(imputation_map: Dict[str, str]) -> str:
    """
    Performs imputation on missing values in a column by using a descriptive statistic.
    Takes in a dictionary that maps column names with imputation strategy.
    The column MUST be present in the data frame and the imputation strategy MUST be mean, median, or mode.

    Args:
        imputation_map: Dictionary mapping column names with imputation strategy.

    Returns:
        Report in the form of a formatted string that summarizes the imputation results.
    """

    return await session.impute_missing_values(imputation_map)

@mcp.tool()
async def alchemy_encode_categorical_features(encode_map: Dict[str, str], ordinal_map: Optional[Dict[str, List[str]]]) -> str:
    """
    Encodes categorical features using 'one-hot' or 'ordinal' encoding.

    Args:
        encode_map: Dictionary mapping column names with the encoding strategy.
                    Ex: {"education_level": "ordinal", "city": "one-hot"}
        ordinal_map (Optional): **REQUIRED if using 'ordinal' strategy with a specific order.**
                                Dictionary mapping the column name with a list of categories in the
                                correctly implied order, from lowest to highest. If a list is not provided,
                                order will be inferred alphabetically
                                Ex: {"education_level": ["High School", "Bachelors", "Masters", PhD]}

    """
    return await session.encode_categorical_features(encode_map, ordinal_map)

@mcp.tool()
async def alchemy_drop_columns(column: Union[str, List[str]]) -> str:
    """
    Drops/removes columns in a DataFrame.

    Args:
        column: Column name or list of column names to drop.
    """
    return await session.drop_columns(column)

@mcp.tool()
async def alchemy_split_data(target_column: str, test_size: float=0.2, random_state: int = 42) -> str:
    """
    Splits the data into training and testing sets.

    Args:
        target_column: The name of the column to be used as the target (y).
        test_size: The proportion of the dataset to include in the test split.
        random_state: Seed for the random number generator for reproducibility.
    """
    return await session.split_data(target_column, test_size, random_state)

@mcp.tool()
async def alchemy_train_linear_regression() -> str:
    """
    Trains a linear regression model and evaluates its performance on the test data.
    """
    return await session.train_linear_regression()


if __name__ == "__main__":
    mcp.run(transport='stdio')