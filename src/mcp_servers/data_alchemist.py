import polars as pl
import numpy as np
from mcp.server.fastmcp import FastMCP
from typing import Optional, Dict, Union, List, Any
import os

import joblib
from dotenv import load_dotenv



from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

load_dotenv()

mcp = FastMCP("data_alchemist")


class DataAlchemist:

    def __init__(self):
        self.data: Optional[pl.DataFrame] = None
        self.encoders: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}

        self.X_train: Optional[pl.DataFrame] = None
        self.X_test: Optional[pl.DataFrame] = None
        self.y_train: Optional[pl.Series] = None
        self.y_test: Optional[pl.Series] = None

        self.model: Optional[Any] = None

        self.working_dir = os.environ.get("MCP_FILESYSTEM_DIR")
        if self.working_dir:
            os.makedirs(self.working_dir, exist_ok=True) # Ensure directory exists
        self.session_base = os.path.join(self.working_dir, "data_session") if self.working_dir else None

    def _save_data(self) -> None:
        if self.session_base and self.data is not None:
            try:
                data_parquet = self.session_base + "_data.parquet"
                self.data.write_parquet(data_parquet)
                if self.X_train is not None:
                    x_train_parquet = self.session_base + "_x_train.parquet"
                    self.X_train.write_parquet(x_train_parquet)
                    x_test_parquet = self.session_base + "_x_test.parquet"
                    self.X_test.write_parquet(x_test_parquet)
                    y_train_parquet = self.session_base + "_y_train.parquet"
                    pl.DataFrame(self.y_train).write_parquet(y_train_parquet)
                    y_test_parquet = self.session_base + "_y_test.parquet"
                    pl.DataFrame(self.y_test).write_parquet(y_test_parquet)

                # Using Joblib for non-DataFrame State
                session_joblib = self.session_base + ".joblib"
                state = {'encoders': self.encoders, 'scalers': self.scalers, 'model': self.model}
                joblib.dump(state, session_joblib)

            except Exception as e:
                print(f"Warning: Failed to save: {e}")

    def _load_data_from_session(self) -> bool:
        if self.session_base:
            session_joblib = self.session_base + ".joblib"
            if os.path.exists(session_joblib):
                try:
                    data_parquet = self.session_base + "_data.parquet"
                    self.data = pl.read_parquet(data_parquet) if os.path.exists(data_parquet) else None
                    x_train_parquet = self.session_base + "_x_train.parquet"
                    self.X_train = pl.read_parquet(x_train_parquet) if os.path.exists(x_train_parquet) else None
                    x_test_parquet = self.session_base + "_x_test.parquet"
                    self.X_test = pl.read_parquet(x_test_parquet) if os.path.exists(x_test_parquet) else None
                    y_train_parquet = self.session_base + "_y_train.parquet"
                    self.y_train = pl.read_parquet(y_train_parquet).to_series(0) if os.path.exists(y_train_parquet) else None
                    y_test_parquet = self.session_base + "_y_test.parquet"
                    self.y_test = pl.read_parquet(y_test_parquet).to_series(0) if os.path.exists(y_test_parquet) else None
                    state = joblib.load(session_joblib)
                    self.encoders = state.get('encoders', {})
                    self.scalers = state.get('scalers', {})
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
            self.data = pl.read_csv(full_path, infer_schema_length=10000)
            self._save_data()
            return f"Data loaded from {full_path}"
        except Exception as e:
            return f"Error loading data from {full_path}: {e}"

    async def reset_session(self) -> str:
        """
        Reset the current session by deleting the presisted data file and clearing in-memory data
        """
        if self.session_base:
            session_joblib = self.session_base + ".joblib"
            if os.path.exists(session_joblib):
                try:
                    os.remove(session_joblib)
                    # Remove Parquet files
                    data_parquet = self.session_base + "_data.parquet"
                    if os.path.exists(data_parquet):
                        os.remove(data_parquet)
                    x_train_parquet = self.session_base + "_x_train.parquet"
                    if os.path.exists(x_train_parquet):
                        os.remove(x_train_parquet)
                    x_test_parquet = self.session_base + "_x_test.parquet"
                    if os.path.exists(x_test_parquet):
                        os.remove(x_test_parquet)
                    y_train_parquet = self.session_base + "_y_train.parquet"
                    if os.path.exists(y_train_parquet):
                        os.remove(y_train_parquet)
                    y_test_parquet = self.session_base + "_y_test.parquet"
                    if os.path.exists(y_test_parquet):
                        os.remove(y_test_parquet)
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
            self.data.write_csv(full_path)
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

        df_head = self.data.head(n_rows).to_dicts()

        describe_str = str(self.data.describe())

        schema_str = str(self.data.schema)

        null_values = self.data.null_count().to_dicts()[0]

        duplicate_rows = self.data.is_duplicated().sum()

        unique_columns = {col: self.data.select(pl.col(col).n_unique()).item() for col in self.data.columns}

        analysis_report = f"""
        DataFrame Analysis Report
        --- FIRST N ROWS ---
        {df_head}
        
        --- DESCRIBE ---
        {describe_str}

        --- SCHEMA (DATA TYPES) ---
        {schema_str}

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
                if imputation_strat in ["mean", "median", "mode"]:

                    if imputation_strat == "mean":
                        fill_value = self.data.select(pl.col(column_name).mean()).item()
                    elif imputation_strat == "median":
                        fill_value = self.data.select(pl.col(column_name).median()).item()
                    else: # mode
                        fill_value = self.data.select(pl.col(column_name).mode().first()).item()
                    self.data = self.data.with_columns(pl.col(column_name).fill_null(fill_value))

                    successful_imputations.append(
                        f"Successfully imputed column '{column_name}' with strategy '{imputation_strat}'.")

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

        allowed_strategies = ["one_hot", "one-hot", "ordinal"]
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
                if encode_strat == "one_hot" or encode_strat == "one-hot":
                    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                    encoder.set_output(transform="polars")  # Enable Polars output
                    encoded_data = encoder.fit_transform(self.data.select(pl.col(column_name)))  # Direct Polars input
                    self.data = self.data.drop(column_name).hstack(encoded_data)
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
                    encoder.set_output(transform="polars")
                    encoded_col = encoder.fit_transform(self.data.select(pl.col(column_name)))
                    self.data = self.data.with_columns(encoded_col)
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

    async def scale_numeric(self, scale_map: Dict[str, str]) -> str:
        """
        Scales numerical features independently on train and test sets.
        Fits scalers on training data only, then transforms both train and test.

        Args:
            scale_map: A dictionary that maps column names with scaling strategy.
        """
        if self.X_train is None or self.X_test is None:
            if not self._load_data_from_session() or self.X_train is None:
                return "Error: Data has not been split. Please splt the data first using 'split_data.'."

        if not scale_map:
            return "No scaling map provided; no changes made."

        allowed_strategies = ["standard", "minmax", "robust"]
        successful_scales = []
        errors = []

        for column_name, scale_strat in scale_map.items():

            if column_name not in self.X_train.columns:
                errors.append(f"Error: Column '{column_name}' not found in the dataset.")
                continue

            if scale_strat not in allowed_strategies:
                errors.append(
                    f"Invalid strategy '{scale_strat}' for column '{column_name}'. Valid strategies are {allowed_strategies}")
                continue

            try:
                if scale_strat == "standard":
                    scaler = StandardScaler()
                elif scale_strat == "minmax":
                    scaler = MinMaxScaler()
                else: # robust
                    scaler = RobustScaler()

                scaler.set_output(transform="polars")

                train_col_df = self.X_train.select(pl.col(column_name))
                scaler.fit(train_col_df)

                scaled_train = scaler.transform(train_col_df)
                self.X_train = self.X_train.with_columns(scaled_train)

                test_col_df = self.X_test.select(pl.col(column_name))
                scaled_test = scaler.transform(test_col_df)
                self.X_test = self.X_test.with_columns(scaled_test)

                self.scalers[column_name] = scaler
                successful_scales.append(f"Successfully scaled column '{column_name}'.")

            except Exception as e:
                errors.append(f"Failed to scale column '{column_name}' with strategy '{scale_strat}': {e}.")

        self._save_data()

        if len(successful_scales) == len(scale_map) and len(errors) == 0:
            scale_report = f"""
            Successfully performed scaling with no errors in: {list(scale_map.keys())}.

            --- SUCCESSFUL SCALES ---
            {'\n'.join(successful_scales)}

            """
        else:
            scale_report = f"""
            Unsuccessfully performed scaling in columns: {list(scale_map.keys())}.

            --- UNSUCCESSFUL SCALES ---
            {'\n'.join(errors)}

            --- SUCCESSFUL SCALES ---
            {'\n'.join(successful_scales)}

            """

        return scale_report.strip()

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
            self.data = self.data.drop(column if isinstance(column, list) else [column])
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
            # Define features (X) and target (y) directly as Polars objects
            X = self.data.drop(target_column)
            y = self.data.select(pl.col(target_column)).to_series()

            # scikit-learn's train_test_split now directly accepts Polars DataFrames and Series
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
            self.model.fit(self.X_train.to_numpy(), self.y_train.to_numpy())

            y_pred = self.model.predict(self.X_test.to_numpy())

            r2 = r2_score(self.y_test.to_numpy(), y_pred)
            mae = mean_absolute_error(self.y_test.to_numpy(), y_pred)
            mse = mean_squared_error(self.y_test.to_numpy(), y_pred)
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
async def alchemy_scale_numeric(scale_map: Dict[str, str]) -> str:
    """
    Scales numerical features independently on train and test sets.
    Fits scalers on training data only, then transforms both train and test.

    Args:
        scale_map: A dictionary that maps column names with scaling strategy.
    """

    return await session.scale_numeric(scale_map)
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