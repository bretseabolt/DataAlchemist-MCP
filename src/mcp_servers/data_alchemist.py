import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from mcp.server.fastmcp import FastMCP
from typing import Optional, Dict, Union, List, Any
import os

import joblib
from dotenv import load_dotenv

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, classification_report


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
            if file_path.lower().endswith('.csv'):
                self.data = pl.read_csv(full_path, infer_schema_length=10000)
            elif file_path.lower().endswith('.xlsx'):
                self.data = pl.read_excel(full_path, sheet_id=0)
            else:
                return f"Error: Unsupported file format for {file_path}. Only .csv and .xlsx are supported."

            self._save_data()
            return f"Data loaded from {full_path}"
        except Exception as e:
            return f"Error loading data from {full_path}: {e}"

    async def reset_session(self) -> str:
        """
        Reset the current session by deleting the persisted data file and clearing in-memory data
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

        if self.working_dir:
            try:
                for file in os.listdir(self.working_dir):
                    if file.endswith((".csv", ".xlsx")):
                        os.remove(os.path.join(self.working_dir, file))
            except Exception as e:
                return f"Error deleting. .csv/.xlsx files during rest: {e}"

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


    async def plot_distribution(self, column_name: str, bins: Optional[int] = None) -> str:
        """
        Generates a histogram to visualize the distribution of a selected numerical column

        Args:
            column_name: The name of the column to plot (must be numerical)
            bins (Optional): Number of bins for the histogram (defaults to auto if None)

        """
        if self.data is None:
            if not self._load_data_from_session():
                return "Error: No data loaded. Please load data first using 'load_data'."

        if column_name not in self.data.columns:
            return f"Error: Column '{column_name}' not found in the dataset."

        col_dtype = self.data.schema[column_name]
        if col_dtype not in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8):
            return f"Error: Column '{column_name}' is not numerical (type: {col_dtype}). Distributions only allow numerical columns."
        try:
            col_series = self.data.select(pl.col(column_name)).to_series().to_numpy()

            plt.figure()
            plt.hist(col_series, bins='auto' if bins is None else bins)
            plt.title(f"Distribution of {column_name}")
            plt.xlabel(column_name)
            plt.ylabel("Frequency")

            plot_path = os.path.join(self.working_dir, f"distribution_{column_name}.png")
            plt.savefig(plot_path)
            plt.close()

            return f"Distribution for '{column_name}' generated and saved at {plot_path}."

        except Exception as e:
            return f"Error: Failed to generate distribution for {column_name}: {e}"

    async def plot_scatter(self, x_column: str, y_column: str, hue_column: Optional[str] = None) -> str:
        """
        Generates a scatter plot to visualize the relationship between two columns.
        Optionally colors points by a third categorical column (hue).

        Args:
            x_column: The name of the column for the x-axis (numerical or categorical).
            y_column: The name of the column for the y-axis (numerical or categorical).
            hue_column (Optional): The name of a categorical column to color points by.
        """
        if self.data is None:
            if not self._load_data_from_session():
                return "Error: No data loaded. Please load data first using 'load_data'."

        missing_cols = [col for col in [x_column, y_column, hue_column] if col and col not in self.data.columns]
        if missing_cols:
            return f"Error: Column(s) {missing_cols} not found in the dataset."

        try:
            x_data = self.data.select(pl.col(x_column)).to_series().to_numpy()
            y_data = self.data.select(pl.col(y_column)).to_series().to_numpy()

            plt.figure()
            if hue_column:
                hue_data = self.data.select(pl.col(hue_column)).to_series().to_numpy()
                unique_hues = np.unique(hue_data)
                colors = plt.cm.get_cmap('viridis', len(unique_hues))  # Use a colormap for categories
                color_map = {hue: colors(i) for i, hue in enumerate(unique_hues)}
                for hue in unique_hues:
                    mask = (hue_data == hue)
                    plt.scatter(x_data[mask], y_data[mask], label=str(hue), color=color_map[hue])
                plt.legend(title=hue_column)
                file_name = f"scatter_{x_column}_{y_column}_hue_{hue_column}.png"
            else:
                plt.scatter(x_data, y_data)
                file_name = f"scatter_{x_column}_{y_column}.png"

            plt.title(f"Scatter Plot: {x_column} vs {y_column}" + (f" (Hue: {hue_column})" if hue_column else ""))
            plt.xlabel(x_column)
            plt.ylabel(y_column)

            plot_path = os.path.join(self.working_dir, file_name)
            plt.savefig(plot_path)
            plt.close()

            return f"Scatter plot for '{x_column}' vs '{y_column}'{f' (hue: {hue_column})' if hue_column else ''} generated and saved at {plot_path}."

        except Exception as e:
            return f"Error: Failed to generate scatter plot: {e}"

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


    async def convert_to_numeric(self, column_name: str) -> str:
        """
        Converts values in a column to a numeric type (Float64). If values cannot be converted, set to null.

        Args:
            column_name: The name of the column to convert.
        """

        if self.data is None:
            if not self._load_data_from_session():
                return "Error: No data loaded. Please load data first using 'load_data'."

        column_name = column_name.strip()
        if not column_name:
            return "Error: Column name cannot be empty."

        if column_name not in self.data.columns:
            return f"Error: Column '{column_name}' not found in the dataset."


        try:

            self.data = self.data.with_columns(
                pl.col(column_name).cast(pl.Float64, strict=False).alias(column_name)
            )

            new_type = self.data.schema[column_name]


        except pl.exceptions.ComputeError as e:
            return f"Error: Column '{column_name}' could not be converted to numeric type: {e}"
        except Exception as e:
            return f"Unexpected error during conversion: {e}"



        self._save_data()

        return f"Column '{column_name}' successfully converted to numeric type ({new_type})."

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

    async def detect_outliers(self, outlier_map: Dict[str, str], z_score_threshold: float = 3.0) -> str:
        """
        Detects outliers in numerical columns using IQR or Z-score and returns a report.

        Args:
            outlier_map: A dictionary mapping column names to the detection strategy ('iqr' or 'z_score').
            z_score_threshold: The threshold for the Z-score method (defaults to 3.0).
        """
        if self.data is None:
            if not self._load_data_from_session():
                return "Error: No data loaded. Please load data first using 'load_data'."

        if not outlier_map:
            return "No outlier map provided; no outlier detection performed."

        report_lines = []
        errors = []

        for column_name, strategy in outlier_map.items():
            if column_name not in self.data.columns:
                errors.append(f"Error: Column '{column_name}' not found.")
                continue

            try:
                total_rows = self.data.shape[0]
                outliers = None

                if strategy == "iqr":
                    q1 = self.data[column_name].quantile(0.25)
                    q3 = self.data[column_name].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outliers = self.data.filter(
                        (pl.col(column_name) < lower_bound) | (pl.col(column_name) > upper_bound))

                elif strategy == "z_score":
                    mean = self.data[column_name].mean()
                    std = self.data[column_name].std()
                    z_scores = (self.data[column_name] - mean) / std
                    outliers = self.data.filter(z_scores.abs() > z_score_threshold)

                else:
                    errors.append(f"Invalid strategy '{strategy}' for column '{column_name}'.")
                    continue

                num_outliers = outliers.shape[0]
                percentage = (num_outliers / total_rows) * 100 if total_rows > 0 else 0
                report_lines.append(
                    f"Column '{column_name}' ({strategy}): Found {num_outliers} outliers ({percentage:.2f}%)."
                )

            except Exception as e:
                errors.append(f"Failed to detect outliers in column '{column_name}': {e}.")

        # Build the final report
        if not report_lines and not errors:
            return "No outliers detected with the specified methods."

        final_report = "--- Outlier Detection Report ---\n"
        if report_lines:
            final_report += "\n".join(report_lines)
        if errors:
            final_report += "\n\n--- Errors ---\n" + "\n".join(errors)

        return final_report.strip()

    async def handle_outliers(self, outlier_map: Dict[str, str], z_score_threshold: float = 3.0) -> str:
        """
        Handles outliers in numerical columns using either the IQR or Z-score method.

        Args:
            outlier_map: A dictionary mapping column names to the outlier handling strategy ('iqr' or 'z_score').
            z_score_threshold: The threshold for the Z-score method (defaults to 3.0).
        """
        if self.data is None:
            if not self._load_data_from_session():
                return "Error: No data loaded. Please load data first using 'load_data'."

        if not outlier_map:
            return "No outlier map provided; no changes made."

        allowed_strategies = ["iqr", "z_score"]
        successful_removals = []
        errors = []

        for column_name, strategy in outlier_map.items():
            if column_name not in self.data.columns:
                errors.append(f"Error: Column '{column_name}' not found in the dataset.")
                continue

            if strategy not in allowed_strategies:
                errors.append(
                    f"Invalid strategy '{strategy}' for column '{column_name}'. Valid strategies are {allowed_strategies}")
                continue

            try:
                initial_rows = self.data.shape[0]

                if strategy == "iqr":
                    q1 = self.data[column_name].quantile(0.25)
                    q3 = self.data[column_name].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    self.data = self.data.filter(
                        (pl.col(column_name) >= lower_bound) & (pl.col(column_name) <= upper_bound))

                elif strategy == "z_score":
                    mean = self.data[column_name].mean()
                    std = self.data[column_name].std()
                    self.data = self.data.filter(((self.data[column_name] - mean) / std).abs() <= z_score_threshold)

                rows_removed = initial_rows - self.data.shape[0]
                successful_removals.append(
                    f"Successfully removed {rows_removed} outliers from column '{column_name}' using the '{strategy}' method.")

            except Exception as e:
                errors.append(f"Failed to handle outliers in column '{column_name}' with strategy '{strategy}': {e}.")

        self._save_data()

        if len(successful_removals) > 0:
            removal_report = f"""
            Successfully handled outliers in columns: {list(outlier_map.keys())}.

            --- SUCCESSFUL REMOVALS ---
            {'\n'.join(successful_removals)}
            """
            if errors:
                removal_report += f"""
                --- ERRORS ---
                {'\n'.join(errors)}
                """
        else:
            removal_report = f"""
            Could not handle outliers in any of the specified columns.

            --- ERRORS ---
            {'\n'.join(errors)}
            """

        return removal_report.strip()

    async def scale_numeric(self, scale_map: Dict[str, str]) -> str:
        """
        Scales numerical features independently on train and test sets.
        Fits scalers on training data only, then transforms both train and test.

        Args:
            scale_map: A dictionary that maps column names with scaling strategy.
        """
        if self.X_train is None or self.X_test is None:
            if not self._load_data_from_session() or self.X_train is None:
                return "Error: Data has not been split. Please split the data first using 'split_data.'."

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

    async def drop_rows_with_null(self, column: Optional[Union[str, List[str]]] = None) -> str:
        """
        Drops rows with null values in a DataFrame based on column name and a threshold.
        If no column is specified, all columns will be considered.

        Args:
            column (Optional): Column name or list of column names to subset
        """
        if self.data is None:
            if not self._load_data_from_session():
                return "Error: No data loaded. Please load data first using 'load_data'."

        try:
            self.data = self.data.drop_nulls(subset=column)
            self._save_data()
            return f"Successfully dropped rows with null values."
        except Exception as e:
            return f"Error dropping rows with null values: {e}"

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

            split_report = f"""Data successfully split into training and testing sets.
            
                    Training set shape: {self.X_train.shape}
                    
                    Testing set shape: {self.X_test.shape}
                    """

            return split_report.strip()
        except Exception as e:
            return f"Error occurred during data splitting: {e}"

    async def perform_kfold_cv(self, model: str, target_column: str, k: int=5,
                               scoring: str = None, random_state: int = 42) -> str:
        """
                Performs K-Fold Cross Validation on the full dataset for the specified model type.

                Args:
                    model: The type of model to use ('linear_regression', 'logistic_regression', 'random_forest_regression', 'random_forest_classification').
                    target_column: The target column for prediction.
                    k: Number of folds for CV (default: 5).
                    scoring: Scoring metric (optional; e.g., 'r2' for regression, 'accuracy' for classification. Defaults to appropriate for model).
                    random_state: Seed for reproducibility (default: 42).
        """

        if self.data is None:
            if not self._load_data_from_session():
                return "Error: No data loaded. Please load data first."

        if target_column not in self.data.columns:
            return f"Error: target column {target_column}"

        try:
            X = self.data.drop(target_column)
            y = self.data.select(pl.col(target_column)).to_series()

            if model == "linear_regression":
                model = LinearRegression()
                default_scoring = "r2"
            elif model == "logistic_regression":
                model = LogisticRegression()
                default_scoring = "accuracy"
            if model == "random_forest_regression":
                model = RandomForestRegressor()
                default_scoring = "r2"
            if model == "random_forest_classification":
                model = RandomForestClassifier()
                default_scoring = "accuracy"

            else:
                return f"Error: Unsupported model '{model}'. Supported: linear_regression, logistic_regression, random_forest_regression, random_forest_classification."

            scoring = scoring if scoring else default_scoring

            cv = KFold(n_splits=k, shuffle=True, random_state=random_state)

            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

            cv_report = f"""
            K-Fold Cross Validation for {model}' with k={k}, scoring={scoring}.

            --- SCORES PER FOLD ---
            {scores}

            --- MEAN SCORE ---
            {np.mean(scores):.4f}

            --- STD DEV ---
            {np.std(scores):.4f}
            """
            return cv_report.strip()
        except Exception as e:
            return f"Error occurred during K-Fold CV: {e}"



    async def train_linear_regression(self) -> str:
        """
        Trains a linear regression model and evaluates its performance on the test data.
        """
        if self.X_train is None or self.y_train is None:
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

    async def train_logistic_regression(self, penalty: Optional[str] = 'l2', l1_ratio: Optional[float] = None) -> str:
        """
            Trains a logistic regression model with optional penalty ('l1', 'l2', 'elasticnet', or None) and evaluates its performance on the test data.
            For 'elasticnet', optionally provide l1_ratio (0-1; defaults to 0.5).

            Args:
                penalty: The regularization penalty to use (optional; defaults to 'l2').
                l1_ratio: The Elastic-Net mixing parameter (optional; only for 'elasticnet').
            """
        if self.X_train is None or self.y_train is None:
            if not self._load_data_from_session() or self.X_train is None or self.y_train is None:
                return "Error: Data has not been split. Please split the data first."

        try:
            penalty_solver_map = {
                'l1': 'liblinear',
                'l2': 'lbfgs',
                'elasticnet': 'saga',
                None: 'lbfgs',
            }

            solver = penalty_solver_map[penalty]

            extra_kwargs = {'l1_ratio': l1_ratio if l1_ratio is not None else 0.5} if penalty == 'elasticnet' else {}

            self.model = LogisticRegression(penalty=penalty, solver=solver, max_iter=1000, **extra_kwargs)
            self.model.fit(self.X_train.to_numpy(), self.y_train.to_numpy())

            y_pred = self.model.predict(self.X_test.to_numpy())

            if self.y_train.dtype == pl.Categorical:
                target_names = self.y_train.cat.get_categories().to_list()
            elif self.y_train.dtype in (pl.Utf8, pl.String):
                target_names = self.y_train.unique().sort().to_list()
            else:
                target_names = self.y_train.unique().sort().cast(pl.String).to_list()

            class_report = classification_report(self.y_test.to_numpy(), y_pred, output_dict=True,
                                                 target_names=target_names if target_names else None)

            acc = accuracy_score(self.y_test.to_numpy(), y_pred)

            self._save_data()

            report = f"""
            Logistic Regression model trained successfully.
    
            --- MODEL PERFORMANCE ON TEST SET ---
            
            "Accuracy": {round(acc, 4)}
            
            --- CLASSIFICATION REPORT ---
            
            {class_report}
            
            """

            return report.strip()

        except KeyError:
            return f"Error: invalid penalty '{penalty}. Supported: 'l1', 'l2', 'elasticnet', None."
        except Exception as e:
            return f"Error occurred during model training: {e}"

    async def train_random_forest_regression(self, n_estimators: int =100, max_depth: Optional[int] = None, random_state: int = 42) -> str:
        """
        Trains a Random Forest regression model and evaluates its performance on the test set.

        Args:
            n_estimators: Number of trees in the forest (default: 100).
            max_depth: Maximum depth of the tree (optional, default None for unlimited).
            random_state: Seed for reproducibility (default: 42).
        """
        if self.X_train is None or self.y_train is None:
            if not self._load_data_from_session() or self.X_train is None or self.y_train is None:
                return "Error: Data has not been split. Please split the data first."

        try:
            self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                               random_state=random_state)
            self.model.fit(self.X_train.to_numpy(), self.y_train.to_numpy().ravel())  # Ensure y is 1D

            y_pred = self.model.predict(self.X_test.to_numpy())

            r2 = r2_score(self.y_test.to_numpy(), y_pred)
            mae = mean_absolute_error(self.y_test.to_numpy(), y_pred)
            mse = mean_squared_error(self.y_test.to_numpy(), y_pred)
            rmse = np.sqrt(mse)

            rf_reg_report = f"""
                    Random Forest Regressor trained successfully.

                    --- MODEL PERFORMANCE ON TEST SET ---
                    R-Squared: {round(r2, 4)}
                    Mean Absolute Error (MAE): {round(mae, 4)}
                    Mean Squared Error (MSE): {round(mse, 4)}
                    Root Mean Squared Error (RMSE): {round(rmse, 4)}
                    """

            self._save_data()

            return rf_reg_report.strip()

        except Exception as e:
            return f"Error occurred during Random Forest regression training: {e}"

    async def train_random_forest_classification(self, n_estimators: int =100, max_depth: Optional[int] = None, random_state: int = 42) -> str:
        """
        Trains a Random Forest classification model and evaluates its performance on the test set.

        Args:
            n_estimators: Number of trees in the forest (default: 100).
            max_depth: Maximum depth of the tree (optional, default None for unlimited).
            random_state: Seed for reproducibility (default: 42).
        """
        if self.X_train is None or self.y_train is None:
            if not self._load_data_from_session() or self.X_train is None or self.y_train is None:
                return "Error: Data has not been split. Please split the data first."

        try:
            self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
            self.model.fit(self.X_train.to_numpy(), self.y_train.to_numpy().ravel())

            y_pred = self.model.predict(self.X_test.to_numpy())

            acc = accuracy_score(self.y_test.to_numpy(), y_pred)

            if self.y_train.dtype == pl.Categorical:
                target_names = self.y_train.cat.get_categories().to_list()
            elif self.y_train.dtype in (pl.Utf8, pl.String):
                target_names = self.y_train.unique().sort().to_list()
            else:
                target_names = self.y_train.unique().sort().cast(pl.String).to_list()

            class_report = classification_report(self.y_test.to_numpy(), y_pred, output_dict=True, target_names=target_names if target_names else None)

            rf_class_report = f"""
            Random Forest Classifier trained successfully.

            --- MODEL PERFORMANCE ON TEST SET ---
            Accuracy: {round(acc, 4)}

            --- CLASSIFICATION REPORT ---
            {class_report}
            """

            self._save_data()
            return rf_class_report.strip()
        except Exception as e:
            return f"Error occurred during Random Forest classification training: {e}"


session = DataAlchemist()

@mcp.tool()
async def alchemy_load_data(file_path: str) -> str:
    """
    Load data from a CSV or Excel file into the session for data preprocessing.

    Args:
        file_path: Path of the file to load (must be .csv or .xlsx
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
async def alchemy_plot_distribution(column_name: str, bins: Optional[int] = None) -> str:
    """
    Generates a histogram to visualize the distribution of a selected numerical column

    Args:
        column_name: The name of the column to plot (must be numerical)
        bins (Optional): Number of bins for the histogram (defaults to auto if None)

    """
    return await session.plot_distribution(column_name, bins)

@mcp.tool()
async def alchemy_plot_scatter(x_column: str, y_column: str, hue_column: Optional[str] = None) -> str:
    """
    Generates a scatter plot to visualize the relationship between two columns.
    Optionally colors points by a third categorical column (hue).

    Args:
        x_column: The name of the column for the x-axis (numerical or categorical).
        y_column: The name of the column for the y-axis (numerical or categorical).
        hue_column (Optional): The name of a categorical column to color points by (e.g., for grouping).
    """
    return await session.plot_scatter(x_column, y_column, hue_column)

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
async def alchemy_convert_to_numeric(column_name: str) -> str:
    """
    Converts values in a column to a numeric type (Float64). If values cannot be converted, set to null.

    Args:
        column_name: The name of the column to convert.
    """

    return await session.convert_to_numeric(column_name)


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
async def alchemy_detect_outliers(outlier_map: Dict[str, str], z_score_threshold: float = 3.0) -> str:
    """
    Detects and reports outliers in numerical columns without removing them.

    Args:
        outlier_map: A dictionary mapping column names to the outlier detection strategy ('iqr' or 'z_score').
                     Example: {"age": "iqr", "income": "z_score"}
        z_score_threshold (Optional): The Z-score threshold for outlier detection (defaults to 3.0).
    """
    return await session.detect_outliers(outlier_map, z_score_threshold)

@mcp.tool()
async def alchemy_handle_outliers(outlier_map: Dict[str, str], z_score_threshold: float = 3.0) -> str:
    """
    Handles outliers in numerical columns using either the IQR or Z-score method.

    Args:
        outlier_map: A dictionary mapping column names to the outlier handling strategy ('iqr' or 'z_score').
                     Example: {"age": "iqr", "income": "z_score"}
        z_score_threshold (Optional): The Z-score threshold to use for outlier detection (defaults to 3.0).
    """
    return await session.handle_outliers(outlier_map, z_score_threshold)

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
async def alchemy_drop_rows_with_null(column: Optional[Union[str, List[str]]]) -> str:
    """
    Drops rows with null values in a DataFrame based on column name and a threshold.
    If no column is specified, all columns will be considered.

    Args:
        column (Optional): Column name or list of column names to subset
    """

    return await session.drop_rows_with_null(column)

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
async def perform_kfold_cv(model: str, target_column: str, k: int = 5,
                           scoring: str = None, random_state: int = 42) -> str:
    """
            Performs K-Fold Cross Validation on the full dataset for the specified model type.

            Args:
                model: The type of model to use ('linear_regression', 'logistic_regression', 'random_forest_regression', 'random_forest_classification').
                target_column: The target column for prediction.
                k: Number of folds for CV (default: 5).
                scoring: Scoring metric (optional; e.g., 'r2' for regression, 'accuracy' for classification. Defaults to appropriate for model).
                random_state: Seed for reproducibility (default: 42).
    """

    return await session.perform_kfold_cv(model, target_column, k, scoring, random_state)

@mcp.tool()
async def alchemy_train_linear_regression() -> str:
    """
    Trains a linear regression model and evaluates its performance on the test data.
    """
    return await session.train_linear_regression()

@mcp.tool()
async def alchemy_train_logistic_regression(penalty: Optional[str] = 'l2', l1_ratio: Optional[float] = None) -> str:
    """
    Trains a logistic regression model with optional penalty ('l1', 'l2', 'elasticnet', or None) and evaluates its performance on the test data.
    For 'elasticnet', optionally provide l1_ratio (0-1; defaults to 0.5).

    Args:
        penalty: The regularization penalty to use (optional; defaults to 'l2').
        l1_ratio: The Elastic-Net mixing parameter (optional; only for 'elasticnet').
    """
    return await session.train_logistic_regression(penalty, l1_ratio)

@mcp.tool()
async def alchemy_train_random_forest_regression(n_estimators: int =100, max_depth: Optional[int] = None, random_state: int =42) -> str:
        """
        Trains Random Forest for regression.
        Args:
            n_estimators: Number of trees.
            max_depth: Max tree depth.
            random_state: Random seed.
    """
        return await session.train_random_forest_regression(n_estimators, max_depth, random_state)

@mcp.tool()
async def alchemy_train_random_forest_classification(n_estimators: int =100, max_depth: Optional[int] = None, random_state: int =42) -> str:
        """
        Trains Random Forest for classification.
        Args:
            n_estimators: Number of trees.
            max_depth: Max tree depth.
            random_state: Random seed.
    """
        return await session.train_random_forest_classification(n_estimators, max_depth, random_state)

if __name__ == "__main__":
    mcp.run(transport='stdio')