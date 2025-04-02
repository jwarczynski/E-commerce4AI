import os
from typing import Any, Dict, List, Optional
import json

import snowflake.connector
import requests
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Snowflake connection details from environment variables
HOST = os.getenv("SNOWFLAKE_HOST")
ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
USER = os.getenv("SNOWFLAKE_USER")
PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
ROLE = os.getenv("SNOWFLAKE_ROLE")

DATABASE = "DOG_CORTEX_ANALYST_DEMO"
SCHEMA = "REVENUE_TIMESERIES"
STAGE = "RAW_DATA"
FILE = "revenue_timeseries.yaml"
WAREHOUSE = "DOG_WH"

# Initialize Snowflake connection
CONN = snowflake.connector.connect(
    user=USER,
    password=PASSWORD,
    account=ACCOUNT,
    host=HOST,
    port=443,
    warehouse=WAREHOUSE,
    role=ROLE,
)


# --- Snowflake Stage Management ---
def upload_to_stage(local_file: str, stage_path: str):
    """Upload a local file to the Snowflake stage."""
    cursor = CONN.cursor()
    try:
        cursor.execute(f"PUT file://{local_file} @{stage_path} OVERWRITE=TRUE")
        print(f"Uploaded {local_file} to {stage_path}")
    except Exception as e:
        raise Exception(f"Failed to upload file to stage: {str(e)}")
    finally:
        cursor.close()


def ensure_stage_exists():
    """Create the stage if it doesn’t exist."""
    cursor = CONN.cursor()
    try:
        cursor.execute(f"CREATE STAGE IF NOT EXISTS {DATABASE}.{SCHEMA}.{STAGE}")
        print(f"Ensured stage {DATABASE}.{SCHEMA}.{STAGE} exists")
    except Exception as e:
        raise Exception(f"Failed to create stage: {str(e)}")
    finally:
        cursor.close()

def load_local_semantic_model(file_path: str) -> str:
    """Load a local semantic model YAML file as a string."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Semantic model file not found: {file_path}")
    with open(file_path, 'r') as file:
        yaml_content = file.read()
        return yaml_content


# --- Cortex Analyst API Interaction ---
def send_cortex_message(prompt: str, semantic_model: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
    """Send a message to Cortex Analyst API and return the response."""
    request_body = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    }
    if semantic_model:
        request_body["semantic_model"] = semantic_model
    else:
        request_body["semantic_model_file"] = f"@{DATABASE}.{SCHEMA}.{STAGE}/{FILE}"

    resp = requests.post(
        url=f"https://{HOST}/api/v2/cortex/analyst/message",
        json=request_body,
        headers={
            "Authorization": f'Snowflake Token="{CONN.rest.token}"',
            "Content-Type": "application/json",
        },
    )
    request_id = resp.headers.get("X-Snowflake-Request-Id")
    if resp.status_code < 400:
        return {**resp.json(), "request_id": request_id}
    else:
        raise Exception(f"Failed request (id: {request_id}): {resp.text}")


def validate_sql(sql: str) -> bool:
    """Validate SQL by executing it directly via the Snowflake connector."""
    cursor = CONN.cursor()
    try:
        # Use EXPLAIN to validate without fully executing (less resource-intensive)
        cursor.execute(f"EXPLAIN {sql}")
        print("SQL validated successfully")
        return True
    except snowflake.connector.errors.ProgrammingError as e:
        print(f"SQL validation failed: {str(e)}")
        return False
    finally:
        cursor.close()


# --- YAML Handling ---
def load_yaml(file_path: str) -> Dict:
    """Load the local YAML file."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(file_path: str, data: Dict):
    """Save the updated YAML file."""
    with open(file_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def validate_yaml(yaml_data: Dict) -> bool:
    """Validate YAML structure (simplified)."""
    required_keys = {"tables"}
    return all(key in yaml_data for key in required_keys) and isinstance(yaml_data["tables"], list)


def update_verified_queries(file_path: str, query_name: str, sql: str):
    """Add a validated SQL query to the verified_queries section."""
    yaml_data = load_yaml(file_path)
    if "verified_queries" not in yaml_data:
        yaml_data["verified_queries"] = []

    if validate_sql(sql):
        yaml_data["verified_queries"].append({"name": query_name, "sql": sql})
        save_yaml(file_path, yaml_data)
        print(f"Added verified query '{query_name}' to {file_path}")
    else:
        print(f"SQL validation failed for '{query_name}'")


# --- Programmatic Semantic Model Creation ---
def generate_semantic_model(file_path: str):
    """Generate a semantic model YAML from Snowflake schema DDL."""
    cursor = CONN.cursor()
    cursor.execute(f"SELECT GET_DDL('SCHEMA', '{SCHEMA}');")
    ddl = cursor.fetchone()[0]

    yaml_data = {
        "version": "1.0",
        "tables": [
            {
                "name": "daily_revenue",
                "description": "Daily revenue data with actuals and forecasts",
                "base_table": f"{DATABASE}.{SCHEMA}.daily_revenue",
                "columns": [
                    {"name": "date", "description": "Date of revenue", "data_type": "DATE"},
                    {"name": "revenue", "description": "Actual revenue", "data_type": "FLOAT"},
                    {"name": "cogs", "description": "Cost of goods sold", "data_type": "FLOAT"},
                    {"name": "forecasted_revenue", "description": "Forecasted revenue", "data_type": "FLOAT"},
                ],
            }
        ],
        "verified_queries": [],
    }
    save_yaml(file_path, yaml_data)
    print(f"Generated semantic model at {file_path}")


# --- Feature Engineering Agent ---
def enhance_yaml_for_feature_engineering(file_path: str):
    """Modify YAML to support executable feature engineering queries."""
    yaml_data = load_yaml(file_path)
    for table in yaml_data["tables"]:
        if table["name"] == "daily_revenue":
            table["description"] += " Supports aggregations and window functions for feature engineering."
            table["columns"].extend(
                [
                    {"name": "total_daily_revenue", "description": "Sum of revenue per day", "data_type": "FLOAT"},
                    {"name": "total_daily_cogs", "description": "Sum of cogs per day", "data_type": "FLOAT"},
                    {
                        "name": "total_daily_forecasted_revenue",
                        "description": "Sum of forecasted revenue per day",
                        "data_type": "FLOAT"
                    },
                    {
                        "name": "seven_day_moving_avg",
                        "description": "7-day moving average of total_daily_revenue",
                        "data_type": "FLOAT"
                    },
                ]
            )
    save_yaml(file_path, yaml_data)
    print("Enhanced YAML for feature engineering")


def generate_feature_query(prompt: str, semantic_model: Optional[Dict[str, Any]]=None) -> str:
    """Generate and return an executable feature engineering SQL query."""
    response = send_cortex_message(prompt, semantic_model)
    for item in response["message"]["content"]:
        if item["type"] == "sql":
            return item["statement"]
    raise ValueError("No SQL generated by Cortex Analyst")


# --- LLM-Based Format Validation ---
def validate_query_format(sql: str) -> bool:
    """Use a mock LLM to validate SQL format (simplified)."""
    required_clauses = ["WITH", "SELECT", "FROM", "GROUP BY", "ORDER BY"]
    return all(clause in sql.upper() for clause in required_clauses)


# --- Main Execution ---
def main():
    semantic_model_path = "revenue_timeseries.yaml"
    stage_path = f"{DATABASE}.{SCHEMA}.{STAGE}"

    # Step 1: Ensure stage exists
    ensure_stage_exists()

    # Step 2: Generate semantic model if it doesn’t exist
    if not os.path.exists(semantic_model_path):
        generate_semantic_model(semantic_model_path)

    # Step 3: Enhance YAML for feature engineering
    # enhance_yaml_for_feature_engineering(semantic_model_path)

    # Step 4: Upload YAML to Snowflake stage
    # upload_to_stage(semantic_model_path, stage_path)

    # Step 5: Generate and validate a feature engineering query
    semantic_model = yaml_data = load_yaml(semantic_model_path)
    prompt = "Generate an executable SQL query to calculate total daily revenue, cogs, forecasted revenue, and a 7-day moving average."
    try:
        # sql = generate_feature_query(prompt, semantic_model)
        sql = generate_feature_query(prompt, )
        print(f"Generated SQL:\n{sql}")

        # Step 6: Validate SQL and update verified_queries
        if validate_sql(sql) and validate_yaml(load_yaml(semantic_model_path)):
            update_verified_queries(semantic_model_path, "daily_revenue_features", sql)
            upload_to_stage(semantic_model_path, stage_path)  # Re-upload updated YAML
        else:
            print("Validation failed")

        # Step 7: LLM-based format validation
        if validate_query_format(sql):
            print("SQL format is valid")
        else:
            print("SQL format is invalid")
    except Exception as e:
        print(f"Error during query generation: {str(e)}")


if __name__ == "__main__":
    main()
