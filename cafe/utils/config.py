import os
from dotenv import load_dotenv

def load_config():
    """Load configuration from environment variables."""
    load_dotenv()
    return {
        "snowflake": {
            "host": os.getenv("SNOWFLAKE_HOST", "sfedu02-llb96263.snowflakecomputing.com"),
            "organization": os.getenv("SNOWFLAKE_ORG", "sfedu02"),
            "account": os.getenv("SNOWFLAKE_ACCOUNT", "llb96263"),
            "user": os.getenv("SNOWFLAKE_USER", "DOG"),
            "role": os.getenv("SNOWFLAKE_ROLE", "TRAINING_ROLE"),
            "warehouse": os.getenv("SNOFLAKE_WAREHOUSE", "DOG_WH"),
            "database": os.getenv("SNOFLAKE_DB", "DOG_CORTEX_ANALYST_DEMO"),
            "schema": os.getenv("SNOFLAKE_SCHEMA", "REVENUE_TIMESERIES"),
            "stage": os.getenv("SNOWFLAKE_STAGE", "RAW_DATA"),
            "private_key_path": os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH", "~/.ssh/snowflake_private_key.p8"),
        }
    }