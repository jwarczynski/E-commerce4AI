import pandas as pd
from typing import Dict, Any

class DataFrameBuilder:
    """Builds pandas DataFrames from query results."""
    def build(self, query_result: Dict[str, Any]) -> pd.DataFrame:
        """Convert query result to a DataFrame."""
        return pd.DataFrame(query_result["data"], columns=query_result["columns"])