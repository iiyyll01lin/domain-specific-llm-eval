from typing import Any, Dict, List

import pandas as pd

from typing import List, Dict, Any
# from .document_loader import Document


class SQLDocumentLoader:
    """
    Medium Priority Feature: Dynamic Database Synthesizer bridging.
    Loads data via SQL queries mirroring the Text-to-SQL pipeline flows.
    """

    def __init__(self, connection_string: str):
        self.connection_string = connection_string

    def load_documents(
        self, query: str
    ) -> List[Any]:  # Return structure mimicking Document loader
        try:
            import sqlalchemy

            # Standard generic pandas read sql
            df = pd.read_sql(query, self.connection_string)

            docs = []
            for _, row in df.iterrows():
                # Convert row to dictionary format or Document
                content = " | ".join(
                    f"{k}: {v}" for k, v in row.to_dict().items() if pd.notnull(v)
                )
                docs.append({"page_content": content, "metadata": row.to_dict()})

            return docs

        except Exception as e:
            print(f"Failed to execute dynamic DB synthesizer query: {e}")
            return []
