from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "config" / "log_schemas"

class LogSchema:
    def __init__(self, schema_dict: dict):
        self.log_name = schema_dict["log_name"]
        self.log_description = schema_dict["log_description"]
        cols = schema_dict["columns"]
        self.case_id = cols["case_id"]                                  # mandatory in log
        self.activity = cols["activity"]                                # mandatory in log
        self.timestamp = cols["timestamp"]                              # mandatory in log
        self.resource = cols.get("resource")                            # optionally in log
        self.case_attributes = schema_dict.get("case_attributes", [])   # optionally in log

def load_log_schema(log_name):
    """Loads log schema from log's schema file"""
    path = SCHEMA_DIR / f"{log_name}.yaml"

    if not path.exists():
        raise FileNotFoundError(f"No schema found for log {log_name} at {path}")

    with open(path, "r", encoding="utf-8") as f:
        schema_dict = yaml.safe_load(f)

    return LogSchema(schema_dict)
