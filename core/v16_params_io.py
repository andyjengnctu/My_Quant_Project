import json
import os

from core.v16_config import V16StrategyParams
from core.v16_log_utils import format_exception_summary


def load_params_from_json(json_file):
    params = V16StrategyParams()
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"找不到參數檔: {json_file}")

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for key, value in data.items():
            if hasattr(params, key):
                setattr(params, key, value)

        return params
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as e:
        raise RuntimeError(f"讀取參數檔 {json_file} 失敗: {format_exception_summary(e)}") from e