"""breakout 策略參數轉接層。"""

from config.execution_policy import EXECUTION_POLICY_PARAM_SPECS
from config.selection_policy import SELECTION_POLICY_PARAM_SPECS
from core.strategy_params import V16StrategyParams, strategy_params_to_dict
from strategies.breakout.schema import BREAKOUT_PARAM_SPECS


BREAKOUT_PARAM_GROUP_SPECS = {
    "strategy": BREAKOUT_PARAM_SPECS,
    "selection": SELECTION_POLICY_PARAM_SPECS,
    "execution": EXECUTION_POLICY_PARAM_SPECS,
}


def build_breakout_strategy_params(**overrides):
    return V16StrategyParams(**overrides)


def split_breakout_param_sections(params):
    payload = strategy_params_to_dict(params)
    return {
        section_name: {field_name: payload[field_name] for field_name in spec_map}
        for section_name, spec_map in BREAKOUT_PARAM_GROUP_SPECS.items()
    }
