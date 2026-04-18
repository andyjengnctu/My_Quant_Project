"""相容層：正式 train/test 切分參數已移至 config.training_policy。"""

from config.training_policy import TRAINING_SPLIT_POLICY

WALK_FORWARD_POLICY = dict(TRAINING_SPLIT_POLICY)
