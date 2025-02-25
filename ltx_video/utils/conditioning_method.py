from enum import Enum


class ConditioningMethod(Enum):
    UNCONDITIONAL = "unconditional"
    FIRST_FRAME = "first_frame"
    MULTIPLE_FRAMES = "multiple_frames"
