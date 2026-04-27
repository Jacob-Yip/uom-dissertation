from torch import nn

"""
Abstract base class of all models

Should not instantiate this directly
"""


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
