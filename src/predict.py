import pandas as pd
import numpy as np
import os
from train_evaluate import Split_Data

class Predict:

    def __init__(self):
        self.report = Split_Data.train_model()

    def Save_Model(self):
        self.report.models

