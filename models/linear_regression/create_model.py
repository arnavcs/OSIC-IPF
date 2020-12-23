# importing all the neccesarry dependencies
import os
import sys

import numpy as np
import pandas as pd

from sklearn import linear_model, ensemble
from sklearn.metrics import mean_squared_error, mean_absolute_error

import plotly.express as px
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')

import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf