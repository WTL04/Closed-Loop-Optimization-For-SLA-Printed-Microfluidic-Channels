import numpy as np
# from ax.api.client import Client
# from ax.api.configs import RangeParameterConfig

# archieved version 
from ax.service.ax_client import AxClient

client = AxClient() # handles state of experiments

# explerimental data
parameters = [
    {"name":"layer_thickness",
     "type":""
    },
    {"name":"exposure_time",
     "type":"range",
     "bounds":[]
    }
]

client.create_experiment(
    parameters=parameters
)
