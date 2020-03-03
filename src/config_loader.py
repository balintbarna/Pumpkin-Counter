# Non executable
"""
    Exposes a method for loading a configuration into
    a Python dictionary from a JSON file.

    Author:     frnyb
    Date:       20200303
"""

###############################################################
# Imports

import json

###############################################################
# Methods

def load_config(json_config_file = "config.json"):
        config = None

        with open(json_config_file) as f:
            config = json.load(f)

        return config