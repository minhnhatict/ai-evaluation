# Add your utilities or helper functions to this file.

import os

def get_phoenix_endpoint():
    phoenix_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
    return phoenix_endpoint


