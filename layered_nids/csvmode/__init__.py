"""CSV mode: one robust entry point for flow files.

It combines the earlier UNSW-NB15 project's model (34 flow features), the NetFlow-feature model (40 NetFlow v2 features) and
host/time rules for Argus-style flow files (the format Slips reads). Run:  python -m layered_nids.csvmode FILE.csv
"""
