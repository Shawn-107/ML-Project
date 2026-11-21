"""NIDS flow model for UNSW-NB15 style flow records: gradient boosting on 11 inputs + a K-Means 'unusual traffic' flag.

This is the model of the owner's earlier project (package `ids`), moved into this repository, with its thresholds set on a
validation split that holds out whole K-Means clusters (so false alarms land close to the target). It won the head-to-head
against the 34-feature model on false alarms at equal accuracy: see reports/flows/compare_earlier_project.json.
"""
