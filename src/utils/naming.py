from __future__ import annotations


PROTOCOL_DISPLAY_NAMES = {
    "standard_leach": "LEACH",
    "energy_aware_leach": "EA-LEACH",
    "tcn_predictive_pollution_aware_leach": "TCN-PPA-LEACH",
    "full_tcn_ppa_leach": "TCN-PPA-LEACH",
    "no_tcn_prediction": "No Prediction",
    "no_aoi_term": "No AoI",
    "no_suppression": "No Suppression",
    "no_priority_scheduler": "No Priority Scheduler",
}

SCENARIO_DISPLAY_NAMES = {
    "normal": "Normal",
    "rising_warning": "Rising Warning",
    "hazardous_spike": "Hazardous Spike",
    "hotspot_heavy": "Hotspot Heavy",
    "sensitivity_nodes_24": "Sensitivity: 24 Nodes",
    "sensitivity_nodes_48": "Sensitivity: 48 Nodes",
    "sensitivity_area_140": "Sensitivity: Area 140 m",
    "sensitivity_energy_12": "Sensitivity: 0.12 J",
}


def protocol_label(protocol_id: str) -> str:
    return PROTOCOL_DISPLAY_NAMES.get(protocol_id, protocol_id.replace("_", " ").title())


def scenario_label(scenario_id: str) -> str:
    return SCENARIO_DISPLAY_NAMES.get(scenario_id, scenario_id.replace("_", " ").title())
