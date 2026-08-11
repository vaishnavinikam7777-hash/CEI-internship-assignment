"""
Optimization Insights
----------------------
Turns the pattern-analysis output into concrete, explainable energy
optimization recommendations for a household — e.g. shifting load off
peak hours, flagging unusually spiky usage, tariff suitability.

This is a rule-based reasoning layer over the quantitative pattern
analysis, so every recommendation is directly traceable to a number
in `peak_offpeak_analysis()` / cluster assignment (explainable by
construction, no black-box text generation needed here).
"""

PEAK_HOURS_GRID = set(range(17, 21))  # typical UK evening peak, 5-9pm


def generate_insights(peak_info: dict, cluster_label: str) -> dict:
    insights = []
    savings_actions = []

    ratio = peak_info["peak_to_average_ratio"]
    if ratio >= 2.5:
        insights.append(
            f"Usage is highly spiky (peak-to-average ratio {ratio}x) — a small number of "
            "high-draw appliances are driving most of the bill. Spreading their usage "
            "would reduce both cost and grid strain."
        )
    elif ratio >= 1.7:
        insights.append(
            f"Usage has a moderate peak (peak-to-average ratio {ratio}x), typical of a "
            "household with concentrated morning/evening activity."
        )
    else:
        insights.append(
            f"Usage is relatively flat across the day (peak-to-average ratio {ratio}x), "
            "which is efficient from a grid-load perspective."
        )

    top_hours = [h for h, _ in peak_info["top_peak_hours"]]
    grid_peak_overlap = [h for h in top_hours if h in PEAK_HOURS_GRID]
    if grid_peak_overlap:
        insights.append(
            f"Peak usage hours ({', '.join(f'{h}:00' for h in top_hours)}) overlap with the "
            f"typical grid peak window (5-9pm). Time-of-use tariffs would likely charge a "
            f"premium during {', '.join(f'{h}:00' for h in grid_peak_overlap)}."
        )
        savings_actions.append(
            "Shift deferrable loads (washing machine, dishwasher, EV/appliance charging) "
            "to off-peak hours (e.g. after 10pm or before 6am) to cut time-of-use tariff costs."
        )
    else:
        insights.append(
            f"Peak usage hours ({', '.join(f'{h}:00' for h in top_hours)}) fall outside the "
            "typical grid peak window, which is favorable for time-of-use tariffs."
        )
        savings_actions.append(
            "This household is a good candidate for a time-of-use or Economy-7-style tariff, "
            "since its natural peak avoids the most expensive grid hours."
        )

    weekday = peak_info["weekday_avg_kwh_hh"]
    weekend = peak_info["weekend_avg_kwh_hh"]
    if weekend > weekday * 1.15:
        insights.append(
            f"Weekend usage ({weekend} kWh/hh avg) is notably higher than weekday usage "
            f"({weekday} kWh/hh avg), consistent with more time spent at home."
        )
    elif weekday > weekend * 1.15:
        insights.append(
            f"Weekday usage ({weekday} kWh/hh avg) is notably higher than weekend usage "
            f"({weekend} kWh/hh avg)."
        )

    insights.append(f"Behavioral segment: {cluster_label} — based on the household's typical daily load shape.")

    savings_actions.append(
        "Run an energy audit on appliances active during the top-3 peak hours "
        "(usually water heating, HVAC, or cooking loads) — these dominate the bill."
    )
    savings_actions.append(
        "Consider smart plugs/scheduling on high-draw, non-time-sensitive appliances "
        "to automatically shift them to cheaper hours."
    )

    return {
        "insights": insights,
        "recommended_actions": savings_actions,
    }
