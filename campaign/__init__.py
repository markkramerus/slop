"""
campaign — Campaign planner for syncom.

A standalone application that decomposes a natural-language scenario into a
structured campaign plan (JSON) that syncom can consume.

Usage
-----
    # Generate a campaign plan
    python campaign/planner.py --scenario brief.txt --rule-text rule.txt --output plan.json

    # Then use it with syncom
    python cli.py --campaign-plan plan.json --docket-id ... --rule-text ... --volume N --output ...
"""

from .campaign_models import CampaignPlan, ArgumentAngle
from .planner import generate_campaign_plan

__all__ = [
    "CampaignPlan",
    "ArgumentAngle",
    "generate_campaign_plan",
]
