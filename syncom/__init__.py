"""
slop — Synthetic Locus of Public comments

A research toolkit for generating synthetic public regulatory comments to
support the development of comment-spam detection systems.

Conceptual pipeline
-------------------
1. Ingest a previous docket CSV → PopulationModel
2. Analyse the proposed rule text → WorldModel
3. For each desired synthetic comment:
   a. Sample a persona from the population distribution
   b. Generate a personal hook for the persona
   c. Map the attack objective × vector × persona → ExpressionFrame
   d. Generate the comment text
   e. Run QC (relevance, argument presence, embedding dedup)
4. Export accepted comments to a Regulations.gov-format CSV

Quick start
-----------
See cli.py for the command-line interface, or import the library directly:

    from slop import pipeline
    pipeline.run(...)
"""

from config import Config
from shared_models import PopulationModel
from .world_model import build_world_model, WorldModel
from .persona import sample_persona, Persona
from .argument_mapper import map_argument, ExpressionFrame
from .generator import generate_comment, GeneratedComment
from .quality_control import QualityController
from .export import export_to_txt
from .pipeline import run_campaign, run_campaign_async

__all__ = [
    "Config",
    "ingest_docket_csv",
    "PopulationModel",
    "build_world_model",
    "WorldModel",
    "sample_persona",
    "Persona",
    "map_argument",
    "ExpressionFrame",
    "generate_comment",
    "GeneratedComment",
    "QualityController",
    "export_to_txt",
    "run_campaign",
    "run_campaign_async",
]
