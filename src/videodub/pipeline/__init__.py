"""pipeline — recipe orchestration. PORTABLE.

Public API:
    RECIPES                                     the catalogue (name -> stages)
    run_recipe(recipe, input, settings, ...)    run a recipe end to end
    PipelineContext                             the per-run artifact bag

Importing this package pulls in every stage's package, but not their heavy GPU
backends — those load lazily inside each stage's factory.
"""

from videodub.pipeline.context import PipelineContext
from videodub.pipeline.recipes import RECIPES
from videodub.pipeline.runner import run_recipe

__all__ = ["RECIPES", "PipelineContext", "run_recipe"]
