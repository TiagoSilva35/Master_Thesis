"""The pieces an equation is made of: a curve per continuous parent, one height per
level of a categorical parent, a surface per pair of parents."""

from src.causal_model.terms.levels import Levels
from src.causal_model.terms.spline import Spline
from src.causal_model.terms.surface import Surface

__all__ = ["Levels", "Spline", "Surface"]
