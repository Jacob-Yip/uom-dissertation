from manim import *
from ambiguity_decomposition import AmbiguityDecomposition
from bias_variance import BiasVariance
from bias_variance_diversity import BiasVarianceDiversity
from ncl import NCL

"""
Run: 
manim -pql background.py Background
"""


class Background(Scene):
    def construct(self):
        AmbiguityDecomposition.construct(self)
        self.wait(0.3)
        BiasVariance.construct(self)
        self.wait(0.8)
        BiasVarianceDiversity.construct(self)
        self.wait(0.5)
        NCL.construct(self)
        self.wait(2)
