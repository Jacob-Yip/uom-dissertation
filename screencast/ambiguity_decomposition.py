from manim import *

"""
Run: 
manim -pql ambiguity_decomposition.py AmbiguityDecomposition
"""


class AmbiguityDecomposition(Scene):
    def construct(self):
        # --- Titles ---
        title = Text("Ambiguity Decomposition", color=BLUE).to_edge(UP)
        self.play(Write(title))

        # 1. Setup Equations (using smaller font to fit screen)
        scale_val = 0.7

        # Equation after applying Ambiguity Decomposition
        #  =  -
        bias_variance_decomposition = MathTex(
            r"\underbrace{(y - \bar{y})^2}_{\text{ensemble loss}}",
            r" = ",
            r"\underbrace{\frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y)^2}_{\text{average loss}}",
            r" - ",
            r"\underbrace{\frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - \bar{y})^2}_{ambiguity}",
        ).scale(scale_val)

        # --- Start of animation ---

        # Ambiguity Decomposition
        self.wait(0.5)
        self.play(
            Write(bias_variance_decomposition[:]),
        )
        self.wait(5.5)

        # End
        self.play(*[FadeOut(mob) for mob in self.mobjects])
