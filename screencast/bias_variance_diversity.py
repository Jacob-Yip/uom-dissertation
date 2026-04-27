from manim import *

"""
Run: 
manim -pql bias_variance_diversity.py BiasVarianceDiversity
"""


class BiasVarianceDiversity(Scene):
    def construct(self):
        # --- Titles ---
        title = Text("Bias-variance-diversity Decomposition",
                     color=BLUE).to_edge(UP)
        self.play(Write(title))

        # Set up Equations (using smaller font to fit screen)
        scale_val = 0.7

        bias_variance_decomposition = MathTex(
            r"\underbrace{\mathbb{E}_{\mathcal{S}}\left[\mathbb{E}_{\mathbf{x}y}\left[(y - ",
            # The term that will be changed for calculating Bias-variance-diversity Decomposition
            r"\hat{y}",
            r")^2\right]\right]}_{\text{expected risk}}",
            r" = ",
            r"\underbrace{\mathbb{E}_{\mathbf{x}y}\left[(y - \hat{y^*})^2\right]}_{\text{noise}}",
            r" + ",
            r"\underbrace{\mathbb{E}_{\mathbf{x}y}\left[\left(\hat{y^*} - \mathbb{E}_{\mathcal{S}}[\hat{y}]\right)^2\right]}_{\text{bias}}",
            r" + ",
            r"\underbrace{\mathbb{E}_{\mathbf{x}y}\left[\mathbb{E}_{\mathcal{S}}\left[(\hat{y} - \mathbb{E}_{\mathcal{S}}[\hat{y}])^2\right]\right]}_{\text{variance}}",
        ).scale(scale_val)

        # --- Start of animation ---
        self.play(
            Write(bias_variance_decomposition[:3]),
        )
        self.wait(5)

        # Equation after applying Ambiguity Decomposition
        after_ambiguity_decomposition = MathTex(
            r"\underbrace{\mathbb{E}_{\mathcal{S}}\left[\mathbb{E}_{\mathbf{x}y}\left[(y - ",
            r"\bar{y}",
            r")^2\right]\right]}_{\text{expected ensemble risk}}",
            r" = ",
            r"\underbrace{\frac{1}{m} \sum_{i=1}^{m} \mathbb{E}_{\mathcal{S}}\left[\mathbb{E}_{\mathbf{x}y}\left[(\hat{y}_i - y)^2\right]\right]}_{\text{average expected individual risk}}",
            r" - ",
            r"\underbrace{\frac{1}{m} \sum_{i=1}^{m} \mathbb{E}_{\mathcal{S}}\left[\mathbb{E}_{\mathbf{x}y}\left[(\hat{y}_i - \bar{y})^2\right]\right]}_{ambiguity}",
        ).scale(scale_val)

        # Final equation
        bias_variance_diversity_decomposition = MathTex(
            # Noise
            r"\underbrace{\mathbb{E}_{\mathbf{x}y}\left[(y - \hat{y^*})^2\right]}_{\text{noise}}",
            # Bias
            r"+ \underbrace{\frac{1}{m} \sum_{i=1}^{m} \mathbb{E}_{\mathbf{x}y}\left[\left(\hat{y^*} - \mathbb{E}_{\mathcal{S}}[\hat{y}_i]\right)^2\right]}_{\text{average bias}}",
            # Variance
            r"+ \underbrace{\frac{1}{m} \sum_{i=1}^{m} \mathbb{E}_{\mathbf{x}y}\left[\mathbb{E}_{\mathcal{S}}\left[(\hat{y}_i - \mathbb{E}_{\mathcal{S}}[\hat{y}_i])^2\right]\right]}_{\text{average variance}}",
            # Diversity
            r"\\\\ &\phantom{=}- \underbrace{\mathbb{E}_{\mathbf{x}y}\left[\mathbb{E}_{\mathcal{S}}\left[\frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - \bar{y})^2\right]\right]}_{\text{diversity}}"
        ).scale(0.6).arrange(RIGHT).shift(DOWN * 1.5)

        # Indicate what variables to change
        self.play(
            Indicate(
                bias_variance_decomposition[1],
                color=BLUE
            ),
            run_time=0.5
        )

        # Apply Ambiguity Decomposition
        self.play(
            # NOTE: A quite aggresive transformation to be honest
            ReplacementTransform(
                bias_variance_decomposition[:3], after_ambiguity_decomposition[:3]),
            run_time=1
        )
        self.wait(1)
        self.play(
            Indicate(
                after_ambiguity_decomposition[1],
                color=BLUE
            ),
            run_time=0.5
        )
        self.wait(1)
        self.play(
            Write(after_ambiguity_decomposition[3:]),
            run_time=1
        )
        self.wait(0.5)

        # Highlight average individual risk
        box = SurroundingRectangle(
            after_ambiguity_decomposition[4], color=YELLOW)
        self.play(
            Create(box),
            run_time=1
        )
        bias_variance_text = Text("Apply Bias-Variance Decomposition",
                                  font_size=24, color=YELLOW).next_to(box, UP)
        self.play(
            Write(bias_variance_text),
            run_time=0.5
        )
        self.wait(1)
        self.play(
            Unwrite(bias_variance_text),
            Uncreate(box),
            run_time=0.5
        )
        self.wait(0.5)

        # Apply Bias-variance Decomposition
        # Group the components
        top_row = VGroup(
            bias_variance_diversity_decomposition[0], bias_variance_diversity_decomposition[1], bias_variance_diversity_decomposition[2])
        bottom_row = bias_variance_diversity_decomposition[3]

        # For noise, average bias and average variance
        top_row_target = top_row.copy()
        # Place it to the right of the equals sign
        top_row_target.next_to(after_ambiguity_decomposition[:4].copy().move_to(
            UP * 1.5 + LEFT * 3.5), DOWN, buff=0.2, aligned_edge=LEFT)
        # Create indentation
        top_row_target.shift(RIGHT * 1.5)

        # For diversity
        bottom_row_target = bottom_row.copy()
        # Place it below the top row and align it slightly to the right (indented)
        bottom_row_target.next_to(
            top_row_target, DOWN, buff=0.2, aligned_edge=LEFT)

        self.play(
            after_ambiguity_decomposition[:4].animate.move_to(
                UP * 1.5 + LEFT * 3.5),
            # Move the Ambiguity term into the diversity part on the right
            ReplacementTransform(
                after_ambiguity_decomposition[5:], bottom_row_target),
            run_time=0.5
        )
        self.play(
            # Move the Average Loss into the 3 noise, bias and variance parts on the right
            ReplacementTransform(
                after_ambiguity_decomposition[4], top_row_target),
            run_time=1
        )
        self.wait(16.5)

        # End
        self.play(*[FadeOut(mob) for mob in self.mobjects])
