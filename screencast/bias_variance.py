from config import DARTBOARD_CENTRE_COLOUR, DARTBOARD_COLOURS, DARTBOARD_RADII, DARTBOARD_DOT_COLOUR, DARTBOARD_DOT_STROKE_WIDTH, DARTBOARD_DOT_STROKE_COLOUR
from dartboard import Dartboard
from manim import *
import random


"""
Run:
manim -pql bias_variance.py BiasVariance
"""

# To enhance reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


class BiasVariance(Scene):
    def construct(self):
        # --- Titles ---
        title = Text("Bias-variance Decomposition", color=BLUE).to_edge(UP)
        self.play(Write(title))

        # Setup Equations (using smaller font to fit screen)
        scale_val = 0.7

        # Bias-variance Decomposition
        bias_variance_decomposition = MathTex(
            r"\underbrace{\mathbb{E}_{\mathcal{S}}\left[\mathbb{E}_{\mathbf{x}y}\left[(y - \hat{y})^2\right]\right]}_{\text{expected risk}}",
            r" = ",
            r"\underbrace{\mathbb{E}_{\mathbf{x}y}\left[(y - \hat{y^*})^2\right]}_{\text{noise}}",
            r" + ",
            r"\underbrace{\mathbb{E}_{\mathbf{x}y}\left[\left(\hat{y^*} - \mathbb{E}_{\mathcal{S}}[\hat{y}]\right)^2\right]}_{\text{bias}}",
            r" + ",
            r"\underbrace{\mathbb{E}_{\mathbf{x}y}\left[\mathbb{E}_{\mathcal{S}}\left[(\hat{y} - \mathbb{E}_{\mathcal{S}}[\hat{y}])^2\right]\right]}_{\text{variance}}",
        ).scale(scale_val)

        # --- Start of animation ---

        # Bias-variance Decomposition
        self.play(
            Write(bias_variance_decomposition),
        )
        self.wait(2)

        # Reset
        self.play(
            Unwrite(bias_variance_decomposition[:]),
            run_time=0.5
        )
        self.wait(0.5)

        # Dartboard settings

        dartboard_low_bias_high_variance_board_centre = (ORIGIN + 4.5 * LEFT)
        dartboard_high_bias_low_variance_board_centre = (ORIGIN + 1.5 * LEFT)
        dartboard_high_bias_high_variance_board_centre = (ORIGIN + 1.5 * RIGHT)
        dartboard_low_bias_low_variance_board_centre = (ORIGIN + 4.5 * RIGHT)
        dartboard_low_bias_high_variance_point_distribution_centre = dartboard_low_bias_high_variance_board_centre
        dartboard_high_bias_low_variance_point_distribution_centre = dartboard_high_bias_low_variance_board_centre + LEFT / 2.5 + UP / 2
        dartboard_high_bias_high_variance_point_distribution_centre = dartboard_high_bias_high_variance_board_centre + LEFT / 2.5 + UP / 2
        dartboard_low_bias_low_variance_point_distribution_centre = dartboard_low_bias_low_variance_board_centre

        # Create dartboards and their labels
        dartboard_low_bias_high_variance = Dartboard(
            ring_radii=DARTBOARD_RADII,
            ring_colours=DARTBOARD_COLOURS,
            bullseye_colour=DARTBOARD_CENTRE_COLOUR,
            point=dartboard_low_bias_high_variance_board_centre
        )

        dartboard_low_bias_high_variance_label = Text(
            "Low bias, high variance", font_size=16)
        dartboard_low_bias_high_variance_label.next_to(
            dartboard_low_bias_high_variance, DOWN, buff=0.2)

        dartboard_high_bias_low_variance = Dartboard(
            ring_radii=DARTBOARD_RADII,
            ring_colours=DARTBOARD_COLOURS,
            bullseye_colour=DARTBOARD_CENTRE_COLOUR,
            point=dartboard_high_bias_low_variance_board_centre
        )

        dartboard_high_bias_low_variance_label = Text(
            "High bias, low variance", font_size=16)
        dartboard_high_bias_low_variance_label.next_to(
            dartboard_high_bias_low_variance, DOWN, buff=0.2)

        dartboard_high_bias_high_variance = Dartboard(
            ring_radii=DARTBOARD_RADII,
            ring_colours=DARTBOARD_COLOURS,
            bullseye_colour=DARTBOARD_CENTRE_COLOUR,
            point=dartboard_high_bias_high_variance_board_centre
        )

        dartboard_high_bias_high_variance_label = Text(
            "High bias, high variance", font_size=16)
        dartboard_high_bias_high_variance_label.next_to(
            dartboard_high_bias_high_variance, DOWN, buff=0.2)

        dartboard_low_bias_low_variance = Dartboard(
            ring_radii=DARTBOARD_RADII,
            ring_colours=DARTBOARD_COLOURS,
            bullseye_colour=DARTBOARD_CENTRE_COLOUR,
            point=dartboard_low_bias_low_variance_board_centre
        )

        dartboard_low_bias_low_variance_label = Text(
            "Low bias, low variance", font_size=16)
        dartboard_low_bias_low_variance_label.next_to(
            dartboard_low_bias_low_variance, DOWN, buff=0.2)

        self.play(
            FadeIn(dartboard_low_bias_high_variance),
            FadeIn(dartboard_high_bias_low_variance),
            FadeIn(dartboard_high_bias_high_variance),
            FadeIn(dartboard_low_bias_low_variance),
            FadeIn(dartboard_low_bias_high_variance_label),
            FadeIn(dartboard_high_bias_low_variance_label),
            FadeIn(dartboard_high_bias_high_variance_label),
            FadeIn(dartboard_low_bias_low_variance_label),
            run_time=0.5
        )

        # Description to explain dots
        dot_legend = Dot(color=DARTBOARD_DOT_COLOUR, stroke_width=DARTBOARD_DOT_STROKE_WIDTH,
                         stroke_color=DARTBOARD_DOT_STROKE_COLOUR).move_to(ORIGIN + 3 * DOWN)
        dot_legend_label = Tex(f"1 prediction in a single model", font_size=18).next_to(
            dot_legend, RIGHT, buff=0.2)

        self.play(
            FadeIn(dot_legend),
            Write(dot_legend_label),
            run_time=0.5
        )

        # Create dartboard points
        # Number of dartboard points on each dartboard
        DARTBOARD_POINT_NUM = 20
        dartboard_low_bias_high_variance_points = []
        dartboard_high_bias_low_variance_points = []
        dartboard_high_bias_high_variance_points = []
        dartboard_low_bias_low_variance_points = []

        # Radius of the distribution of the points, i.e. maximum distance from its centre
        # NOTE: We need both horizonal and vertical axes
        NORMAL_DISTRIBUTION_RADIUS = (UP + LEFT) / 8
        # How much to scale the above radius if the model has a high variance
        NORMAL_DISTRIBUTION_RADIUS_FACTOR = 3

        # Create random dartboard points
        for _ in range(DARTBOARD_POINT_NUM):
            dartboard_low_bias_high_variance_points.append(Dot(color=DARTBOARD_DOT_COLOUR, stroke_width=DARTBOARD_DOT_STROKE_WIDTH, stroke_color=DARTBOARD_DOT_STROKE_COLOUR).move_to(dartboard_low_bias_high_variance_point_distribution_centre + LEFT * random.uniform(-NORMAL_DISTRIBUTION_RADIUS * NORMAL_DISTRIBUTION_RADIUS_FACTOR,
                                                           NORMAL_DISTRIBUTION_RADIUS * NORMAL_DISTRIBUTION_RADIUS_FACTOR) + UP * random.uniform(-NORMAL_DISTRIBUTION_RADIUS * NORMAL_DISTRIBUTION_RADIUS_FACTOR, NORMAL_DISTRIBUTION_RADIUS * NORMAL_DISTRIBUTION_RADIUS_FACTOR)))
            dartboard_high_bias_low_variance_points.append(Dot(color=DARTBOARD_DOT_COLOUR, stroke_width=DARTBOARD_DOT_STROKE_WIDTH, stroke_color=DARTBOARD_DOT_STROKE_COLOUR).move_to(dartboard_high_bias_low_variance_point_distribution_centre + LEFT / 4 + UP / 4 + LEFT *
                                                           random.uniform(-NORMAL_DISTRIBUTION_RADIUS, NORMAL_DISTRIBUTION_RADIUS) + UP * random.uniform(-NORMAL_DISTRIBUTION_RADIUS, NORMAL_DISTRIBUTION_RADIUS)))
            dartboard_high_bias_high_variance_points.append(Dot(color=DARTBOARD_DOT_COLOUR, stroke_width=DARTBOARD_DOT_STROKE_WIDTH, stroke_color=DARTBOARD_DOT_STROKE_COLOUR).move_to(dartboard_high_bias_high_variance_point_distribution_centre + LEFT * random.uniform(-NORMAL_DISTRIBUTION_RADIUS * NORMAL_DISTRIBUTION_RADIUS_FACTOR,
                                                            NORMAL_DISTRIBUTION_RADIUS * NORMAL_DISTRIBUTION_RADIUS_FACTOR) + UP * random.uniform(-NORMAL_DISTRIBUTION_RADIUS * NORMAL_DISTRIBUTION_RADIUS_FACTOR, NORMAL_DISTRIBUTION_RADIUS * NORMAL_DISTRIBUTION_RADIUS_FACTOR)))
            dartboard_low_bias_low_variance_points.append(Dot(color=DARTBOARD_DOT_COLOUR, stroke_width=DARTBOARD_DOT_STROKE_WIDTH, stroke_color=DARTBOARD_DOT_STROKE_COLOUR).move_to(dartboard_low_bias_low_variance_point_distribution_centre + LEFT / 4 + UP / 4 + LEFT *
                                                          random.uniform(-NORMAL_DISTRIBUTION_RADIUS, NORMAL_DISTRIBUTION_RADIUS) + UP * random.uniform(-NORMAL_DISTRIBUTION_RADIUS, NORMAL_DISTRIBUTION_RADIUS)))

            self.play(
                FadeIn(dartboard_low_bias_high_variance_points[-1]),
                FadeIn(dartboard_high_bias_low_variance_points[-1]),
                FadeIn(dartboard_high_bias_high_variance_points[-1]),
                FadeIn(dartboard_low_bias_low_variance_points[-1]),
                run_time=0.08
            )

        self.wait(1.5)

        # Reset
        self.play(
            FadeOut(dot_legend),
            Unwrite(dot_legend_label),
            FadeOut(*dartboard_low_bias_high_variance_points),
            FadeOut(*dartboard_high_bias_low_variance_points),
            FadeOut(*dartboard_high_bias_high_variance_points),
            FadeOut(*dartboard_low_bias_low_variance_points),
            FadeOut(dartboard_low_bias_high_variance),
            FadeOut(dartboard_high_bias_low_variance),
            FadeOut(dartboard_high_bias_high_variance),
            FadeOut(dartboard_low_bias_low_variance),
            FadeOut(dartboard_low_bias_high_variance_label),
            FadeOut(dartboard_high_bias_low_variance_label),
            FadeOut(dartboard_high_bias_high_variance_label),
            FadeOut(dartboard_low_bias_low_variance_label),
            run_time=0.5
        )

        # Use dartboard to explain noise
        # NOTE: We need to define here because Unwrite(...) (I think) removes the object bias_variance_decomposition from memory -> no text is shown
        bias_variance_decomposition = MathTex(
            r"\underbrace{\mathbb{E}_{\mathcal{S}}\left[\mathbb{E}_{\mathbf{x}y}\left[(y - \hat{y})^2\right]\right]}_{\text{expected risk}}",
            r" = ",
            r"\underbrace{\mathbb{E}_{\mathbf{x}y}\left[(y - \hat{y^*})^2\right]}_{\text{noise}}",
            r" + ",
            r"\underbrace{\mathbb{E}_{\mathbf{x}y}\left[\left(\hat{y^*} - \mathbb{E}_{\mathcal{S}}[\hat{y}]\right)^2\right]}_{\text{bias}}",
            r" + ",
            r"\underbrace{\mathbb{E}_{\mathbf{x}y}\left[\mathbb{E}_{\mathcal{S}}\left[(\hat{y} - \mathbb{E}_{\mathcal{S}}[\hat{y}])^2\right]\right]}_{\text{variance}}",
        ).scale(scale_val)
        bias_variance_decomposition.to_edge(DOWN)
        self.play(
            Write(bias_variance_decomposition),
            run_time=1
        )
        self.wait(1.5)

        # Add dartboard to show noise
        dartboard_noise = Dartboard(
            ring_radii=DARTBOARD_RADII,
            ring_colours=DARTBOARD_COLOURS,
            bullseye_colour=DARTBOARD_CENTRE_COLOUR,
            point=ORIGIN
        )

        self.play(
            FadeIn(dartboard_noise),
            # Initially, set all texts to translucent
            bias_variance_decomposition.animate.set_opacity(0.3),
            bias_variance_decomposition[2].animate.set_opacity(1.0),
        )

        for _ in range(DARTBOARD_POINT_NUM):
            dart = Dot(color=DARTBOARD_DOT_COLOUR, point=ORIGIN,
                       stroke_width=DARTBOARD_DOT_STROKE_WIDTH, stroke_color=DARTBOARD_DOT_STROKE_COLOUR)
            self.add(dart)

            self.play(
                dart.animate(),
                # Move board
                dartboard_noise.animate.shift(
                    np.random.uniform(-NORMAL_DISTRIBUTION_RADIUS, NORMAL_DISTRIBUTION_RADIUS, 3)),
                run_time=0.4,
                rate_func=linear
            )

            # Add the updater so it sticks to the board's movement
            current_offset = dart.get_center() - dartboard_noise.get_center()
            # NOTE: Must use the name "dt"
            dart.add_updater(lambda d, dt, offset=current_offset: d.move_to(
                dartboard_noise.get_center() + offset))

        self.wait(5)

        # End
        self.play(*[FadeOut(mob) for mob in self.mobjects])
