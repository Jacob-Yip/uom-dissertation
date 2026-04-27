from config import DARTBOARD_CENTRE_COLOUR, DARTBOARD_COLOURS, DARTBOARD_RADII, DARTBOARD_MODEL_COLOUR, DARTBOARD_MODEL_STROKE_WIDTH, DARTBOARD_MODEL_STROKE_COLOUR
from dartboard import Dartboard
from manim import *
import numpy as np
import random

"""
Run: 
manim -pql ncl.py NCL
"""

# To enhance reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def maximise_diversity(coordinates, iterations=1, k=0.01, step_size=0.1, move_ratio=1.5, manual=True):
    """
    :param: coordinates: In the form of [(x, y)]
    :param: iterations: How many steps to simulate
    :param: k: Repulsion strength
    :param: step_size: How much the dots move per iteration (learning rate)
    :param: move_ratio: For when manual=True, how much the points are moving from the origin (should be larger than 1)
    :param: manual: True to manually push dots
    """
    points = np.array(coordinates, dtype=float)
    n = len(points)
    for _ in range(iterations):
        forces = np.zeros_like(points)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                difference = points[i] - points[j]
                distance = np.linalg.norm(difference)
                distance = max(distance, 0.1)  # Avoid singularity
                # Push force
                forces[i] += (difference / distance) * (k / distance**2)
        points += forces * step_size
    return points


def minimise_accuracy(coordinates, move_ratio=0.7):
    points = np.array(coordinates, dtype=float)

    points *= move_ratio

    return points


class NCL(Scene):
    def construct(self):
        # --- Titles ---
        title = Text("Negative Correlation Learning",
                     color=BLUE).to_edge(UP)
        self.play(Write(title))

        # Setup Equations (using smaller font to fit screen)
        scale_val = 0.7

        ncl_loss = MathTex(
            r"\underbrace{\mathcal{L}_i}_{\text{ith base learner loss}}",
            r" = ",
            r"\underbrace{(\hat{y}_i - y)^2}_{\text{squared loss of 1 data point}}",
            r"- \lambda",
            r" \underbrace{(\hat{y}_i - \bar{y})^2}_{\text{correlation penalty function}}",
        ).scale(scale_val)

        # --- Start of animation ---
        self.wait(4)

        self.play(Write(ncl_loss[:3]))
        self.wait(1)
        self.play(Write(ncl_loss[3:]))
        self.wait(1)

        # Prepare for explanation

        self.play(
            ncl_loss.animate.move_to(UP * 2),
        )
        self.wait(1)

        # Target dot
        dartboard_target = Dartboard(
            ring_radii=DARTBOARD_RADII,
            ring_colours=DARTBOARD_COLOURS,
            bullseye_colour=DARTBOARD_CENTRE_COLOUR,
            point=ORIGIN + DOWN
        )

        self.play(
            FadeIn(dartboard_target),
        )
        self.wait(0.5)

        learner_num = 50
        # For both LEFT_SCALE and UP_SCALE
        MIN_SCALE = -0.8
        MAX_SCALE = 0.8
        # In the form of [(LEFT_SCALE, UP_SCALE)]
        # LEFT_SCALE and UP_SCALE are [-1, 1]
        original_learner_coordinates = []
        # In the form of [(LEFT_SCALE, UP_SCALE)]
        # LEFT_SCALE and UP_SCALE are [-1, 1]
        repulsed_learner_coordinates = []
        learner_dots = []

        # Create learners
        for _ in range(learner_num):
            original_learner_coordinates.append(
                (random.uniform(MIN_SCALE, MAX_SCALE), random.uniform(MIN_SCALE, MAX_SCALE)))
            learner_dots.append(Dot(color=DARTBOARD_MODEL_COLOUR, stroke_width=DARTBOARD_MODEL_STROKE_WIDTH, stroke_color=DARTBOARD_MODEL_STROKE_COLOUR).shift(
                LEFT * original_learner_coordinates[-1][0] + UP * original_learner_coordinates[-1][1] + DOWN))
        learners = VGroup(*learner_dots)

        # Description to explain dots
        learner_legend = Dot(color=DARTBOARD_MODEL_COLOUR, stroke_width=DARTBOARD_MODEL_STROKE_WIDTH,
                             stroke_color=DARTBOARD_MODEL_STROKE_COLOUR).move_to(ORIGIN + 3 * DOWN)
        learer_legend_label = Tex(f"1 model in an ensemble model", font_size=18).next_to(
            learner_legend, RIGHT, buff=0.2)

        self.play(
            FadeIn(learner_legend),
            Write(learer_legend_label),
            FadeIn(learners),
            run_time=0.5
        )
        self.wait(0.5)

        # Minimising loss through gradient descent
        self.play(
            ncl_loss[0].animate.scale(0.8),
            run_time=1
        )
        self.wait(1)

        # Minimising individual accuracy
        more_accurate_learner_coordinates = minimise_accuracy(
            original_learner_coordinates).tolist()
        more_accurate_learners = [
            learners[i].animate.move_to(LEFT * more_accurate_learner_coordinates[i][0] + UP * more_accurate_learner_coordinates[i][1] + DOWN) for i in range(len(learners))
        ]
        self.play(
            *more_accurate_learners,
            ncl_loss[2].animate.scale(0.8),
            run_time=1
        )
        self.wait(0.5)

        # Reset
        self.play(
            ncl_loss[0].animate.scale(1.25),
            ncl_loss[2].animate.scale(1.25),
            FadeOut(learners),
            run_time=0.5
        )
        self.wait(0.5)
        for learner_index in range(len(learners)):
            learners[learner_index].animate.move_to(
                LEFT * original_learner_coordinates[learner_index][0] + UP * original_learner_coordinates[learner_index][1] + DOWN)
        self.play(
            FadeIn(learners),
            run_time=1
        )

        # Maximising collective diversity
        repulsed_learner_coordinates = maximise_diversity(
            original_learner_coordinates, manual=False).tolist()
        repulsion = [
            learners[i].animate.move_to(LEFT * repulsed_learner_coordinates[i][0] + UP * repulsed_learner_coordinates[i][1] + DOWN) for i in range(len(learners))
        ]

        # Shrink individual loss
        self.play(
            ncl_loss[0].animate.scale(0.8),
            run_time=1
        )
        self.wait(0.5)

        self.play(
            *repulsion,
            ncl_loss[4].animate.scale(1.25),
            run_time=1
        )
        self.wait(9.5)

        # End
        self.play(*[FadeOut(mob) for mob in self.mobjects])
