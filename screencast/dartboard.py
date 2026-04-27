from manim import *

"""
The entity representing a dard board
"""


class Dartboard(VGroup):
    def __init__(self, ring_radii: list = [], ring_colours: list = [], bullseye_colour=None, point=ORIGIN) -> None:
        super().__init__()

        self.radii = ring_radii
        self.colours = ring_colours

        assert len(self.radii) == len(
            self.colours), f"Invalid lengths of radii and colours: {len(ring_radii)} != {len(ring_colours)}"

        for radius, colour in zip(self.radii, self.colours):
            ring = Circle(radius=radius, stroke_width=0, fill_opacity=1)
            ring.set_fill(colour)
            ring.move_to(point)

            self.add(ring)

        assert bullseye_colour is not None, f"Invalid colour of bullseye: {bullseye_colour}"

        # Make bullseye bigger
        bullseye = Dot(point=point, color=bullseye_colour).scale(4)
        self.add(bullseye)
