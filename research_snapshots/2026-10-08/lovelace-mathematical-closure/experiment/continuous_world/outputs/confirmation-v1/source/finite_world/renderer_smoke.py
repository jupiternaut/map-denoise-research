"""Small exact-arithmetic checks; does not write results or run reconstruction."""

import json
import unittest
from fractions import Fraction

from renderer import (Q, Camera, ImageGrid, Material, Surface, World,
                      DEFAULT_CAMERAS, demo_worlds, evaluation_target_z, render,
                      render_image)


class RendererSmoke(unittest.TestCase):
    def test_solid_and_exact_types(self):
        color = (Q(1, 3), Q(1, 2), Q(2, 3))
        world = World("solid", [Surface("6", -10, 10, -10, 10, Material(color0=color))])
        pixels = render(world)
        self.assertEqual(len(pixels), 3 * 8 * 6 * 3)
        self.assertTrue(all(isinstance(value, Fraction) for value in pixels))
        self.assertEqual(pixels, color * (3 * 8 * 6))
        self.assertEqual(evaluation_target_z(world), Q(6))

    def test_partial_pixel_area(self):
        grid = ImageGrid(1, 1, 0, 1, 0, 1)
        world = World("half_black", (Surface(1, 0, Q(1, 2), 0, 1, Material()),))
        self.assertEqual(render_image(world, Camera(0), grid), (Q(1, 2),) * 3)
        # Uneven rational coverage: 2/3 black and 1/3 white.
        world = World("two_thirds", (Surface(1, 0, Q(2, 3), 0, 1, Material()),))
        self.assertEqual(render_image(world, Camera(0), grid), (Q(1, 3),) * 3)

    def test_front_occludes_rear(self):
        grid = ImageGrid(1, 1, 0, 1, 0, 1)
        red, blue = Material(color0=(1, 0, 0)), Material(color0=(0, 0, 1))
        front = Surface(1, 0, Q(1, 2), 0, 1, red, "front")
        rear = Surface(2, 0, 2, 0, 2, blue, "rear")
        world = World("occlusion", (rear, front))
        self.assertEqual(render_image(world, Camera(0), grid), (Q(1, 2), 0, Q(1, 2)))
        self.assertEqual(evaluation_target_z(world), 1)

    def test_checker_exact_integral_and_world_coordinates(self):
        material = Material("checker", (0, 0, 0), (1, 1, 1), 1)
        world = World("checker", (Surface(2, -10, 10, -10, 10, material),))
        grid = ImageGrid(1, 1, 0, Q(1, 2), 0, Q(1, 2))
        self.assertEqual(render_image(world, Camera(0), grid), (0, 0, 0))
        self.assertEqual(render_image(world, Camera(1), grid), (1, 1, 1))
        large_grid = ImageGrid(1, 1, 0, 1, 0, 1)
        self.assertEqual(render_image(world, Camera(0), large_grid), (Q(1, 2),) * 3)
        # Negative tiles follow floor, with no float-based truncation toward zero.
        self.assertEqual(material.sample(Q(-1, 2), Q(1, 2)), (1, 1, 1))

    def test_subdivision_invariance(self):
        checker = Material("checker", (Q(1, 4), 0, 0), (0, Q(3, 4), 1), Q(2, 3))
        world = World("mixed", (Surface(3, Q(-2, 3), Q(5, 3), -1, 1, checker),
                                Surface(5, -4, 4, -3, 3, Material(color0=(0, 0, 1)))))
        camera = Camera(Q(1, 3))
        coarse = render_image(world, camera, ImageGrid(1, 1))
        fine = render_image(world, camera, ImageGrid(4, 3))
        mean = tuple(sum(fine[c::3], Q(0)) / 12 for c in range(3))
        self.assertEqual(coarse, mean)

    def test_same_color_ambiguity(self):
        examples = demo_worlds()
        self.assertNotEqual(evaluation_target_z(examples[-2]), evaluation_target_z(examples[-1]))
        self.assertEqual(render(examples[-2]), render(examples[-1]))
        self.assertNotEqual(render(examples[0]), render(examples[1]))
        self.assertEqual(evaluation_target_z(examples[2]), 6)

    def test_serialization_and_background(self):
        world = World("outside", (Surface(2, 10, 11, 10, 11, Material()),), (0, Q(1, 3), 1))
        self.assertEqual(render_image(world, Camera(0)), (0, Q(1, 3), 1) * 48)
        self.assertEqual(json.loads(json.dumps(world.to_dict()))["background"], ["0", "1/3", "1"])
        self.assertEqual(DEFAULT_CAMERAS[1].to_dict()["center"], ["0", "0", "0"])
        self.assertEqual(ImageGrid().to_dict()["width"], 8)
        with self.assertRaises(ValueError):
            evaluation_target_z(world)

    def test_invalid_inputs(self):
        for factory in (lambda: Camera(0.1), lambda: ImageGrid(u_min=-0.5),
                        lambda: Material(color0=(0.1, 0, 0)),
                        lambda: Surface(1.0, -1, 1, -1, 1, Material()),
                        lambda: World("bad", (Surface(1, -1, 1, -1, 1, Material()),), (0, 0, 1.0))):
            with self.assertRaises(TypeError):
                factory()
        for factory in (lambda: Material(tile_size=0), lambda: Material(color0=(-1, 0, 0)),
                        lambda: ImageGrid(width=True), lambda: Surface(0, -1, 1, -1, 1, Material()),
                        lambda: World("coplanar", (Surface(1, -1, 1, -1, 1, Material()),
                                                   Surface(1, -2, 2, -2, 2, Material())))):
            with self.assertRaises(ValueError):
                factory()


if __name__ == "__main__":
    unittest.main(verbosity=2)
