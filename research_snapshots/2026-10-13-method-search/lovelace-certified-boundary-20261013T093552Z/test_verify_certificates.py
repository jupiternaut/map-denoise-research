"""Independent checker tests; no solver or builder imports and no truth files."""
from copy import deepcopy
from fractions import Fraction as Q
import json
import math
import random
import time
import unittest

import verify_certificates as verifier


def blank_observation():
    return {"photo": [1.0] * 432,
            "camera": {"u": [.08, 0.0, -.08], "Xf": 0.0, "Zf": 1000.0,
                       "survey_bound": .01, "localization_bound": .01},
            "appearance": {"coupons": [1.0] * 9, "references": [.5] * 3,
                           "bias_bound": .01, "transfer_bound": .005, "noise_bound": .005}}


def full_state():
    return {"lower": 540.0, "upper": 660.0,
            "cells": [{"branch": branch, "lower": 540.0, "upper": 660.0}
                      for branch in verifier.BRANCHES], "exclusions": []}


def dark_pixel_proof(low=600, high=600):
    o = blank_observation()
    o["appearance"]["coupons"] = [.5] * 9
    o["photo"][144 + 3 * 19:144 + 3 * 19 + 3] = [0.0] * 3
    rows, box = verifier.reconstruct_constraints(o, "none", low, high)
    index = next(i for i, (a, b) in enumerate(rows) if dict(a).get(10) == 1 and b < Q(-9, 10))
    a, b = rows[index]
    margin = sum(coefficient * box[j][0 if coefficient >= 0 else 1] for j, coefficient in a) - b
    cert = {"schema": verifier.SCHEMA, "observation_sha256": verifier.observation_sha256(o),
            "branch": "none", "lo": verifier.fraction_text(low), "hi": verifier.fraction_text(high),
            "variable_order": verifier.VARIABLE_ORDER.copy(), "row_count": len(rows),
            "multipliers": [[index, "1/1"]], "margin": verifier.fraction_text(margin)}
    return o, cert


def exact_world(z, branch, cameras, colors, photo_bias=None, appearance_bias=None):
    """Point-world renderer from direct overlap widths, independent of boxes."""
    photo_bias = photo_bias or [Q(0)] * 3
    appearance_bias = appearance_bias or [Q(0)] * 3
    def rectangle_pixels(rect, depth, cam):
        if rect is None:
            return [Q(0)] * 48
        x0, x1, y0, y1 = [Q(x) for x in rect]
        x0, x1, y0, y1 = (x0-cam)/depth, (x1-cam)/depth, y0/depth, y1/depth
        result = []
        for j in range(6):
            for i in range(8):
                p0, p1 = Q(i-4, 20), Q(i-3, 20)
                p2, p3 = Q(2*j-6, 75), Q(2*j-4, 75)
                result.append(max(Q(0), min(x1, p1)-max(x0, p0)) *
                              max(Q(0), min(y1, p3)-max(y0, p2)) * 750)
        return result
    o = blank_observation()
    photo = []
    for v, camera in enumerate(cameras):
        left = rectangle_pixels((-20, 0, -15, 15), z, camera)
        right = rectangle_pixels((0, 20, -15, 15), z, camera)
        rear = {"none": None, "partial": (-20, 36, -17, 23),
                "full": (-100, 100, -80, 80)}[branch]
        r = rectangle_pixels(rear, Q(720), camera)
        # Compute front/rear projection overlap in image coordinates directly.
        front_image = ((Q(-20)-camera)/z, (Q(20)-camera)/z, Q(-15)/z, Q(15)/z)
        if rear is None:
            overlap = [Q(0)] * 48
        else:
            rear_image = ((Q(rear[0])-camera)/720, (Q(rear[1])-camera)/720,
                          Q(rear[2], 720), Q(rear[3], 720))
            common = (max(front_image[0], rear_image[0]), min(front_image[1], rear_image[1]),
                      max(front_image[2], rear_image[2]), min(front_image[3], rear_image[3]))
            overlap = rectangle_pixels(common, Q(1), Q(0))
        for p in range(48):
            intensity = 1 - left[p]*(1-colors[3*v]) - right[p]*(1-colors[3*v+1]) - \
                        (r[p]-overlap[p])*(1-colors[3*v+2])
            photo.extend([float(max(Q(0), min(Q(1), intensity + photo_bias[v])))] * 3)
    o["photo"] = photo
    o["camera"]["u"] = [float(-camera/1000) for camera in cameras]
    o["appearance"]["coupons"] = [float(max(Q(0), min(Q(1), c + appearance_bias[i//3])))
                                    for i, c in enumerate(colors)]
    o["appearance"]["references"] = [float(Q(1, 2) + b) for b in appearance_bias]
    return o, [1-c for c in colors] + photo_bias + appearance_bias


class CertificateTests(unittest.TestCase):
    def test_rounding_preimages_enclose_exact_source(self):
        for exact in [Q(0), Q(1), Q(1, 10), Q(-8, 100), Q(17, 333), Q(1, 10**60)]:
            lo, hi = verifier._rounding_interval(float(exact))
            self.assertLessEqual(lo, exact)
            self.assertGreaterEqual(hi, exact)
        lo, hi = verifier._rounding_interval(.1)
        self.assertLess(lo, Q(1, 10))
        self.assertGreater(hi, Q(1, 10))
        self.assertNotEqual(Q(.1), Q(1, 10))

    def test_known_single_row_exact_margin(self):
        o, cert = dark_pixel_proof()
        expected = Q(71, 200) - Q(1, 2**1075)
        self.assertEqual(Q(cert["margin"]), expected)
        result = verifier.verify_certificate(o, cert)
        self.assertTrue(result["valid"], result)
        self.assertEqual(Q(result["margin"]), expected)

    def test_deliberate_certificate_corruption(self):
        o, cert = dark_pixel_proof()
        mutations = [lambda c: c.update(margin="1/1"),
                     lambda c: c["multipliers"][0].__setitem__(1, "-1/1"),
                     lambda c: c["multipliers"][0].__setitem__(1, "2/2"),
                     lambda c: c["multipliers"].append(c["multipliers"][0].copy()),
                     lambda c: c.update(row_count=c["row_count"]+1),
                     lambda c: c["variable_order"].reverse(),
                     lambda c: c.update(branch="full"),
                     lambda c: c.update(lo="660/1", hi="660/1"),
                     lambda c: c["multipliers"][0].__setitem__(0, 10000)]
        for mutate in mutations:
            changed = deepcopy(cert)
            mutate(changed)
            self.assertFalse(verifier.verify_certificate(o, changed)["valid"])
        swapped = deepcopy(o)
        swapped["photo"][0] = .9
        self.assertFalse(verifier.verify_certificate(swapped, cert)["valid"])
        wrapper = {"branch": "partial", "lower": 600.0, "upper": 600.0, "certificate": cert}
        self.assertFalse(verifier.verify_certificate(o, wrapper)["valid"])

    def test_proof_can_transfer_when_still_mathematically_valid(self):
        o, cert = dark_pixel_proof()
        swapped = deepcopy(o)
        swapped["appearance"]["coupons"][0] = .6
        # This changes unused coupon rows but no row layout or supported row.
        cert["observation_sha256"] = verifier.observation_sha256(swapped)
        self.assertTrue(verifier.verify_certificate(swapped, cert)["valid"])

    def test_metadata_and_malformed_input_rejected(self):
        o, cert = dark_pixel_proof()
        for key, value in [("Zf", 999), ("survey_bound", .02)]:
            bad = deepcopy(o)
            bad["camera"][key] = value
            changed = deepcopy(cert)
            changed["observation_sha256"] = verifier.observation_sha256(bad)
            self.assertFalse(verifier.verify_certificate(bad, changed)["valid"])
        for bad in [None, {}, {"photo": [math.nan], "camera": {}, "appearance": {}}]:
            self.assertFalse(verifier.verify_certificate(bad, cert)["valid"])

    def test_exact_feasible_worlds_satisfy_every_necessary_row(self):
        rng = random.Random(20261013)
        for branch in verifier.BRANCHES:
            for z in (Q(540), Q(600), Q(660), Q(60123, 100)):
                cameras = [Q(-80) + Q(rng.randint(-180, 180), 1000), Q(0),
                           Q(80) + Q(rng.randint(-180, 180), 1000)]
                colors = [Q(1, 5), Q(4, 5), Q(2, 5)] * 3
                o, point = exact_world(z, branch, cameras, colors)
                for low, high in [(Q(540), Q(660)), (z, z)]:
                    rows, boxes = verifier.reconstruct_constraints(o, branch, low, high)
                    self.assertTrue(all(a <= x <= b for x, (a, b) in zip(point, boxes)))
                    for row, rhs in rows:
                        self.assertLessEqual(sum(a * point[j] for j, a in row), rhs)
                    # Any nonnegative aggregation is unable to contradict this point.
                    weights = [Q(rng.randrange(4)) for _ in rows]
                    coeff, rhs = [Q(0)] * 15, Q(0)
                    for (row, b), weight in zip(rows, weights):
                        rhs += weight*b
                        for j, a in row:
                            coeff[j] += weight*a
                    minimum = sum(a * (box[0] if a >= 0 else box[1]) for a, box in zip(coeff, boxes))
                    self.assertLessEqual(minimum-rhs, 0)

    def test_partition_gaps_overlap_binding_and_fallback(self):
        o = blank_observation()
        state = full_state()
        self.assertTrue(verifier.verify_state(o, state)["valid"])
        state["cells"][0]["upper"] = 600.0
        self.assertFalse(verifier.verify_state(o, state)["valid"])
        state["cells"].append({"branch": "none", "lower": 600.0, "upper": 660.0})
        self.assertTrue(verifier.verify_state(o, state)["valid"])
        state["cells"][-1]["lower"] = math.nextafter(600.0, math.inf)
        self.assertFalse(verifier.verify_state(o, state)["valid"])
        state["cells"][-1]["lower"] = math.nextafter(600.0, -math.inf)
        self.assertFalse(verifier.verify_state(o, state)["valid"])
        o, cert = dark_pixel_proof(540, 600)
        state = full_state()
        state["cells"][0]["lower"] = 600.0
        exclusion = {"branch": "none", "lower": 540.0, "upper": 600.0,
                     "verified": True, "certificate": cert}
        state["exclusions"] = [exclusion]
        self.assertTrue(verifier.verify_state(o, state)["valid"])
        state["upper"] = 659.0
        self.assertFalse(verifier.verify_state(o, state)["valid"])
        state["upper"] = 660.0
        state["cells"][0]["lower"] = 540.0
        self.assertFalse(verifier.verify_state(o, state)["valid"])
        state["fallback_reason"] = "encoding_or_solver_error"
        result = verifier.verify_state(o, state)
        self.assertTrue(result["valid"], result)
        self.assertFalse(result["partitioned"])

    def test_saturated_known_feasible_all_white(self):
        o = blank_observation()
        point = [Q(0)] * 15
        for branch in verifier.BRANCHES:
            rows, boxes = verifier.reconstruct_constraints(o, branch, 540, 660)
            self.assertTrue(all(sum(a*point[j] for j, a in row) <= rhs for row, rhs in rows))
            self.assertTrue(all(a <= x <= b for x, (a, b) in zip(point, boxes)))

    def test_photo_and_appearance_biases_remain_independent(self):
        for branch in verifier.BRANCHES:
            o, point = exact_world(Q(61325, 100), branch, [Q(-80), Q(0), Q(80)],
                                   [Q(1, 5), Q(4, 5), Q(2, 5)] * 3,
                                   [Q(7, 1000), Q(-9, 1000), Q(3, 1000)],
                                   [Q(-8, 1000), Q(9, 1000), Q(-2, 1000)])
            rows, boxes = verifier.reconstruct_constraints(o, branch, 613, 614)
            self.assertTrue(all(sum(a*point[j] for j, a in row) <= rhs for row, rhs in rows))
            self.assertTrue(all(a <= x <= b for x, (a, b) in zip(point, boxes)))


if __name__ == "__main__":
    start = time.process_time()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CertificateTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({"cpu_seconds": time.process_time()-start, "tests": result.testsRun,
                      "successful": result.wasSuccessful()}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
