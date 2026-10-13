"""Independent A14 coverage verifier using frozen A13 verifier primitives.

Never imports joint_kernel or cert_kernel. It recomputes the camera partition
and selected exact physical rows from the public observation, then rechecks all
four rational box contradictions. Standard library only.
"""
from fractions import Fraction as Q
import importlib.util
from pathlib import Path

A13_ROOT = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-certified-boundary-20261013T093552Z')
_spec = importlib.util.spec_from_file_location('_a14_independent_a13_verifier', A13_ROOT/'verify_certificates.py')
old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(old)
if Path(old.__file__).resolve() != (A13_ROOT/'verify_certificates.py').resolve():
    raise RuntimeError('wrong independent verifier identity')

SCHEMA = 'a14-four-camera-cover-v1'
TILE_SCHEMA = 'a14-camera-tile-box-farkas-v1'


def _encode(box):
    return [[old.fraction_text(lo), old.fraction_text(hi)] for lo, hi in box]


def _tiles(cameras):
    # Independently derive the exact midpoints and four Cartesian products.
    a, b = cameras[0]
    c, d = cameras[2]
    left = ((a, (a+b)/2), ((a+b)/2, b))
    right = ((c, (c+d)/2), ((c+d)/2, d))
    return tuple((left[i//2], (Q(0), Q(0)), right[i % 2]) for i in range(4))


def _rows(canonical, branch, lo, hi, camera_box, support):
    _cameras, photo, boxes, fixed = old._decode_observation(canonical)
    selected = {i: fixed[i] for i in support if i < len(fixed)}
    count = len(fixed)
    for v in range(3):
        geometry, areas = None, {}
        for side in range(2):
            for pixel_index in range(48):
                bound = photo[v][pixel_index][side]
                if bound is None:
                    continue
                if count in support:
                    if geometry is None:
                        geometry = old._view_geometry(branch, lo, hi, camera_box[v])
                    if pixel_index not in areas:
                        areas[pixel_index] = old._pixel_geometry(geometry, old.PIXELS[pixel_index])
                    sign = Q(1) if side == 0 else Q(-1)
                    row = tuple((3*v+k, sign*a) for k, a in enumerate(areas[pixel_index][side]) if a)
                    selected[count] = (row+((9+v, -sign),), sign*(1-bound))
                count += 1
    return selected, boxes, count


def verify_certificate(observation, record):
    try:
        wrapper = record if isinstance(record, dict) and 'certificate' in record else None
        cert = record['certificate'] if wrapper is not None else record
        if not isinstance(cert, dict) or cert.get('schema') != SCHEMA:
            raise ValueError('unsupported four-tile schema')
        canonical = old._canonical_observation(observation)
        digest = old.observation_sha256(observation)
        if cert.get('observation_sha256') != digest:
            raise ValueError('observation digest mismatch')
        branch, lo, hi = cert['branch'], old._fraction(cert['lo']), old._fraction(cert['hi'])
        if branch not in old.BRANCHES or not 540 <= lo <= hi <= 660:
            raise ValueError('invalid depth cell')
        if wrapper is not None and (wrapper.get('branch') != branch
                or old._endpoint(wrapper['lower']) != lo or old._endpoint(wrapper['upper']) != hi):
            raise ValueError('wrapper mismatch')
        cameras, _photo, _boxes, _fixed = old._decode_observation(canonical)
        if cert.get('camera_domain') != _encode(cameras):
            raise ValueError('camera domain mismatch')
        expected = _tiles(cameras)
        proofs = cert.get('tiles')
        if not isinstance(proofs, list) or len(proofs) != 4:
            raise ValueError('all four tile proofs required')
        margins = []
        for tile_id, (proof, box) in enumerate(zip(proofs, expected)):
            if not isinstance(proof, dict) or proof.get('schema') != TILE_SCHEMA:
                raise ValueError('invalid tile schema')
            if (type(proof.get('tile_id')) is not int or proof['tile_id'] != tile_id
                    or proof.get('camera_box') != _encode(box)):
                raise ValueError('tile identity or exact midpoint coverage mismatch')
            if (proof.get('observation_sha256') != digest or proof.get('branch') != branch
                    or proof.get('lo') != cert['lo'] or proof.get('hi') != cert['hi']
                    or proof.get('variable_order') != old.VARIABLE_ORDER):
                raise ValueError('tile proof target mismatch')
            weights = {}
            terms = proof.get('multipliers')
            if not isinstance(terms, list) or not terms:
                raise ValueError('empty multipliers')
            for term in terms:
                if not isinstance(term, list) or len(term) != 2:
                    raise ValueError('malformed multiplier')
                index, weight = term
                if type(index) is not int or index < 0 or index in weights:
                    raise ValueError('invalid multiplier index')
                weight = old._fraction(weight)
                if weight <= 0:
                    raise ValueError('nonpositive multiplier')
                weights[index] = weight
            rows, bounds, count = _rows(canonical, branch, lo, hi, box, set(weights))
            if type(proof.get('row_count')) is not int or proof['row_count'] != count or set(rows) != set(weights):
                raise ValueError('row layout mismatch')
            combined, rhs = [Q(0)]*15, Q(0)
            for index, weight in weights.items():
                row, upper = rows[index]
                rhs += weight*upper
                for variable, coefficient in row:
                    combined[variable] += weight*coefficient
            margin = sum((value*bounds[j][0 if value >= 0 else 1]
                          for j, value in enumerate(combined)), Q(0))-rhs
            if margin <= 0 or old._fraction(proof['margin']) != margin:
                raise ValueError('not a strict exact contradiction')
            margins.append(old.fraction_text(margin))
        return {'verified': True, 'branch': branch, 'lo': cert['lo'], 'hi': cert['hi'],
                'tile_count': 4, 'margins': margins}
    except (ValueError, KeyError, TypeError, ArithmeticError, OverflowError) as exc:
        return {'verified': False, 'error': f'{type(exc).__name__}: {exc}'}
