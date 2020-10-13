import numpy as np
import pytest
from healpix_util import DensityMap


@pytest.mark.parametrize('seed', [94282, 51339, 53912])
@pytest.mark.parametrize('scheme', ['ring', 'nest'])
@pytest.mark.parametrize('system', ['eq', 'ang', 'vec'])
def test_map_genrand(seed, scheme, system):

    rng = np.random.RandomState(seed)

    nside = 256
    map_data = np.ones(12*nside**2)
    hmap = DensityMap(scheme, map_data, rng=rng)

    num = 10000
    res = hmap.genrand(num, system=system)

    if system == 'vec':
        assert res.shape[0] == num
        assert res.shape[1] == 3
    else:
        for d in res:
            assert d.size == num

    if system == 'eq':
        ra, dec = res

        assert np.all((ra >= 0.0) & (ra <= 360.0))
        assert np.all((dec >= -90.0) & (dec <= 90.0))
    elif system == 'ang':
        theta, phi = res

        assert np.all((theta >= 0.0) & (theta <= np.pi))
        assert np.all((phi >= 0.0) & (phi <= 2*np.pi))
    elif system == 'vec':
        x, y, z = res[:, 0], res[:, 1], res[:, 2]
        assert np.all((x >= -1.0) & (x <= 1.0))
        assert np.all((y >= -1.0) & (y <= 1.0))
        assert np.all((z >= -1.0) & (z <= 1.0))


@pytest.mark.parametrize('seed', [66908, 28628, 80037])
@pytest.mark.parametrize('scheme', ['ring', 'nest'])
@pytest.mark.parametrize('ra_range', [(10, 20), (350, 359)])
@pytest.mark.parametrize('dec_range', [(-89, -80), (-5, 5)])
def test_randsphere_eq_range(seed, scheme, ra_range, dec_range):

    rng = np.random.RandomState(seed)

    nside = 256
    map_data = np.ones(12*nside**2)
    hmap = DensityMap(scheme, map_data, rng=rng)

    num = 10000
    res = hmap.genrand(
        num, system='eq',
        ra_range=ra_range,
        dec_range=dec_range,
    )

    for d in res:
        assert d.size == num

    ra, dec = res

    assert np.all(
        (ra >= ra_range[0]) & (ra <= ra_range[1]) &
        (dec >= dec_range[0]) & (dec <= dec_range[1])
    )


@pytest.mark.parametrize('seed', [92060, 17665, 38200])
@pytest.mark.parametrize('scheme', ['ring', 'nest'])
@pytest.mark.parametrize('system', ['eq', 'ang', 'vec'])
def test_randsphere_repeatable(seed, scheme, system):

    rng = np.random.RandomState(seed)

    num = 10000

    nside = 256
    map_data = np.ones(12*nside**2)

    hmap = DensityMap(scheme, map_data, rng=rng)
    res1 = hmap.genrand(num, system=system)

    rng = np.random.RandomState(seed)
    hmap = DensityMap(scheme, map_data, rng=rng)
    res2 = hmap.genrand(num, system=system)

    if system == 'vec':
        assert np.all(res1 == res2)
    else:
        for d1, d2 in zip(res1, res2):
            assert np.all(d1 == d2)
