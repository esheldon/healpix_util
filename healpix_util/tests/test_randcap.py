import numpy as np
import pytest
from healpix_util.coords import randcap, sphdist


@pytest.mark.parametrize('seed', [58205, 80258, 64376])
def test_randcap_eq(seed):

    rng = np.random.RandomState(seed)

    num = 10000
    ra_cen = 235.0
    dec_cen = 85.0
    radius = 1.0

    ra, dec = randcap(num, ra_cen, dec_cen, radius, system='eq', rng=rng)

    assert ra.size == num
    assert dec.size == num

    dist = sphdist(ra_cen, dec_cen, ra, dec)
    assert np.all(dist <= radius)


@pytest.mark.parametrize('seed', [58205, 80258, 64376])
def test_randcap_eq_repeatable(seed):

    rng = np.random.RandomState(seed)

    num = 10000
    ra_cen = 235.0
    dec_cen = 85.0
    radius = 1.0

    ra, dec = randcap(num, ra_cen, dec_cen, radius, system='eq', rng=rng)

    rng = np.random.RandomState(seed)
    ratest, dectest = randcap(num, ra_cen, dec_cen, radius, system='eq', rng=rng)

    assert np.all(ra == ratest)
    assert np.all(dec == dectest)
