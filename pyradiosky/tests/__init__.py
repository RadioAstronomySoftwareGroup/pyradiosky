import pytest

# define a pytest marker to skip astropy_healpix tests
try:
    import astropy_healpix  # noqa

    healpix_installed = True
except(ImportError):
    healpix_installed = False
reason = 'astropy_healpix is not installed, skipping tests that require it.'
skipIf_no_healpix = pytest.mark.skipif(not healpix_installed, reason=reason)
