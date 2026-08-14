"""Clustering and summary statistics. Not yet populated -- later milestone.

Planned modules, and the code they consolidate:

``abundance``
    ``get_threshold``: the mass or multiplicity cut giving a target number
    density. Every MAP-versus-observation comparison goes through this, so it
    should have one implementation and a test.
``corr``
    ``xi``, ``wp`` and redshift-space multipoles over pycorr, gathering
    ``xi_from_pos``, ``xi_abundance_cut`` and the several ad-hoc wrappers in
    ``extend_utilities.py``.
``pk``
    Monopole P(k) from ``aurora_pk_monopole.py`` / ``summit_pk_monopole.py``,
    which are near-duplicates of one another.
``binning``
    Running medians and binned statistics -- ``_running_median`` alone has
    three copies.
"""
