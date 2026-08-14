from aurora_analysis.match.naming import (
    format_match_stem,
    match_catalog_tag,
    parse_match_catalog_tag,
)


def test_ball_method_is_labeled_sph():
    assert match_catalog_tag("ball") == "sph"
    assert match_catalog_tag("knn") == "knn"


def test_stem_embeds_sph_for_sphere_matches():
    assert (
        format_match_stem(ztag="0.300", method="ball", source="maplog", suffix="fullbox")
        == "match_maplog_sph_z0.300_fullbox"
    )
    assert (
        format_match_stem(ztag="0.300", method="ball", source="subsample", suffix="rank000")
        == "match_sph_z0.300_rank000"
    )
    assert (
        format_match_stem(ztag="0.800", method="knn", source="subsample", suffix="fullbox")
        == "match_knn_z0.800_fullbox"
    )


def test_parse_recovers_tag_from_filename():
    assert parse_match_catalog_tag("match_maplog_sph_z0.300_fullbox.npz") == "sph"
    assert parse_match_catalog_tag("match_knn_z0.300_rank000.npz") == "knn"
    assert parse_match_catalog_tag("match_z0.300_fullbox.npz") is None
