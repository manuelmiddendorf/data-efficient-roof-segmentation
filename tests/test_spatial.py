from shapely.geometry import box

from roofseg.data import check_role_partition
from roofseg.spatial import (overlapping_pairs, remove_crossing_training_images,
                             select_nested_spatial_subsets, select_uniform_random_subset)


def test_cross_role_overlap_is_removed():
    footprints = {"1": box(0, 0, 2, 2), "2": box(3, 0, 5, 2), "3": box(1, 0, 3, 2)}
    assert [(row["left"], row["right"]) for row in overlapping_pairs(footprints, ["1", "2"], ["3"])] == [("1", "3")]
    retained, excluded = remove_crossing_training_images(footprints, ["1", "2"], {"validation": ["3"]})
    assert retained == ["2"]
    assert excluded == [{"id": "1", "reason": "footprint_intersects_validation"}]


def test_nested_subsets_are_deterministic_and_nested():
    footprints = {str(index): box(index, 0, index + 0.5, 0.5) for index in range(1, 11)}
    first = select_nested_spatial_subsets(footprints, footprints, [2, 5, 8], seed=17)
    second = select_nested_spatial_subsets(footprints, footprints, [2, 5, 8], seed=17)
    assert first == second
    assert set(first["2"]) < set(first["5"]) < set(first["8"])


def test_uniform_random_subset_is_deterministic_unique_and_order_independent():
    pool = [str(index) for index in range(150)]
    first = select_uniform_random_subset(reversed(pool), 100, seed=29)
    second = select_uniform_random_subset(pool, 100, seed=29)
    assert first == second
    assert len(first) == len(set(first)) == 100
    assert first == sorted(first, key=int)
    assert set(first) <= set(pool)
    assert first != select_uniform_random_subset(pool, 100, seed=43)


def test_partition_rejects_reused_ids():
    try:
        check_role_partition({"train": ["1", "2"], "test": ["2", "3"]}, ["1", "2", "3"])
    except ValueError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("Overlapping roles were accepted.")
