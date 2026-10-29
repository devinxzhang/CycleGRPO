from generalized.verl.utils.spatiotemporal import spatiotemporal_temporal_proximity_reward
from generalized.verl.utils.temporal_dense_reward import temporal_dense_components


def test_dense_interval_reward_orders_disjoint_predictions():
    target = [(80.0, 90.0)]
    near = temporal_dense_components([(68.0, 78.0)], target)
    far = temporal_dense_components([(5.0, 15.0)], target)
    assert 0.0 < far["dense"] < near["dense"] < 1.0


def test_dense_interval_reward_is_one_for_exact_match():
    result = temporal_dense_components([(12.0, 18.0)], [(12.0, 18.0)])
    assert result == {
        "center": 1.0,
        "boundary": 1.0,
        "duration": 1.0,
        "dense": 1.0,
    }


def test_dense_interval_reward_rejects_invalid_predictions():
    result = temporal_dense_components([], [(12.0, 18.0)])
    assert result["dense"] == 0.0


def test_spatiotemporal_timestamp_proximity_is_dense():
    gt = "<answer><obj>person</obj><box>[0,0,10,10]</box>at<t>80</t>s</answer>"
    near = "<answer><obj>person</obj><box>[0,0,10,10]</box>at<t>75</t>s</answer>"
    far = "<answer><obj>person</obj><box>[0,0,10,10]</box>at<t>10</t>s</answer>"
    assert 0.0 < spatiotemporal_temporal_proximity_reward(far, gt) < spatiotemporal_temporal_proximity_reward(near, gt) < 1.0
