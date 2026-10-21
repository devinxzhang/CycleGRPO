import math

from generalized.verl.utils.spatiotemporal import (
    SpatioTemporalClaim,
    atomic_spatiotemporal_reward,
    parse_spatiotemporal_claims,
    spatiotemporal_caption_no_leak_reward,
    spatiotemporal_support_iou,
    strip_spatiotemporal_priors,
    trajectory_spatiotemporal_reward,
)


def claim(t, box):
    return SpatioTemporalClaim("target", t, (tuple(box),))


def test_normalized_and_unit_coordinates_match():
    parsed = parse_spatiotemporal_claims(
        '<answer><obj>person</obj><box>[0.1,0.2,0.4,0.6]</box>at<t>5</t>s</answer>',
        only_answer=True,
    )
    assert parsed[0].boxes == ((100.0, 200.0, 400.0, 600.0),)


def test_exact_sparse_support_is_one():
    region = [claim(5, [100, 100, 500, 500]), claim(9, [200, 200, 300, 400])]
    assert spatiotemporal_support_iou(region, region) == 1.0


def test_wrong_time_is_not_spatially_correct():
    gt = [claim(5, [100, 100, 500, 500])]
    pred = [claim(9.1, [100, 100, 500, 500])]
    assert spatiotemporal_support_iou(pred, gt, temporal_tolerance=3.0) == 0.0


def test_extra_support_is_penalized():
    gt = [claim(5, [0, 0, 100, 100])]
    pred = gt + [claim(7, [200, 200, 300, 300])]
    assert math.isclose(spatiotemporal_support_iou(pred, gt), 0.5)


def test_atomic_reward_retains_temporal_sensitivity():
    gt = '<answer><obj>person</obj><box>[100,100,500,500]</box>at<t>5</t>s</answer>'
    offset = gt.replace('<t>5</t>', '<t>7</t>')
    assert atomic_spatiotemporal_reward(gt, gt) == 1.0
    assert 0.0 < atomic_spatiotemporal_reward(offset, gt) < 1.0


def test_captions_cannot_leak_region_coordinates():
    clean = 'A person raises a glass beside another person.'
    leaked = 'The person at 5 seconds is in [0, 0, 100, 100].'
    assert spatiotemporal_caption_no_leak_reward(clean) == 1.0
    assert spatiotemporal_caption_no_leak_reward(leaked) == 0.0
    assert '5 seconds' not in strip_spatiotemporal_priors(leaked)


def test_trajectory_exact_match():
    gt = ('<answer><obj>person</obj><box>[0,0,100,100]</box>at<t>5</t>s'
          '<obj>person</obj><box>[20,20,120,120]</box>at<t>7</t>s</answer>')
    assert trajectory_spatiotemporal_reward(gt, gt) == 1.0
