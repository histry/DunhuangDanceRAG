import numpy as np
import pytest

from training import motion_models as m

pytestmark = pytest.mark.skipif(m.torch is None, reason="PyTorch unavailable")


def test_observable_scale_floor_reduces_tiny_defect_normalization():
    torch=m.torch
    cfg=m.MotionGenerationConfig(
        device="cpu",
        product_refiner_observable_floor_quantile=.5,
        product_refiner_observable_floor_ratio=1.0,
    )
    baseline=torch.tensor([1e-5, 1e-2],dtype=torch.float64)
    proposed=baseline.clone().requires_grad_(True)
    floor=m._observable_scale_floor(baseline,cfg)
    _,gap=m._smooth_observable_margin(proposed,baseline,.10,scale_floor=floor)
    assert floor.item()>1e-5
    assert gap[0] < gap[1]


def test_clean_noop_has_no_deadband_for_nonzero_edit():
    torch=m.torch
    frames=12
    clean=torch.zeros((1,frames,151),dtype=torch.float32)
    clean[...,7:]=torch.as_tensor(np.tile(m.identity6d_np(),24),dtype=torch.float32)
    clean[...,5]=.95
    pred=clean.clone()
    pred[...,4]+=1e-4
    pred.requires_grad_(True)
    joint=torch.ones((1,frames,24))
    root=torch.ones((1,frames,1))
    contact=torch.zeros((1,frames,1))
    cfg=m.MotionGenerationConfig(device="cpu",product_refiner_clean_noop_weight=.03)
    loss,terms=m._product_refiner_clean_identity_loss(pred,clean,joint,root,contact,cfg)
    assert terms['geometry_excess']==0
    assert terms['noop']>0
    loss.backward()
    assert pred.grad is not None and pred.grad.abs().sum()>0
