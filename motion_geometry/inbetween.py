"""Duration-aware, boundary-conditioned SMPL24 inbetweening (no learned weights).

The bridge excludes both anchors.  All boundary derivatives use observed context,
never an occluded clean interior.  Contact IK is deliberately conservative: only
a foot supported at BOTH anchors, with compatible world positions, may be locked.
This is a proposal, not a proof of physical validity; downstream audits still run.
"""
from __future__ import annotations

import numpy as np

from motion_geometry.rotations import (
    matrix_to_rot6d_np, rot6d_to_matrix_np, so3_exp_np, so3_log_np,
)
from motion_geometry.smpl24 import OFFSETS, PARENTS

INBETWEEN_PROTOCOL = "duration_c2_so3_contact_bridge_v2"


def _skew(x):
    out = np.zeros(x.shape[:-1] + (3, 3), dtype=np.float64)
    out[..., 0, 1], out[..., 0, 2] = -x[..., 2], x[..., 1]
    out[..., 1, 0], out[..., 1, 2] = x[..., 2], -x[..., 0]
    out[..., 2, 0], out[..., 2, 1] = -x[..., 1], x[..., 0]
    return out


def right_jacobian(x, *, inverse=False):
    """Map log-chart derivatives to body angular velocity; stable at zero/pi."""
    x = np.asarray(x, dtype=np.float64)
    theta = np.linalg.norm(x, axis=-1)[..., None, None]
    t2 = theta * theta
    k = _skew(x)
    if inverse:
        coefficient = np.where(theta < 1e-4, 1 / 12 + t2 / 720,
            (1 - 0.5 * theta / np.maximum(np.tan(theta / 2), 1e-15)) / np.maximum(t2, 1e-15))
        return np.eye(3) + 0.5 * k + coefficient * (k @ k)
    a = np.where(theta < 1e-4, 0.5 - t2 / 24, (1 - np.cos(theta)) / np.maximum(t2, 1e-15))
    b = np.where(theta < 1e-4, 1 / 6 - t2 / 120,
                 (theta - np.sin(theta)) / np.maximum(theta**3, 1e-15))
    return np.eye(3) - a * k + b * (k @ k)


def quintic(p0, p1, v0, v1, a0, a1, phase):
    """Quintic Hermite; derivatives are with respect to normalized phase."""
    p0, p1, v0, v1, a0, a1 = [np.asarray(x, dtype=np.float64) for x in (p0,p1,v0,v1,a0,a1)]
    s = np.asarray(phase, dtype=np.float64).reshape((-1,) + (1,) * p0.ndim)
    c0, c1, c2 = p0, v0, a0 / 2
    d = p1 - (c0 + c1 + c2)
    v = v1 - (c1 + 2 * c2)
    a = a1 - 2 * c2
    c3, c4, c5 = 10*d - 4*v + a/2, -15*d + 7*v - a, 6*d - 3*v + a/2
    return c0 + s*(c1 + s*(c2 + s*(c3 + s*(c4 + s*c5))))


def _linear_boundary(clip, fps, end):
    clip = np.asarray(clip, dtype=np.float64)
    v = np.zeros_like(clip[0])
    a = np.zeros_like(v)
    if len(clip) >= 2:
        v = (clip[-1] - clip[-2] if end else clip[1] - clip[0]) * fps
    if len(clip) >= 3:
        a = (clip[-1] - 2*clip[-2] + clip[-3] if end else clip[2] - 2*clip[1] + clip[0]) * fps**2
        # One-sided velocity extrapolated from the interval centre to the anchor.
        v = v + (0.5 if end else -0.5) * a / fps
    return v, a


def _angular_boundary(clip, fps, end):
    zero = np.zeros(clip.shape[1:-2] + (3,), dtype=np.float64)
    if len(clip) < 2:
        return zero, zero.copy()
    w = so3_log_np(np.swapaxes(clip[:-1], -1, -2) @ clip[1:], project=False).astype(np.float64) * fps
    velocity = w[-1] if end else w[0]
    acceleration = (w[-1] - w[-2] if end else w[1] - w[0]) * fps if len(w) > 1 else zero
    return velocity + (0.5 if end else -0.5)*acceleration/fps, acceleration


def duration_displacement(prev, curr, n_frames, fps, max_speed=1.35):
    """XZ landing displacement for an inserted interval, in metres.

    A zero-length bridge keeps legacy hard-concatenation alignment.  Nonzero
    bridges use n+1 intervals, the same duration used by interpolation.
    """
    if n_frames <= 0:
        return np.zeros(2), {"duration_seconds": 0.0, "velocity_clipped": False}
    v0, _ = _linear_boundary(np.asarray(prev)[:, [4, 6]], fps, True)
    v1, _ = _linear_boundary(np.asarray(curr)[:, [4, 6]], fps, False)
    clipped = False
    for v in (v0, v1):
        norm = np.linalg.norm(v)
        if norm > max_speed:
            v *= max_speed / norm
            clipped = True
    duration = (n_frames + 1) / float(fps)
    displacement = (v0 + v1) * 0.5 * duration
    return displacement, {"duration_seconds": duration, "velocity_clipped": clipped,
                          "endpoint_velocity_xz_mps": [v0.tolist(), v1.tolist()]}


def forward_kinematics(motion):
    rotations = rot6d_to_matrix_np(np.asarray(motion)[..., 7:].reshape(-1,24,6), project=False).astype(np.float64)
    world = np.empty_like(rotations)
    joints = np.empty(rotations.shape[:-1], dtype=np.float64)
    joints[:, 0] = motion[:, 4:7]
    world[:, 0] = rotations[:, 0]
    for joint in range(1,24):
        parent = PARENTS[joint]
        world[:, joint] = world[:, parent] @ rotations[:, joint]
        joints[:, joint] = joints[:, parent] + (world[:, parent] @ OFFSETS[joint].astype(np.float64)[:,None])[..., 0]
    return joints, world, rotations


def _between_vectors(a, b):
    a = a / np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)
    b = b / np.maximum(np.linalg.norm(b,axis=-1,keepdims=True),1e-12)
    cross = np.cross(a,b)
    norm = np.linalg.norm(cross,axis=-1,keepdims=True)
    dot = np.clip(np.sum(a*b,axis=-1,keepdims=True),-1,1)
    axis = cross / np.maximum(norm,1e-12)
    # Deterministic antipodal axis; normal legs should not need this branch.
    basis = np.eye(3)[np.argmin(np.abs(a),axis=-1)]
    alternative = np.cross(a,basis)
    alternative /= np.maximum(np.linalg.norm(alternative,axis=-1,keepdims=True),1e-12)
    axis = np.where((norm < 1e-10) & (dot < 0),alternative,axis)
    return so3_exp_np(axis * np.arctan2(norm,dot)).astype(np.float64)


def constrain_supported_feet(bridge, prev, curr, fps):
    """Batched analytic two-bone IK with unchanged root/upper body/context.

    No hidden target motion is used. Eligibility is determined ONLY from the
    outside anchors. Swing feet and incompatible foot placements are untouched.
    Foot orientation is preserved; unreachable targets are reported, not hidden.
    """
    out = np.asarray(bridge,dtype=np.float32).copy()
    left, _, _ = forward_kinematics(prev)
    right, _, _ = forward_kinematics(curr)
    phase = np.arange(1,len(out)+1,dtype=np.float64)/(len(out)+1)
    report = {"eligible_feet": [], "max_target_error_m": 0.0, "before_target_error_m":0.0, "unreachable_frames": 0}
    for hip,knee,ankle,toe in ((1,4,7,10),(2,5,8,11)):
        p0,p1 = left[-1,toe],right[0,toe]
        v0,a0 = _linear_boundary(left[:,toe],fps,True)
        v1,a1 = _linear_boundary(right[:,toe],fps,False)
        floor = min(left[-1,[10,11],1].min(),right[0,[10,11],1].min())
        eligible = (max(p0[1],p1[1]) <= floor + .055 and
                    max(np.linalg.norm(v0),np.linalg.norm(v1)) <= .18 and
                    np.linalg.norm(p1-p0) <= .03)
        if not eligible:
            continue
        report["eligible_feet"].append(toe)
        duration = (len(out)+1)/float(fps)
        target = quintic(p0,p1,v0*duration,v1*duration,a0*duration**2,a1*duration**2,phase)
        joints,world,local = forward_kinematics(out)
        report["before_target_error_m"] = max(report["before_target_error_m"],
            float(np.linalg.norm(joints[:,toe]-target,axis=-1).max()))
        foot_rotation = world[:,ankle].copy()
        target_ankle = target - (foot_rotation @ OFFSETS[toe].astype(np.float64)[:,None])[...,0]
        base = joints[:,hip]
        vector = target_ankle-base
        distance = np.linalg.norm(vector,axis=-1,keepdims=True)
        direction = vector/np.maximum(distance,1e-12)
        upper,lower = float(np.linalg.norm(OFFSETS[knee])),float(np.linalg.norm(OFFSETS[ankle]))
        reachable = np.clip(distance,abs(upper-lower)+1e-6,upper+lower-1e-6)
        report["unreachable_frames"] += int(np.count_nonzero(np.abs(reachable-distance) > 1e-5))
        plane = joints[:,knee]-base
        plane -= np.sum(plane*direction,axis=-1,keepdims=True)*direction
        alternate = np.cross(direction,np.eye(3)[np.argmin(np.abs(direction),axis=-1)])
        plane = np.where(np.linalg.norm(plane,axis=-1,keepdims=True)>1e-8,plane,alternate)
        plane /= np.maximum(np.linalg.norm(plane,axis=-1,keepdims=True),1e-12)
        along = (upper**2-lower**2+reachable**2)/(2*reachable)
        height = np.sqrt(np.maximum(upper**2-along**2,0))
        desired_knee = base+direction*along+plane*height
        desired_ankle = base+direction*reachable
        hip_delta = _between_vectors(joints[:,knee]-base,desired_knee-base)
        new_hip = hip_delta @ world[:,hip]
        knee_delta = _between_vectors((hip_delta @ (joints[:,ankle]-joints[:,knee])[...,None])[...,0],
                                     desired_ankle-desired_knee)
        new_knee = knee_delta @ hip_delta @ world[:,knee]
        local[:,hip] = np.swapaxes(world[:,PARENTS[hip]],-1,-2) @ new_hip
        local[:,knee] = np.swapaxes(new_hip,-1,-2) @ new_knee
        local[:,ankle] = np.swapaxes(new_knee,-1,-2) @ foot_rotation
        out[:,7:] = matrix_to_rot6d_np(local,project=False).reshape(-1,144)
        actual,_,_ = forward_kinematics(out)
        report["max_target_error_m"] = max(report["max_target_error_m"],float(np.linalg.norm(actual[:,toe]-target,axis=-1).max()))
    return out,report


def _local_boundary_scores(prev, bridge, curr, fps):
    """Same seam stencil/units as boundary_metrics_torch, without Torch import."""
    before,after = prev[-3:],curr[:3]
    joined = np.concatenate([before,bridge,after])
    xyz = forward_kinematics(joined)[0]
    start,stop = len(before),len(before)+len(bridge)
    core = np.zeros(len(joined),bool); core[start:stop] = True
    jumps = np.linalg.norm(np.diff(xyz,n=2,axis=0)*fps,axis=-1).mean(-1)
    boundary = []
    if start>=2: boundary.append(jumps[start-2])
    if len(after)>=2: boundary.append(jumps[stop-1])
    energy = 0.0
    jerk = 0.0
    for order,scale in ((2,10),(3,1000)):
        count = len(xyz)-order
        support = np.stack([core[i:i+count] for i in range(order+1)]).any(0)
        values = np.linalg.norm(np.diff(xyz,n=order,axis=0)*fps**order,axis=-1).mean(-1)
        mean = float(values[support].mean())
        energy += mean/scale
        if order==3: jerk=mean
    return {"endpoint":float(np.mean(boundary)) if boundary else 0.0,"temporal":energy,"jerk":jerk}


def build_inbetween(prev, curr, n_frames, fps, *, max_root_speed=1.35, contact_ik=True,
                     max_vertical_speed=.9, max_angular_speed=8.0, root_tangent_margin=.12):
    prev,curr = np.asarray(prev,dtype=np.float64),np.asarray(curr,dtype=np.float64)
    if prev.ndim != 2 or curr.ndim != 2 or prev.shape[1] != 151 or curr.shape[1] != 151:
        raise ValueError("inbetween requires nonempty [T,151] contexts")
    if not len(prev) or not len(curr) or fps <= 0 or not np.isfinite(prev).all() or not np.isfinite(curr).all():
        raise ValueError("invalid inbetween context/fps")
    if any(not np.isfinite(v) or v < 0 for v in (max_root_speed,max_vertical_speed,max_angular_speed,root_tangent_margin)):
        raise ValueError("invalid bridge derivative caps")
    if n_frames <= 0:
        return np.empty((0,151),np.float32),{"schema":INBETWEEN_PROTOCOL}
    duration = (n_frames+1)/float(fps)
    phase = np.arange(1,n_frames+1,dtype=np.float64)/(n_frames+1)
    v0,a0 = _linear_boundary(prev[:,4:7],fps,True)
    v1,a1 = _linear_boundary(curr[:,4:7],fps,False)
    clipped = False
    tangent_limit = np.linalg.norm(curr[0,4:7]-prev[-1,4:7])+root_tangent_margin
    for velocity in (v0,v1):
        norm = np.linalg.norm(velocity[[0,2]])
        if norm > max_root_speed:
            velocity[[0,2]] *= max_root_speed/norm
            clipped = True
        if abs(velocity[1]) > max_vertical_speed:
            velocity[1] = np.clip(velocity[1],-max_vertical_speed,max_vertical_speed)
            clipped = True
        norm = np.linalg.norm(velocity)*duration
        if norm > tangent_limit:
            velocity *= tangent_limit/max(norm,1e-12)
            clipped = True
    root = quintic(prev[-1,4:7],curr[0,4:7],v0*duration,v1*duration,a0*duration**2,a1*duration**2,phase)
    ra = rot6d_to_matrix_np(prev[:,7:].reshape(-1,24,6),project=False).astype(np.float64)
    rb = rot6d_to_matrix_np(curr[:,7:].reshape(-1,24,6),project=False).astype(np.float64)
    delta = so3_log_np(np.swapaxes(ra[-1],-1,-2) @ rb[0],project=False).astype(np.float64)
    w0,alpha0 = _angular_boundary(ra,fps,True)
    w1,alpha1 = _angular_boundary(rb,fps,False)
    angular_clipped = 0
    for velocity in (w0,w1):
        norm = np.linalg.norm(velocity,axis=-1,keepdims=True)
        angular_clipped += int(np.count_nonzero(norm>max_angular_speed))
        velocity *= np.minimum(1,max_angular_speed/np.maximum(norm,1e-12))
    u0 = w0*duration
    inv = right_jacobian(delta,inverse=True)
    u1 = (inv @ (w1*duration)[...,None])[...,0]
    eps = 1e-5 / max(1.0,float(np.linalg.norm(u1,axis=-1).max()))
    jacobian_derivative = (right_jacobian(delta+eps*u1)-right_jacobian(delta-eps*u1))/(2*eps)
    chart_a1 = (inv @ (alpha1*duration**2-(jacobian_derivative @ u1[...,None])[...,0])[...,None])[...,0]
    chart = quintic(np.zeros_like(delta),delta,u0,u1,alpha0*duration**2,chart_a1,phase)
    rotation = ra[-1][None] @ so3_exp_np(chart).astype(np.float64)
    out = np.empty((n_frames,151),dtype=np.float32)
    s = phase[:,None]**3*(10-15*phase[:,None]+6*phase[:,None]**2)
    out[:,:4] = (1-s)*prev[-1,:4]+s*curr[0,:4]
    out[:,4:7] = root
    out[:,7:] = matrix_to_rot6d_np(rotation,project=False).reshape(n_frames,144)
    support = {"eligible_feet": [],"max_target_error_m":0.0,"unreachable_frames":0,"committed":False}
    if contact_ik:
        candidate,support = constrain_supported_feet(out,prev,curr,fps)
        if support["eligible_feet"]:
            initial = _local_boundary_scores(prev,out,curr,fps)
            proposed = _local_boundary_scores(prev,candidate,curr,fps)
            reasons = [key+"_regressed" for key in ("endpoint","temporal") if proposed[key]>initial[key]+1e-6]
            if proposed["jerk"]>initial["jerk"]*1.02+1e-6: reasons.append("jerk_regressed")
            if support["unreachable_frames"]: reasons.append("unreachable_foot_target")
            support.update(committed=not reasons,rollback_reasons=reasons,before=initial,candidate=proposed)
            support["applied_max_target_error_m"] = support["max_target_error_m"] if not reasons else support["before_target_error_m"]
            if not reasons:
                out = candidate
        else:
            support.update(committed=False,rollback_reasons=["no_two_sided_support_evidence"])
    return out,{"schema":INBETWEEN_PROTOCOL,"duration_seconds":duration,
                "root_velocity_clipped":clipped,"angular_velocity_clipped_joints":angular_clipped,
                "contact_ik":bool(contact_ik),"support":support}
