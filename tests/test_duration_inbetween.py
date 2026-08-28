import unittest
from unittest import mock

import numpy as np

from motion_geometry.inbetween import (build_inbetween, duration_displacement,
    forward_kinematics, right_jacobian, quintic, constrain_supported_feet)
from motion_geometry.rotations import so3_exp_np, so3_log_np, matrix_to_rot6d_np, rot6d_to_matrix_np
from training import motion_models as m
from routing.boundary_closed_loop import align_core_to_prev, build_bridge


def motion(frames=6):
    out = np.zeros((frames,151),np.float32)
    out[:,5] = 1
    out[:,7:] = matrix_to_rot6d_np(np.broadcast_to(np.eye(3),(frames,24,3,3)),project=False).reshape(frames,144)
    return out


class DurationInbetweenTests(unittest.TestCase):
    def test_forward_landing_through_real_runtime_no_backtracking(self):
        cfg = m.MotionGenerationConfig()
        for width in (10,28):
            left,right = motion(),motion()
            left[:,4] = np.arange(6)/cfg.fps
            right[:,4] = 5+np.arange(6)/cfg.fps
            aligned,report = align_core_to_prev(m,left,right,cfg,transition_frames=width)
            bridge = build_bridge(m,left,aligned,width,cfg)
            joined = np.concatenate([left,bridge,aligned])
            np.testing.assert_allclose(np.diff(joined[:,4])*cfg.fps,1,atol=3e-5)
            self.assertAlmostEqual(report["landing"]["duration_seconds"],(width+1)/cfg.fps)
            self.assertEqual(len(joined),12+width)
            np.testing.assert_array_equal(left[:,4],np.arange(6,dtype=np.float32)/cfg.fps)

    def test_stationary_landing_and_zero_bridge(self):
        left,right = motion(),motion()
        right[:,4] = 5
        np.testing.assert_array_equal(duration_displacement(left,right,28,30)[0],0)
        self.assertEqual(build_inbetween(left,right,0,30)[0].shape,(0,151))

    def test_rotation_preserves_constant_angular_velocity(self):
        for width in (10,28):
            left,right = motion(),motion()
            a = np.arange(-5,1)/30
            b = (width+1+np.arange(6))/30
            for clip,angles in ((left,a),(right,b)):
                rotations = rot6d_to_matrix_np(clip[:,7:].reshape(-1,24,6),project=False)
                axis = np.array([.3,.4,.5]); axis /= np.linalg.norm(axis)
                rotations[:,16] = so3_exp_np(angles[:,None]*axis)
                clip[:,7:] = matrix_to_rot6d_np(rotations,project=False).reshape(-1,144)
            bridge,_ = build_inbetween(left,right,width,30,contact_ik=False)
            r = rot6d_to_matrix_np(np.concatenate([left,bridge,right])[:,7:].reshape(-1,24,6),project=False)[:,16]
            speed = np.linalg.norm(so3_log_np(np.swapaxes(r[:-1],-1,-2)@r[1:],project=False),axis=-1)*30
            np.testing.assert_allclose(speed,1,atol=5e-5)

    def test_mismatched_pose_still_matches_boundary_velocity(self):
        # The endpoints need not lie on a constant-speed path; test boundary limits.
        from motion_geometry.inbetween import _angular_boundary
        left,right = motion(4),motion(4)
        for clip,angles in ((left,np.arange(-3,1)/30),(right,.1+np.arange(4)/30)):
            rotations = rot6d_to_matrix_np(clip[:,7:].reshape(-1,24,6),project=False)
            rotations[:,16] = so3_exp_np(np.stack([angles,angles*0,angles*0],-1))
            clip[:,7:] = matrix_to_rot6d_np(rotations,project=False).reshape(-1,144)
        # At a much finer sampling rate, endpoint estimates must converge to 1rad/s.
        fine_left,fine_right = motion(4),motion(4)
        for clip,angles in ((fine_left,np.arange(-3,1)/3000),(fine_right,.1+np.arange(4)/3000)):
            rotations = rot6d_to_matrix_np(clip[:,7:].reshape(-1,24,6),project=False)
            rotations[:,16] = so3_exp_np(np.stack([angles,angles*0,angles*0],-1))
            clip[:,7:] = matrix_to_rot6d_np(rotations,project=False).reshape(-1,144)
        bridge,_ = build_inbetween(fine_left,fine_right,2899,3000,contact_ik=False)
        r = rot6d_to_matrix_np(np.concatenate([fine_left[-1:],bridge,fine_right[:1]])[:,7:].reshape(-1,24,6),project=False)[:,16]
        speed = so3_log_np(np.swapaxes(r[:-1],-1,-2)@r[1:],project=False)[:,0]*3000
        np.testing.assert_allclose(speed[[0,-1]],1,atol=.002)

    def test_near_pi_and_jacobian(self):
        x = np.array([[0,0,0],[np.pi-1e-7,0,0],[.7,.3,-1.2]])
        np.testing.assert_allclose(right_jacobian(x)@right_jacobian(x,inverse=True),
            np.broadcast_to(np.eye(3),(3,3,3)),atol=1e-9)
        left,right = motion(),motion()
        right[:,7:] = matrix_to_rot6d_np(so3_exp_np(np.broadcast_to([np.pi-1e-6,0,0],(6,24,3))),project=False).reshape(6,144)
        bridge,_ = build_inbetween(left,right,28,30,contact_ik=False)
        self.assertTrue(np.isfinite(bridge).all())
        r = rot6d_to_matrix_np(bridge[:,7:].reshape(-1,24,6),project=False)
        np.testing.assert_allclose(np.linalg.det(r),1,atol=1e-5)

    def test_contact_ik_reduces_supported_foot_error(self):
        left,right = motion(),motion()
        base,_ = build_inbetween(left,right,28,30,contact_ik=False)
        # Controlled knee flexion moves the planted toe; outside anchors stay fixed.
        r = rot6d_to_matrix_np(base[:,7:].reshape(-1,24,6),project=False)
        perturb = .08*np.sin(np.linspace(0,np.pi,28))**2
        r[:,4] = so3_exp_np(np.stack([perturb,perturb*0,perturb*0],-1))
        base[:,7:] = matrix_to_rot6d_np(r,project=False).reshape(-1,144)
        fixed,report = constrain_supported_feet(base,left,right,30)
        anchor = forward_kinematics(left)[0][-1,10]
        before = np.linalg.norm(forward_kinematics(base)[0][:,10]-anchor,axis=-1).max()
        after = np.linalg.norm(forward_kinematics(fixed)[0][:,10]-anchor,axis=-1).max()
        self.assertIn(10,report["eligible_feet"])
        self.assertLess(after,before*.01)
        np.testing.assert_array_equal(fixed[:,4:7],base[:,4:7])
        np.testing.assert_array_equal(left,motion())

    def test_swing_and_different_foot_placements_not_locked(self):
        left,right = motion(),motion()
        right[:,4] = 1
        base,_ = build_inbetween(left,right,10,30,contact_ik=False)
        fixed,report = constrain_supported_feet(base,left,right,30)
        self.assertEqual(report["eligible_feet"],[])
        np.testing.assert_array_equal(fixed,base)

    def test_formal_errors_do_not_fall_back(self):
        with mock.patch.object(m,"reference_motion_inbetween_np",side_effect=ValueError("invalid bridge")):
            with self.assertRaisesRegex(ValueError,"invalid bridge"):
                build_bridge(m,motion(),motion(),10,m.MotionGenerationConfig())
        with mock.patch.object(m,"_align_core_to_previous",side_effect=ValueError("invalid landing")):
            with self.assertRaisesRegex(ValueError,"invalid landing"):
                align_core_to_prev(m,motion(),motion(),m.MotionGenerationConfig(),transition_frames=10)

    def test_ik_jerk_regression_rolls_back_with_explicit_report(self):
        import motion_geometry.inbetween as module
        left,right = motion(),motion()
        pure,_ = build_inbetween(left,right,28,30,contact_ik=False)
        bad = pure.copy(); bad[::2,4] += .02
        detail = {"eligible_feet":[10],"max_target_error_m":0.,"before_target_error_m":.01,"unreachable_frames":0}
        with mock.patch.object(module,"constrain_supported_feet",return_value=(bad,detail)):
            result,report = build_inbetween(left,right,28,30)
        np.testing.assert_array_equal(result,pure)
        self.assertFalse(report["support"]["committed"])
        self.assertIn("jerk_regressed",report["support"]["rollback_reasons"])
        self.assertEqual(report["support"]["applied_max_target_error_m"],.01)

    def test_scheduler_and_training_share_geometry_and_duration(self):
        from support.motion_geometry import make_so3_transition
        from routing.boundary_closed_loop import choose_transition_lengths
        left,right = motion(),motion()
        left[:,4] = np.arange(6)/30
        right[:,4] = 5+np.arange(6)/30
        aligned,_ = m._align_core_to_previous(left,right,m.MotionGenerationConfig(),transition_frames=28)
        reference = m.reference_motion_inbetween_np(left,aligned,28,m.MotionGenerationConfig(),finalize_contract=False)
        scheduler = make_so3_transition(left,aligned,28)
        np.testing.assert_array_equal(reference[:,4:],scheduler[:,4:])
        cfg = m.MotionGenerationConfig()
        core,length,info = m._choose_core_and_transition_lengths(20,300,True,cfg)
        self.assertLessEqual(length,28)
        self.assertEqual(core+length,300)
        with mock.patch.dict("os.environ",{"MOTION_TRANSITION_MAX_FRAMES":"36"}):
            with self.assertRaisesRegex(ValueError,"exceeds the trained seam length"):
                choose_transition_lengths(m,left,6,120,right,{},cfg)
