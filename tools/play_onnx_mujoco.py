"""Play an exported policy (tools/export_onnx.py) in plain MuJoCo with mujoco.viewer, on CPU.

Unlike tools/play_onnx.py this needs no mjlab / MuJoCo-Warp / CUDA: it runs the ONNX policy
directly against the MJCF in the repo, with the joint PD loop applied through `qfrc_applied`.
The observation is assembled from raw simulator state exactly as a deployment controller would
(joint encoders, root attitude and gyro), which is the same contract as tools/play_onnx.py.

The MJCF has no actuators (mjlab injects them at training time), so the PD law is applied here:
    tau = kp * (q_des - q) - kd * qd  [+ qfrc_bias]
with kp / kd / action_scale / default pose read from the ONNX metadata.

Usage:
    python tools/play_onnx_mujoco.py my_run/jumps1_subject1.onnx --viewer
    python tools/play_onnx_mujoco.py my_run/jumps1_subject1.onnx --seconds 10 --start 400
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("--xml", default="data/g1_description/g1_29dof.xml")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after N seconds (0 = whole motion)")
    ap.add_argument("--start", type=int, default=0, help="first reference frame (50 Hz)")
    ap.add_argument("--viewer", action="store_true", help="open mujoco.viewer (needs a display)")
    ap.add_argument("--no-floor", action="store_true", help="do not add a ground plane and light")
    ap.add_argument("--no-bias", action="store_true", help="drop the qfrc_bias feed-forward term")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed for the viewer")
    ap.add_argument("--no-follow", action="store_true", help="keep the camera fixed instead of tracking the root")
    ap.add_argument("--no-ghost", action="store_true", help="do not draw the reference motion skeleton")
    ap.add_argument("--ghost-offset", type=float, default=0.0, help="shift the reference skeleton sideways (m) for a side-by-side view")
    ap.add_argument("--cam-distance", type=float, default=3.0, help="viewer camera distance in metres")
    ap.add_argument("--cam-elevation", type=float, default=-15.0, help="viewer camera elevation in degrees")
    ap.add_argument("--cam-azimuth", type=float, default=90.0, help="viewer camera azimuth in degrees")
    a = ap.parse_args()

    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    vec = lambda k: np.array(meta[k].split(","), dtype=np.float64)  # noqa: E731
    q0, kp, kd, scale = vec("default_joint_pos"), vec("joint_stiffness"), vec("joint_damping"), vec("action_scale")
    names, H = meta["joint_names"].split(","), int(meta["observation_history"])
    arm = vec("joint_armature") if "joint_armature" in meta else np.zeros(len(names))
    eff = vec("joint_effort_limit") if "joint_effort_limit" in meta else np.full(len(names), np.inf)
    last = int(meta["motion_num_frames"]) - 1

    spec = mujoco.MjSpec.from_file(a.xml)  # the bare robot MJCF has neither floor nor light
    if not a.no_floor and not any(g.type == mujoco.mjtGeom.mjGEOM_PLANE for g in spec.geoms):
        floor = spec.worldbody.add_geom()
        floor.type, floor.size, floor.rgba = mujoco.mjtGeom.mjGEOM_PLANE, [0.0, 0.0, 0.05], [0.22, 0.24, 0.27, 1.0]
        sun = spec.worldbody.add_light()
        sun.pos, sun.dir, sun.diffuse = [0.0, 0.0, 4.0], [0.0, 0.0, -1.0], [0.8, 0.8, 0.8]
    model = spec.compile()
    data = mujoco.MjData(model)
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    assert all(j >= 0 for j in jids), "joint names in the ONNX do not match the MJCF"
    qadr, vadr = model.jnt_qposadr[np.array(jids)], model.jnt_dofadr[np.array(jids)]
    assert model.nu == 0, "the reference MJCF is unactuated by design; torques go through qfrc_applied"
    model.dof_armature[vadr] = arm  # mjlab adds the actuator armature; without it the joints are far too light
    substeps = int(round(float(meta["control_dt"]) / model.opt.timestep))

    def run(hist: np.ndarray, f: int) -> dict:
        outs = sess.run(None, {"obs_history": hist[None].astype(np.float32), "time_step": np.array([[float(f)]], dtype=np.float32)})
        return dict(zip([o.name for o in sess.get_outputs()], outs))

    def obs_of(a_prev: np.ndarray, ref_quat: np.ndarray) -> np.ndarray:
        p, R = data.qpos[:3].copy(), quat_to_mat(data.qpos[3:7])
        q, qd = data.qpos[qadr], data.qvel[vadr]
        w = data.qvel[3:6]  # free-joint angular velocity is already in the body frame
        R_rel = R.T @ quat_to_mat(ref_quat)
        return np.concatenate((q - q0, qd, w, -R[2], a_prev, R_rel[:, :2].reshape(6)))

    f = max(0, min(a.start, last))
    init = run(np.zeros((H, len(names) + len(names) + 3 + 3 + len(names) + 6)), f)
    data.qpos[:3] = init["body_pos_w"][0, 0]
    data.qpos[3:7] = init["body_quat_w"][0, 0] / np.linalg.norm(init["body_quat_w"][0, 0])
    data.qpos[qadr] = init["joint_pos"][0]
    data.qvel[vadr] = init["joint_vel"][0]
    mujoco.mj_forward(model, data)
    ref_quat = init["body_quat_w"][0, 0].copy()

    a_prev = np.zeros(len(names))
    hist = np.repeat(obs_of(a_prev, ref_quat)[None], H, 0)
    viewer = mujoco.viewer.launch_passive(model, data) if a.viewer else None
    a.follow, a.ghost = not a.no_follow, not a.no_ghost

    # Blue translucent skeleton = the reference motion the policy is asked to track
    # (positions come straight out of the ONNX graph, body 0 = anchor/root).
    REF_RGBA = np.array([0.25, 0.62, 1.0, 0.45], dtype=np.float32)
    scn, ref_edges = None, []
    if viewer is not None and a.ghost:
        tracked = meta["body_names"].split(",")
        scn = viewer.user_scn
        idx = {n: i for i, n in enumerate(tracked)}
        for n in tracked:
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
            p = int(model.body_parentid[b]) if b >= 0 else 0
            while p > 0 and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, p) not in idx:
                p = int(model.body_parentid[p])
            if p > 0:
                ref_edges.append((idx[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, p)], idx[n]))

    shift = np.array([0.0, a.ghost_offset, 0.0])

    def draw_reference(pos: np.ndarray) -> None:
        pos = pos + shift
        scn.ngeom = 0
        for i, j in ref_edges:
            if scn.ngeom >= scn.maxgeom:
                return
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.018, pos[i].astype(np.float64), pos[j].astype(np.float64))
            g.rgba = REF_RGBA
            scn.ngeom += 1
        for k in range(len(pos)):
            if scn.ngeom >= scn.maxgeom:
                return
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([0.032, 0.032, 0.032]), pos[k].astype(np.float64), np.eye(3).reshape(-1), REF_RGBA)
            scn.ngeom += 1
    if viewer is not None:
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = a.cam_distance, a.cam_elevation, a.cam_azimuth
        viewer.cam.lookat[:] = [0.0, 0.0, 0.7]

    t_end = time.perf_counter() + a.seconds if a.seconds > 0 else None
    t_wall0, steps = time.perf_counter(), 0
    print(f"policy {meta['motion']} | frames {f}..{last} | {substeps} physics steps per {float(meta['control_dt'])*1e3:.0f} ms")
    while f < last:
        if viewer is not None and not viewer.is_running():
            break
        obs = obs_of(a_prev, ref_quat)
        hist = np.concatenate((hist[1:], obs[None]))
        out = run(hist, f)
        act = out["actions"][0].astype(np.float64)
        q_des = q0 + scale * act
        for _ in range(substeps):
            tau = np.clip(kp * (q_des - data.qpos[qadr]) - kd * data.qvel[vadr], -eff, eff)
            data.qfrc_applied[vadr] = tau if a.no_bias else tau + data.qfrc_bias[vadr]
            mujoco.mj_step(model, data)
        a_prev, ref_quat, f = act, out["next_anchor_quat_w"][0], f + 1
        steps += 1

        if viewer is not None:
            if scn is not None:
                draw_reference(out["body_pos_w"][0])
            if a.follow:
                viewer.cam.lookat[:] = data.qpos[:3]
            viewer.sync()
            lag = steps * float(meta["control_dt"]) / a.speed - (time.perf_counter() - t_wall0)
            if lag > 0:
                time.sleep(lag)
        if steps % 50 == 0:
            err = np.abs(data.qpos[qadr] - out["joint_pos"][0]).mean()
            print(f"  f={f:5d} t={f*float(meta['control_dt']):6.2f}s  root_z={data.qpos[2]:.3f}  |q-q_ref|={err:.3f}  steps/s={steps/(time.perf_counter()-t_wall0):.0f}")
        if t_end is not None and time.perf_counter() > t_end:
            break

    print(f"done: {steps} control steps, {steps*float(meta['control_dt']):.2f} s simulated, final root_z={data.qpos[2]:.3f}")
    if viewer is not None:
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)
        viewer.close()


if __name__ == "__main__":
    main()
