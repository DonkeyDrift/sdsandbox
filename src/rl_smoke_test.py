#!/usr/bin/env python3
"""RL P0 protocol smoke test for the DonkeySim Unity sim (sdsandbox).

Talks the plain TCP JSON line protocol directly (stdlib only, no
gym_donkeycar). Assumes the sim app is already running in a track scene
(default mountain_track) on --host/--port, built with the RL P0 protocol
changes, and not paused (window focused or run-in-background enabled).

Numeric values are sent as strings: the sim's JSONObject handlers parse
fields via .str, exactly like test_client.py / gym_donkeycar clients.

Checks:
  a. telemetry fields: frame_id increments by 1, flipped/stuck_time/lap
     defaults, collision_impulse present
  b. time_scale=2.0 -> telemetry 'time' advances ~2x wall clock (+/-30%)
  c. node_position + reset_car(node_index/lateral_offset/heading_noise/seed)
     -> car lands next to the node, |cte| ~ lateral_offset, speed ~ 0
  d. same-seed resets are deterministic (pos delta < 0.01)
  e. deadman: controls stop -> brake engages -> speed falls below 0.5
  f. collision impulse soft observation (printed, never fails)
  g. car still alive and streaming telemetry at the end

Exit code 0 iff all asserted checks pass.
"""

import argparse
import json
import math
import socket
import sys
import time

STEP_TIMEOUT = 10.0


class SimClient:
    def __init__(self, host, port):
        self.sock = socket.create_connection((host, port), timeout=STEP_TIMEOUT)
        self.rfile = self.sock.makefile("r", encoding="utf-8", newline="\n")

    def send(self, msg):
        self.sock.sendall((json.dumps(msg) + "\n").encode("utf-8"))

    def recv_json(self, timeout):
        self.sock.settimeout(max(0.1, timeout))
        try:
            line = self.rfile.readline()
        except socket.timeout:
            raise TimeoutError("no message from sim within %.1fs" % timeout)
        if not line:
            raise ConnectionError("sim closed the connection")
        line = line.strip()
        if not line:
            return self.recv_json(timeout)
        return json.loads(line)

    def wait_for(self, msg_type, timeout=STEP_TIMEOUT):
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("no %r message within %.1fs" % (msg_type, timeout))
            msg = self.recv_json(remaining)
            if msg.get("msg_type") == msg_type:
                return msg

    def next_telemetry(self, timeout=STEP_TIMEOUT):
        return self.wait_for("telemetry", timeout)

    def settle(self, seconds):
        """Keep reading telemetry for `seconds`, return the last frame."""
        deadline = time.monotonic() + seconds
        last = None
        while time.monotonic() < deadline:
            msg = self.next_telemetry(timeout=max(1.0, deadline - time.monotonic() + 2.0))
            last = msg
        return last

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    line = "[%s] %s" % ("PASS" if ok else "FAIL", name)
    if detail:
        line += " -- %s" % detail
    print(line)
    return ok


def info(msg):
    print("[INFO] %s" % msg)


def send_control(c, steering=0.0, throttle=0.0, brake=0.0):
    c.send({"msg_type": "control",
            "steering": "%.3f" % steering,
            "throttle": "%.3f" % throttle,
            "brake": "%.3f" % brake})


def send_configs(c):
    c.send({"msg_type": "car_config",
            "body_style": "car01", "body_r": "255", "body_g": "0", "body_b": "255",
            "car_name": "RLSmoke",
            "rl_mode": "1", "control_timeout_ms": "800"})
    time.sleep(0.2)
    c.send({"msg_type": "cam_config",
            "fov": "150", "fish_eye_x": "0.0", "fish_eye_y": "0.0",
            "img_w": "160", "img_h": "120", "img_d": "3", "img_enc": "JPG",
            "offset_x": "0.0", "offset_y": "3.0", "offset_z": "0.0", "rot_x": "90.0"})
    time.sleep(0.2)


def handshake(c):
    deadline = time.monotonic() + 15.0
    loaded = False
    while time.monotonic() < deadline and not loaded:
        msg = c.recv_json(deadline - time.monotonic())
        mt = msg.get("msg_type")
        if mt in ("need_car_config", "car_loaded"):
            send_configs(c)
            loaded = mt == "car_loaded"
    if not loaded:
        raise TimeoutError("car_loaded not received; is the sim in a track scene?")


def step_a_basic_telemetry(c):
    print("\n-- a. basic telemetry fields")
    frames = []
    deadline = time.monotonic() + 3.0
    while len(frames) < 6 and time.monotonic() < deadline:
        frames.append(c.next_telemetry())
    check("a1 received >= 3 telemetry frames", len(frames) >= 3, "got %d" % len(frames))
    if len(frames) < 2:
        return None
    fids = [f.get("frame_id") for f in frames]
    ok = all(isinstance(f, int) for f in fids) and all(fids[i + 1] == fids[i] + 1 for i in range(len(fids) - 1))
    check("a2 frame_id increments by 1", ok, "frame_ids=%s" % fids)
    t = frames[-1]
    check("a3 flipped == False", t.get("flipped") is False, "flipped=%r" % t.get("flipped"))
    stuck = t.get("stuck_time")
    check("a4 stuck_time ~ 0", isinstance(stuck, (int, float)) and abs(stuck) < 0.05, "stuck_time=%r" % stuck)
    check("a5 lap == 0", t.get("lap") == 0, "lap=%r" % t.get("lap"))
    ci = t.get("collision_impulse")
    check("a6 collision_impulse present and numeric",
          isinstance(ci, (int, float)) and not isinstance(ci, bool), "collision_impulse=%r" % ci)
    return t


def step_b_time_scale(c):
    print("\n-- b. time_scale")
    c.send({"msg_type": "time_scale", "time_scale": "2.0"})
    time.sleep(0.5)  # 让 time_scale 生效并排空缓冲
    c.settle(0.5)
    # 连续读取一个墙钟窗口,比较窗口内 telemetry time 字段的总推进
    t0 = time.monotonic()
    first = c.next_telemetry()
    last = first
    while time.monotonic() - t0 < 2.0:
        last = c.next_telemetry(timeout=5.0)
    wall_dt = time.monotonic() - t0
    sim_dt = last["time"] - first["time"]
    ratio = sim_dt / wall_dt if wall_dt > 0 else 0.0
    check("b1 sim/wall time ratio ~ 2 at time_scale=2.0", 1.4 <= ratio <= 2.6,
          "ratio=%.2f (sim_dt=%.3fs wall_dt=%.3fs)" % (ratio, sim_dt, wall_dt))
    c.send({"msg_type": "time_scale", "time_scale": "1.0"})
    time.sleep(0.3)


def step_c_reset_at_node(c, total_nodes):
    print("\n-- c. reset_car with node_index/lateral_offset/heading_noise/seed")
    index = 50 if total_nodes > 50 else max(0, total_nodes // 2)
    c.send({"msg_type": "node_position", "index": "%d" % index})
    node = c.wait_for("node_position")
    node_pos = (node["pos_x"], node["pos_y"], node["pos_z"])
    check("c1 node_position received", True, "index=%d pos=%s" % (index, node_pos))

    c.send({"msg_type": "reset_car",
            "node_index": "%d" % index, "lateral_offset": "1.5",
            "heading_noise": "10", "seed": "42"})
    t = c.settle(0.6)
    car_pos = (t["pos_x"] * 8.0, t["pos_y"] * 8.0, t["pos_z"] * 8.0)
    dist = math.dist(car_pos, node_pos)
    check("c2 car lands near node (< 5m)", dist < 5.0,
          "dist=%.2fm node=%s car=%s" % (dist, node_pos, tuple(round(v, 2) for v in car_pos)))
    cte = t.get("cte")
    check("c3 |cte| ~ 1.5 (+-1)", isinstance(cte, (int, float)) and abs(abs(cte) - 1.5) < 1.0, "cte=%r" % cte)
    speed = t.get("speed")
    check("c4 speed ~ 0 after reset", isinstance(speed, (int, float)) and speed < 0.5, "speed=%r" % speed)


def step_d_deterministic_reset(c):
    print("\n-- d. deterministic seeded reset")

    def do_reset():
        c.send({"msg_type": "reset_car",
                "node_min": "10", "node_max": "20", "seed": "7", "lateral_offset": "1.0"})
        return c.settle(0.6)

    r1 = do_reset()
    r2 = do_reset()
    p1 = (r1["pos_x"], r1["pos_y"], r1["pos_z"])
    p2 = (r2["pos_x"], r2["pos_y"], r2["pos_z"])
    dp = math.dist(p1, p2)
    check("d1 same-seed reset pos delta < 0.01", dp < 0.01, "delta=%.5f p1=%s p2=%s" % (dp, p1, p2))


def step_f_drive_and_observe_collisions(c):
    print("\n-- f. collision impulse soft observation (never fails)")
    max_impulse = 0.0
    start = time.monotonic()
    while time.monotonic() - start < 8.0:
        t = time.monotonic() - start
        send_control(c, steering=0.5 * math.sin(t * 1.5), throttle=1.0)
        msg = c.next_telemetry()
        imp = msg.get("collision_impulse", 0.0)
        if isinstance(imp, (int, float)) and not isinstance(imp, bool):
            max_impulse = max(max_impulse, imp)
    info("max collision_impulse over 8s drive = %.3f (observation only)" % max_impulse)


def step_e_deadman(c):
    print("\n-- e. deadman control-timeout brake")
    # legacy plain reset first: exercises the backward-compatible path and
    # gives the acceleration phase a clean, on-track car
    c.send({"msg_type": "reset_car"})
    c.settle(0.6)

    start = time.monotonic()
    last_speed = None
    while time.monotonic() - start < 2.5:
        send_control(c, throttle=0.8)
        msg = c.next_telemetry()
        last_speed = msg.get("speed")
    info("speed after 2.5s throttle 0.8: %r (telemetry units)" % last_speed)

    last_ctrl = time.monotonic()
    stopped_at = None
    min_speed = float("inf")
    speed_at_15 = None
    deadline = last_ctrl + 5.0
    while time.monotonic() < deadline:
        msg = c.next_telemetry()
        now = time.monotonic()
        s = msg.get("speed")
        if not isinstance(s, (int, float)) or isinstance(s, bool):
            continue
        min_speed = min(min_speed, s)
        if speed_at_15 is None and now - last_ctrl >= 1.5:
            speed_at_15 = s
        if s < 0.5 and stopped_at is None:
            stopped_at = now - last_ctrl
    check("e1 speed falls below 0.5 after controls stop (deadman brake)",
          stopped_at is not None,
          "stopped_after=%s min_speed=%.3f speed_at_1.5s=%s (timeout=800ms, 5s window)"
          % ("%.2fs" % stopped_at if stopped_at is not None else "never",
             min_speed, "%.3f" % speed_at_15 if speed_at_15 is not None else "n/a"))


def step_g_still_alive(c):
    print("\n-- g. car still alive")
    t = c.next_telemetry()
    check("g1 telemetry still streaming", t.get("msg_type") == "telemetry",
          "frame_id=%r" % t.get("frame_id"))


def main():
    parser = argparse.ArgumentParser(description="RL P0 protocol smoke test for DonkeySim")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9091)
    args = parser.parse_args()

    print("connecting to %s:%d ..." % (args.host, args.port))
    c = SimClient(args.host, args.port)

    try:
        handshake(c)
        info("car loaded and configured (rl_mode=1, control_timeout_ms=800, cam 160x120)")

        t = step_a_basic_telemetry(c)
        total_nodes = 0
        if t is not None:
            total_nodes = t.get("totalNodes", 0)
        info("totalNodes=%s" % total_nodes)

        step_b_time_scale(c)
        step_c_reset_at_node(c, total_nodes)
        step_d_deterministic_reset(c)
        step_f_drive_and_observe_collisions(c)
        step_e_deadman(c)
        step_g_still_alive(c)
    except (TimeoutError, ConnectionError, OSError, ValueError, KeyError) as e:
        check("harness: unexpected error", False, "%s: %s" % (type(e).__name__, e))
    finally:
        c.close()

    passed = sum(results)
    print("\n==== %d/%d checks passed ====" % (passed, len(results)))
    sys.exit(0 if results and passed == len(results) else 1)


if __name__ == "__main__":
    main()
