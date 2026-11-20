#!/usr/bin/env python3
"""[HBC] 人驾 BC 录制器:订阅窗口 sim 的本地玩家车遥测,(图像, 键盘动作) 落盘。

原理:menu→Manual Drive 下键盘经 JoystickCarControl 直接写车的 requestSteering/
requestTorque;TcpCarHandler 的遥测回显 steering_angle/throttle(GetSteering/
GetThrottle,即当前请求值)。客户端只发 car_config(声明接收)+ cam_config
(JPG/120x160/20fps),**绝不发 control**——驾驶权完全在人,零干扰。

任务零探针:python3 src/hbc_record.py --seconds 120 --probe
正式采集:   python3 src/hbc_record.py --seconds 2100(35 分钟,自动分片落盘)

探针门(硬执行):① 图像 std>20(非灰图)② 输入真的进了日志(steer/throttle 方差非零,
   量程 [-1,1]/[0,1])③ |steer|>0.3 占比合理(≥10%)④ npz 可被 bc_train.py 读取。
"""
import argparse
import base64
import io
import json
import os
import socket
import time

import numpy as np
from PIL import Image


def send(sock, obj):
    sock.sendall((json.dumps(obj) + "\n").encode())


def recv_frames(sock, buf):
    """按 '}' 定界提取 JSON 对象(协议:'{...}' 拼接,可选 '\n' 终止)。"""
    try:
        data = sock.recv(1 << 20)
    except socket.timeout:
        return [], buf
    if not data:
        raise ConnectionError("sim closed")
    buf += data.decode("utf-8", "ignore")
    frames, rest = [], buf
    while True:
        i = rest.find("}")
        if i < 0:
            break
        candidate = rest[: i + 1]
        rest = rest[i + 1:]
        try:
            frames.append(json.loads(candidate))
        except ValueError:
            continue  # 撕裂/半帧,丢弃
    return frames, rest.lstrip("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9091)
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--out", default="outputs/bc_human_data")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--shard", type=int, default=5000)
    args = ap.parse_args()

    if args.probe:
        args.out = "outputs/bc_human_probe"
    os.makedirs(args.out, exist_ok=True)

    s = socket.create_connection(("127.0.0.1", args.port), timeout=5)
    s.settimeout(0.5)
    send(s, {"msg_type": "car_config", "body_style": "donkey", "body_r": "255",
             "body_g": "255", "body_b": "255", "car_name": "Racer Name", "font_size": "100",
             "observe_player": "1"})  # 重绑到本地玩家车(人驾录制)
    send(s, {"msg_type": "cam_config", "img_enc": "JPG", "img_w": "160", "img_h": "120",
             "img_d": "3", "fps": "20"})
    print("[hbc] 已发 car/cam_config——请确认 sim 在 Manual Drive(mountain_track)驾驶中")

    imgs, steers, thrs, t0 = [], [], [], time.time()
    buf, started, n = "", False, 0
    while time.time() - t0 < args.seconds:
        try:
            frames, buf = recv_frames(s, buf)
        except ConnectionError:
            print("[hbc] 连接断开——sim 是否还在?")
            break
        for f in frames:
            mt = f.get("msg_type")
            if mt == "car_loaded":
                started = True
                print("[hbc] car_loaded,遥测开始")
            elif mt == "telemetry" and f.get("image"):
                img = np.asarray(Image.open(io.BytesIO(base64.b64decode(f["image"]))))
                # 人驾桥:echo = h×MaximumSteerAngle(25)/steer_to_angle(16),×16/25 还原
                st = float(f.get("steering_angle", 0)) * (16.0 / 25.0)
                st = max(-1.0, min(1.0, st))
                th = max(0.0, min(1.0, float(f.get("throttle", 0))))
                spd = float(f.get("speed", 0))
                imgs.append(img)
                steers.append(st)
                thrs.append(th)
                n += 1
                if n % 500 == 0:
                    print("[hbc] %d 帧 (%.0f fps) speed=%.1f steer=%.2f thr=%.2f" % (
                        n, n / (time.time() - t0), spd, st, th), flush=True)
                if len(imgs) >= args.shard:
                    _flush(args.out, imgs, steers, thrs)
                    imgs, steers, thrs = [], [], []
    if imgs:
        _flush(args.out, imgs, steers, thrs)
    s.close()
    _card(args.out, probe=args.probe)


def _flush(out, imgs, steers, thrs):
    import glob
    idx = len(glob.glob(os.path.join(out, "bc_*.npz")))
    path = os.path.join(out, "bc_%05d.npz" % idx)
    N = len(imgs)
    np.savez_compressed(path, image=np.stack(imgs),
                        steer=np.array(steers), throttle=np.array(thrs),
                        meta=np.zeros((N, 4), dtype=np.float32))  # 人驾单段,ep_id 由分片隐含
    print("[hbc] 落盘 %s (%d 帧)" % (path, N), flush=True)


def _card(out, probe=False):
    import glob
    imgs_l, st_l, th_l = [], [], []
    for f in sorted(glob.glob(os.path.join(out, "bc_*.npz"))):
        d = np.load(f)
        imgs_l.append(d["image"][:200])
        st_l.append(d["steer"])
        th_l.append(d["throttle"])
    img = np.concatenate(imgs_l)
    st, th = np.concatenate(st_l), np.concatenate(th_l)
    std = float(img.std())
    hard = float((np.abs(st) > 0.3).mean())
    thr_pos = float((th > 0.2).mean())
    hist, _ = np.histogram(th, bins=6, range=(0, 1))
    print("[hbc-card] 帧数=%d 图像std=%.1f(门>20) | steer std=%.3f var非零=%s | "
          "|steer|>0.3=%.1f%%(门≥10%%) | throttle>0.2=%.1f%%" % (
              len(st), std, st.std(), st.std() > 1e-4, hard * 100, thr_pos * 100))
    print("[hbc-card] 油门直方图(6箱[0,1]):", hist.tolist(), " mean=%.3f" % th.mean())
    sheet = None
    from PIL import Image as I
    tail = img[-16:]
    h, w = tail[0].shape[:2]
    sheet = I.new("RGB", (4 * w, 4 * h))
    for i, fr in enumerate(tail):
        sheet.paste(I.fromarray(fr), ((i % 4) * w, (i // 4) * h))
    sheet.save(os.path.join(out, "montage.jpg"), quality=90)
    ok = std > 20 and st.std() > 1e-4 and 1e-4 < th.std() and hard >= 0.10
    if not ok and std > 20 and st.std() <= 1e-4:
        print("[hbc-card] 判定提示:图像有运动但 steer/throttle 全零——大概率还在")
        print("[hbc-card]   主菜单(展示车环绕镜头):请点击 Manual Drive → mountain_track 后再采!")
    print("==== HBC PROBE %s ====" % ("PASS" if ok else "FAIL"))
    assert ok or not probe, "探针门未过,按纪律停线(不许静默放行)"


if __name__ == "__main__":
    main()
