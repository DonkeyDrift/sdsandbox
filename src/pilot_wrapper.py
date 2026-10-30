#!/usr/bin/env python3
"""mycar pilot(tflite) 包装器:供 rl_eval 以新门口径评测,并给蒸馏/正则提供教师接口。

管线表征(与 mycar 部署对齐或显式声明差异,证据源见 journal [PILOT] 任务零):
- 输入: HWC uint8, 模型原生 (240,320,3); normalize = /255.0 (donkeycar.utils.normalize_image)
- 输出: (steer, throttle) 连续值,线性头无界; DriveMode AI_THROTTLE_MULT=1.0
- 环路后处理(可选 --mycar-post,默认开以贴近部署):
    SteeringSmoother: EMA alpha=0.35, slew 8.0 满量程/s
    Throttle EMA:     alpha=0.5,  slew 6.0 满量程/s
  时间感知:用真实 dt,与 60Hz 部署在 α 语义上一致(20Hz 评测时时间常数不变)
- 无 ROI crop(mycar TRANSFORMATIONS=[]);无 AI_LAUNCH;无 STOP_SIGN/ThrottleFilter
- 控制频率: mycar 60Hz; rl_eval 默认异步 ~20Hz——差异显式声明,评测可用 --cam-fps 60 收窄
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np


class _EmaSmoother:
    """与 donkeydrifter.parts.steering_smoother.EmaSmoother 同语义(时间感知)。"""

    def __init__(self, alpha=0.35, slew_per_sec=8.0):
        self.alpha = float(alpha)
        self.slew_per_sec = float(slew_per_sec)
        self.value = 0.0
        self.initialized = False
        self._last_ts = None

    def run(self, value: float) -> float:
        if value is None:
            return value
        now = time.time()
        if not self.initialized:
            self.value = float(value)
            self.initialized = True
            self._last_ts = now
            return self.value
        dt = now - self._last_ts
        self._last_ts = now
        if self.alpha >= 1.0:
            v = float(value)
        else:
            v = self.alpha * float(value) + (1.0 - self.alpha) * self.value
        if self.slew_per_sec > 0.0:
            max_delta = self.slew_per_sec * max(dt, 1e-3)
            delta = v - self.value
            if delta > max_delta:
                v = self.value + max_delta
            elif delta < -max_delta:
                v = self.value - max_delta
        self.value = v
        return v

    def reset(self):
        self.initialized = False
        self.value = 0.0
        self._last_ts = None


class PilotPolicy:
    """SB3-like predict 接口的 pilot 策略。"""

    def __init__(
        self,
        model_path: str,
        mycar_post: bool = True,
        throttle_mult: float = 1.0,
        clamp: Optional[Tuple[float, float, float, float]] = None,
        # clamp=(steer_lo, steer_hi, thr_lo, thr_hi), None=不夹(mycar 线性头原语义)
    ):
        import tensorflow as tf  # 延迟导入

        self._tf = tf
        self.model_path = model_path
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        sig = self.interpreter.get_signature_list()
        assert "serving_default" in sig, "缺少 serving_default signature: %r" % sig
        self.runner = self.interpreter.get_signature_runner()
        self.input_keys = sig["serving_default"]["inputs"]
        self.output_keys = sig["serving_default"]["outputs"]
        details = self.interpreter.get_input_details()
        # img_in shape
        self.input_shape = None
        for d in details:
            name = d["name"]
            if "img_in" in name or d["index"] == 0:
                self.input_shape = tuple(int(x) for x in d["shape"])  # (1,H,W,C)
                break
        assert self.input_shape is not None, "找不到 img_in 输入"
        self.input_dtype = details[0]["dtype"]
        self.mycar_post = bool(mycar_post)
        self.throttle_mult = float(throttle_mult)
        self.clamp = clamp
        self._steer_ema = _EmaSmoother(alpha=0.35, slew_per_sec=8.0)
        self._thr_ema = _EmaSmoother(alpha=0.5, slew_per_sec=6.0)

    # ---- 元数据(探针用) ----
    @property
    def meta(self):
        return {
            "model_path": self.model_path,
            "input_keys": list(self.input_keys),
            "output_keys": list(self.output_keys),
            "input_shape": list(self.input_shape),
            "input_dtype": str(np.dtype(self.input_dtype).name),
            "mycar_post": self.mycar_post,
            "throttle_mult": self.throttle_mult,
            "note": "normalize=/255; no ROI crop; steer EMA a=0.35 slew=8; thr EMA a=0.5 slew=6",
        }

    def reset(self):
        self._steer_ema.reset()
        self._thr_ema.reset()

    def _prepare_img(self, img: np.ndarray) -> np.ndarray:
        """img: HWC uint8 或 float。对齐模型 (1,H,W,C) float32 /255。"""
        arr = np.asarray(img)
        if arr.ndim == 3:
            arr = arr[None, ...]
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        # 已是 [0,1] 时不再除;uint8 路径已在上面转 float 但值域仍 0..255
        if arr.max() > 1.5:
            arr = arr / 255.0
        # 空间尺寸不符则双线性缩放(评测用低分辨率相机时)
        _, h, w, c = self.input_shape
        if arr.shape[1] != h or arr.shape[2] != w:
            # 用 PIL 避免拉 torch 依赖进本模块
            from PIL import Image

            out = np.empty((arr.shape[0], h, w, c), dtype=np.float32)
            for i in range(arr.shape[0]):
                u8 = (np.clip(arr[i], 0, 1) * 255.0).astype(np.uint8)
                out[i] = np.asarray(
                    Image.fromarray(u8).resize((w, h), Image.BILINEAR),
                    dtype=np.float32,
                ) / 255.0
            arr = out
        return np.ascontiguousarray(arr, dtype=np.float32)

    def infer_raw(self, img: np.ndarray) -> Tuple[float, float]:
        """裸模型输出(无 EMA/无 mult)。蒸馏/一致度用。"""
        x = self._prepare_img(img)
        key = self.input_keys[0]
        out = self.runner(**{key: x})
        # output_keys 顺序: n_outputs0(steer), n_outputs1(throttle) —— 从 meta 确认
        vals = [float(np.asarray(out[k]).reshape(-1)[0]) for k in self.output_keys]
        # 按名字优先;若顺序反了靠名字兜底
        if len(self.output_keys) >= 2:
            by_name = {}
            for k in self.output_keys:
                kl = k.lower()
                if "0" in kl or "steer" in kl or "angle" in kl:
                    by_name["steer"] = float(np.asarray(out[k]).reshape(-1)[0])
                elif "1" in kl or "thr" in kl:
                    by_name["thr"] = float(np.asarray(out[k]).reshape(-1)[0])
            if "steer" in by_name and "thr" in by_name:
                return by_name["steer"], by_name["thr"]
        return vals[0], vals[1] if len(vals) > 1 else 0.0

    def predict(self, obs, deterministic: bool = True):
        """SB3 风格:obs=(H,W,3) uint8 或 (1,H,W,3)。返回 (action, None)。
        action = [steer, throttle] float32,范围约定 [-1,1]x[0,1](线性头可能越界,可选 clamp)。"""
        img = obs[0] if isinstance(obs, np.ndarray) and obs.ndim == 4 else obs
        steer, thr = self.infer_raw(img)
        thr = thr * self.throttle_mult
        if self.mycar_post:
            steer = self._steer_ema.run(steer)
            thr = self._thr_ema.run(thr)
        if self.clamp is not None:
            lo_s, hi_s, lo_t, hi_t = self.clamp
            steer = float(np.clip(steer, lo_s, hi_s))
            thr = float(np.clip(thr, lo_t, hi_t))
        action = np.array([steer, thr], dtype=np.float32)
        return action, None


def load_pilot(model_path: str, **kwargs) -> PilotPolicy:
    return PilotPolicy(model_path, **kwargs)
