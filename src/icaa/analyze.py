"""Failure typing（Fig.1）、层轨迹（Fig.2）、collapse 检查。

Fig.1 就是 go/no-go 判据本身：在所有 "不该执行却执行了" 的样本里，
模型自述 C=0（知道不该）的占多少。这一比例是整篇论文科学问题成立与否的开关。
"""

from __future__ import annotations

import math

import numpy as np

INTENT_FAILURE = "intent_failure"      # I=0, C=1, E=1：模型误解了用户
KNOWING_DOING = "knowing_doing"        # I=0, C=0, E=1：知道不该做，还是做了


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def failure_typing(
    intent: np.ndarray, c_bin: np.ndarray, executed: np.ndarray, in_scope: np.ndarray
) -> dict:
    """in_scope: 排除掉调用了**别的** tool 的样本（那是 selection 错误，不在本文范围）。"""
    false_action = in_scope & (intent == 0) & (executed == 1)
    missed_action = in_scope & (intent == 1) & (executed == 0)

    n_fa = int(false_action.sum())
    n_kd = int((false_action & (c_bin == 0)).sum())
    n_if = int((false_action & (c_bin == 1)).sum())

    lo, hi = wilson_ci(n_kd, n_fa)
    return {
        "n_in_scope": int(in_scope.sum()),
        "n_should_not_act": int((in_scope & (intent == 0)).sum()),
        "n_false_action": n_fa,
        "n_missed_action": int(missed_action.sum()),
        "false_action_rate": n_fa / max(1, int((in_scope & (intent == 0)).sum())),
        "missed_action_rate": int(missed_action.sum())
        / max(1, int((in_scope & (intent == 1)).sum())),
        "n_knowing_doing": n_kd,
        "n_intent_failure": n_if,
        "knowing_doing_share": n_kd / n_fa if n_fa else float("nan"),
        "knowing_doing_ci95": (lo, hi),
        "mask_knowing_doing": false_action & (c_bin == 0),
        "mask_intent_failure": false_action & (c_bin == 1),
        "mask_correct_refusal": in_scope & (intent == 0) & (executed == 0),
    }


def association_test(intent, c_bin, executed, in_scope) -> dict:
    """在 I=0 内部检验 C 是否真的预测执行。**这是比 knowing_doing_share 更硬的判据。**

    单看 share 会被 C 的边际分布骗：若 C 在 90% 的 should-not-act 上都说 "no"，
    那么误执行里自然有 ~90% 落进 knowing-doing 桶，分桶不携带任何信息。
    正确的问法是 2x2：C=0 与 C=1 两组的执行率是否不同。
    """
    sn = in_scope & (intent == 0)
    a = int((sn & (c_bin == 0) & (executed == 1)).sum())
    b = int((sn & (c_bin == 0) & (executed == 0)).sum())
    c = int((sn & (c_bin == 1) & (executed == 1)).sum())
    d = int((sn & (c_bin == 1) & (executed == 0)).sum())

    rate_c0 = a / (a + b) if a + b else float("nan")
    rate_c1 = c / (c + d) if c + d else float("nan")
    odds = (c * b) / (d * a) if d and a else float("nan")

    p = float("nan")
    try:
        from scipy.stats import fisher_exact
        p = float(fisher_exact([[a, b], [c, d]])[1])
    except Exception:
        pass

    return {
        "table": {"C0_exec": a, "C0_noexec": b, "C1_exec": c, "C1_noexec": d},
        "exec_rate_given_C0": rate_c0,
        "exec_rate_given_C1": rate_c1,
        "odds_ratio_C1_vs_C0": odds,
        "fisher_p": p,
        "c_marginal_P_C0_given_I0": float((c_bin[sn] == 0).mean()) if sn.any() else float("nan"),
        "informative": bool(p == p and p < 0.05),
    }


def go_no_go(stats: dict, min_false_actions: int = 30) -> tuple[str, str]:
    """把上一轮定死的判据写成代码，避免事后挪门槛。"""
    n = stats["n_false_action"]
    if n < min_false_actions:
        return "INCONCLUSIVE", (
            f"误执行样本仅 {n} 条（需 ≥{min_false_actions}）。"
            "先扩数据或换更容易出错的 level/模型，不要在这个样本量上下结论。"
        )
    share = stats["knowing_doing_share"]
    lo, hi = stats["knowing_doing_ci95"]
    if hi < 0.15:
        return "NO-GO", (
            f"knowing-doing 占比 {share:.1%}（CI 上界 {hi:.1%} < 15%）。"
            "误执行几乎全部在 intent 表征阶段就已经错了，三段分解不成立，"
            "应转向 intent calibration（§23）。"
        )
    if lo > 0.50:
        return "GO-BUT-VERIFY", (
            f"knowing-doing 占比 {share:.1%}（CI 下界 {lo:.1%} > 50%）。"
            "比例偏高，优先排查 C-elicitation 的 sycophancy：审计者视角下模型"
            "可能倾向一律答 no。先看 02_elicit 的 variant_agreement 与 C 在 L0 上的准确率。"
        )
    return "GO", (
        f"knowing-doing 占比 {share:.1%}（95% CI [{lo:.1%}, {hi:.1%}]）。"
        "两类失败共存，科学问题成立，可以推进 Fig.2 / Fig.3。"
    )


def collapse_check(c_direction: np.ndarray, action_dir: np.ndarray) -> np.ndarray:
    """逐层计算 C-probe 方向与 unembedding action 方向的 |cos|。

    这是上一轮警告的那条红线：若中后层 |cos| 普遍偏高（经验阈值 ~0.5），
    说明 C 和 A 只是同一个特征在不同层的读数，"两类失败"会塌缩成
    "一个信号在第 k 层翻转"，Fig.3 的差异化 steering 必然失败。
    """
    a = action_dir / (np.linalg.norm(action_dir) + 1e-8)
    W = c_direction / (np.linalg.norm(c_direction, axis=1, keepdims=True) + 1e-8)
    return np.abs(W @ a)


def layer_trajectory(A: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Fig.2：按 failure type 分组的 A_l 均值曲线。"""
    return {name: A[m].mean(0) if m.any() else np.full(A.shape[1], np.nan)
            for name, m in masks.items()}


def divergence_layer(curve_a: np.ndarray, curve_b: np.ndarray, thresh: float = 0.1) -> int:
    """两条均值轨迹首次持续分离的层。

    **注意尺度陷阱**：若 curve 是 logit-lens 的原始概率，中前层对任何特定 token
    的概率都接近 0，绝对差不可能超过 0.1，结果恒为最后一层——那是测量假象，
    不是"没有信号"。主分析请用 separability_curve()，它是尺度无关的。
    """
    diff = np.abs(curve_a - curve_b)
    for l in range(len(diff)):
        if np.all(diff[l:] > thresh):
            return l
    return -1


def separability_curve(X: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    """逐层计算 X_l 区分 A 组与 B 组的 AUROC。尺度无关，取代绝对差。

    X: (N, L+1)，可传 A（概率）或 A_margin（logit 差）——AUROC 只看序，两者等价。
    主用途：knowing_doing vs correct_refusal（同 I、同 C，只有 E 不同），
    曲线抬升的那一层就是"动作决策成形"的位置。
    """
    from sklearn.metrics import roc_auc_score

    y = np.concatenate([np.ones(int(mask_a.sum())), np.zeros(int(mask_b.sum()))])
    if len(set(y.tolist())) < 2:
        return np.full(X.shape[1], np.nan)
    out = []
    for l in range(X.shape[1]):
        s = np.concatenate([X[mask_a, l], X[mask_b, l]])
        out.append(float(roc_auc_score(y, s)) if np.ptp(s) > 0 else 0.5)
    return np.array(out)


def onset_layer(curve: np.ndarray, thresh: float = 0.70) -> int:
    """曲线首次并持续越过阈值的层。用于报告"动作倾向从第几层开始可分"。"""
    for l in range(len(curve)):
        seg = curve[l:]
        if np.all(seg[~np.isnan(seg)] > thresh):
            return l
    return -1
