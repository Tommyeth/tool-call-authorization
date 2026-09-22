"""Layer-wise linear probe。

两条硬性方法学约束：

1. **CV 必须按 pair_id 分组。** minimal pair 的两端只差措辞，若一端在 train、
   另一端在 test，probe 会靠场景记忆刷分，AUROC 虚高十几个点。
2. **主协议是 L0/L4 训、L1–L3 测。** 在全部 level 上做随机 CV 只能说明
   "极端例子可分"；论文要回答的是"意图信号在歧义区是否仍然线性可读"。
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def _fit(X_tr, y_tr, X_te, C: float = 1.0):
    sc = StandardScaler().fit(X_tr)
    # hidden state probe 恒是 p >> n（3584 维 vs 百来个样本）。lbfgs 在这种形状下
    # 收敛极慢——29 层 × 6 折 = 174 次拟合要跑十几分钟。liblinear 的对偶形式解的是
    # 同一个 L2 目标，但复杂度随 n 而非 p，快一个量级，结果等价。
    if X_tr.shape[1] > X_tr.shape[0]:
        clf = LogisticRegression(C=C, solver="liblinear", dual=True, max_iter=5000)
    else:
        clf = LogisticRegression(C=C, max_iter=2000)
    clf.fit(sc.transform(X_tr), y_tr)
    return clf.predict_proba(sc.transform(X_te))[:, 1], clf, sc


def probe_layerwise_holdout(
    H: np.ndarray, y: np.ndarray, train_mask: np.ndarray, C: float = 1.0
) -> dict:
    """主协议：train_mask 指定的样本（L0/L4）训练，其余（L1–L3）测试。"""
    n_layers = H.shape[1]
    te = ~train_mask
    aucs, preds, dirs = [], [], []
    for l in range(n_layers):
        X = H[:, l, :].astype(np.float32)
        p, clf, sc = _fit(X[train_mask], y[train_mask], X[te], C)
        aucs.append(_safe_auc(y[te], p))
        full = np.full(len(y), np.nan)
        full[te] = p
        preds.append(full)
        # 反标准化回原空间的方向，供 collapse_check 用
        dirs.append((clf.coef_[0] / np.maximum(sc.scale_, 1e-8)).astype(np.float32))
    return {
        "auc": np.array(aucs),
        "pred": np.stack(preds, 1),      # (N, L+1)，train 部分为 nan
        "direction": np.stack(dirs),     # (L+1, d)
        "test_mask": te,
    }


def probe_layerwise_cv(
    H: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5, C: float = 1.0
) -> dict:
    """对照协议：全 level 上按 pair_id 分组的 GroupKFold。"""
    n_layers = H.shape[1]
    gkf = GroupKFold(n_splits=min(n_splits, len(set(groups))))
    preds = np.zeros((len(y), n_layers))
    for tr, te in gkf.split(H[:, 0, :], y, groups):
        for l in range(n_layers):
            X = H[:, l, :].astype(np.float32)
            preds[te, l] = _fit(X[tr], y[tr], X[te], C)[0]
    return {
        "auc": np.array([_safe_auc(y, preds[:, l]) for l in range(n_layers)]),
        "pred": preds,
    }


def probe_concat_baseline(
    H: np.ndarray, y: np.ndarray, groups: np.ndarray, layers: list[int] | None = None,
    n_splits: int = 5, C: float = 1.0, max_layers: int = 8,
) -> dict:
    """When2Tool 风格：把多层 hidden state 拼成 H=[h1;...;hL] 再训一个 probe。

    在这里的作用是 sanity check ——如果这个 baseline 的 AUROC 明显低于
    When2Tool 报告的量级，说明抽取管线（位置、template、dtype）有问题，
    应该先修管线再谈任何 finding。

    拼接层数上限 max_layers：全部 22 层 × 3584 = 78k 维时 lbfgs 要跑十几分钟，
    而相邻层高度冗余。等距抽 8 层足以体现"多层拼接"的效果，代价降一个量级。
    """
    idx = layers if layers is not None else list(range(H.shape[1]))
    if len(idx) > max_layers:  # 等距下采样，保留首尾
        idx = [idx[i] for i in np.linspace(0, len(idx) - 1, max_layers).round().astype(int)]
    X = H[:, idx, :].reshape(len(y), -1).astype(np.float32)
    gkf = GroupKFold(n_splits=min(n_splits, len(set(groups))))
    pred = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        pred[te] = _fit(X[tr], y[tr], X[te], C)[0]
    return {"auc": _safe_auc(y, pred), "pred": pred, "layers": idx}


def _safe_auc(y, p) -> float:
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))
