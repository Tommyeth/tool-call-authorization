"""模板因子实验：role (system/user) × deontic force (permissive/conditional/mandatory)。

native vs neutral 的单次对照证明了模板效应巨大（FAR 0.996 vs 0.100），
但一次改了六件事，不能归因。这里只变两个因子，其余逐字不变。

回答：
  * mandatory 措辞是否只整体抬高调用率，还是让模型不再区分授权与非授权？
  * tool spec 放 user turn 是否比放 system 更强？
  * 原生模板的 0.996 是语气、位置，还是两者交互？

J（audit judgment）与模板无关，直接复用已有的 cognition.jsonl，不重跑。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from icaa.analyze import association_test, failure_typing  # noqa: E402
from icaa.modeling import load_config, load_model, resolve_model_cfg  # noqa: E402
from icaa.prompts import DEONTIC, render_factorial_prompt  # noqa: E402
from icaa.rollout import parse_tool_call  # noqa: E402
from icaa.schema import load_actions, load_items, read_jsonl  # noqa: E402

ROLES = ["system", "user"]


@torch.no_grad()
def run_condition(model, tok, device, items, actions, mcfg, role, force, max_new=128):
    recs = []
    for it in tqdm(items, desc=f"{role}-{force}", leave=False):
        p = render_factorial_prompt(it, actions[it.action], tok, mcfg, role, force)
        enc = tok(p, return_tensors="pt").to(device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             temperature=None, top_p=None, pad_token_id=tok.pad_token_id)
        new = gen[0, enc["input_ids"].shape[1]:]
        text = tok.decode(new, skip_special_tokens=False)
        ok, called = parse_tool_call(text, it.action)
        recs.append({"item_id": it.item_id, "executed": ok, "called_tool": called,
                     "completion": text})
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-8b-neutral")
    ap.add_argument("--items", default="data/pairs/pilot_v2.jsonl")
    ap.add_argument("--actions", default="data/actions.yaml")
    ap.add_argument("--cognition", default="runs/llama-8b-v2/cognition.jsonl",
                    help="J 与模板无关，复用即可")
    ap.add_argument("--out-dir", default="runs/factorial-llama")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mcfg = resolve_model_cfg(load_config(), args.model)
    items = load_items(args.items)
    actions = load_actions(args.actions)
    cog = {r["item_id"]: r for r in read_jsonl(args.cognition)}
    model, tok, device = load_model(mcfg)

    I = np.array([it.intent for it in items])
    J = np.array([cog[it.item_id]["c_auth_bin"] for it in items])
    lv = np.array([it.level for it in items])

    rows = []
    for role in ROLES:
        for force in DEONTIC:
            f = out / f"rollout_{role}_{force}.jsonl"
            if f.exists():
                recs = read_jsonl(f)
                print(f"复用 {f.name}")
            else:
                recs = run_condition(model, tok, device, items, actions, mcfg, role, force)
                with f.open("w", encoding="utf-8") as fh:
                    for r in recs:
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            by = {r["item_id"]: r for r in recs}
            P = np.array([by[it.item_id]["executed"] for it in items])
            called = [by[it.item_id]["called_tool"] for it in items]
            scope = np.array([c in (None, it.action) for c, it in zip(called, items)])
            # 强制模板可能逼出不存在的工具，单独计数
            halluc = sum(1 for c, it in zip(called, items)
                         if c is not None and c != it.action)

            st = failure_typing(I, J, P, scope)
            asc = association_test(I, J, P, scope)
            rows.append({
                "role": role, "force": force,
                "FAR": st["false_action_rate"], "MAR": st["missed_action_rate"],
                "n_FA": st["n_false_action"],
                "OR": asc["odds_ratio_C1_vs_C0"], "p": asc["fisher_p"],
                "exec_C0": asc["exec_rate_given_C0"], "exec_C1": asc["exec_rate_given_C1"],
                "halluc_tool": halluc,
                "exec_by_level": {L: float(P[lv == L].mean()) for L in
                                  ["L0", "L1", "L2", "L3", "L4"]},
            })
            r = rows[-1]
            print(f"  {role:6s} {force:11s} FAR={r['FAR']:.3f} MAR={r['MAR']:.3f} "
                  f"OR={r['OR']:.2f} p={r['p']:.4f} halluc={halluc}")

    (out / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))

    print(f"\n{'='*86}\n模板因子表 (model={args.model})\n{'='*86}")
    print(f"{'role':7s} {'force':12s} {'FAR':>6s} {'MAR':>6s} {'OR':>6s} {'p':>8s} "
          f"{'halluc':>7s}   " + " ".join(f"{L:>5s}" for L in ["L0", "L1", "L2", "L3", "L4"]))
    for r in rows:
        print(f"{r['role']:7s} {r['force']:12s} {r['FAR']:6.3f} {r['MAR']:6.3f} "
              f"{r['OR']:6.2f} {r['p']:8.4f} {r['halluc_tool']:7d}   " +
              " ".join(f"{r['exec_by_level'][L]:5.2f}" for L in ["L0", "L1", "L2", "L3", "L4"]))

    print("\n主效应（FAR 均值）")
    for role in ROLES:
        v = [r["FAR"] for r in rows if r["role"] == role]
        print(f"  role={role:7s} {np.mean(v):.3f}")
    for force in DEONTIC:
        v = [r["FAR"] for r in rows if r["force"] == force]
        print(f"  force={force:12s} {np.mean(v):.3f}")
    print("\n判读：若 OR 在六个条件下相对稳定而 FAR 大幅变化，")
    print("      说明模板改的是 baseline call propensity（截距），不是授权敏感度（斜率）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
