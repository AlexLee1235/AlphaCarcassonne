#!/usr/bin/env python3
"""ladder.py -- 用固定錨點建立 Elo 階梯。

對局引擎直接用 build/examples/alpha_zero_torch_game_example，它已經做好了
成對同牌序對局(num_games 必須偶數，每一對用同一副牌序打兩腿、交換先後手)。
這裡補的是它沒有的三件事：排程、增量快取、Bradley-Terry 擬合。

錨點：random-rollout MCTS。它不會學習，所以強度只是 sims 的函數，
      永遠可重現。rollout-800 定義為 0 Elo —— 這正是四個 run 的 eval
      一直在對打的對手，所以階梯跟既有的 eval 數字相容。

用法:
    python3 ladder.py --config ladder_config.json          # 跑 + 擬合
    python3 ladder.py --config ... --dry-run               # 只印指令
    python3 ladder.py --config ... --fit-only              # 只用快取重新擬合

config 範例見 ladder_config.example.json。
"""
import argparse, itertools, json, math, os, re, subprocess, sys
import numpy as np

SCALE = math.log(10) / 400.0          # Elo -> logit


# ---------------------------------------------------------------- 對局

def build_cmd(binary, game, a, b, n_games, seed, opts):
    """a, b 是 participant dict。a 當 agent0、b 當 agent1。"""
    cmd = [binary, f"--game={game}", f"--num_games={n_games}",
           f"--num_workers={opts['workers']}", f"--seed={seed}",
           "--quiet=true", "--verbose=false",
           f"--az_threads={opts['az_threads']}",
           f"--az_cache_size={opts['az_cache']}",
           f"--az_value_is_current_player={'true' if opts['value_is_current_player'] else 'false'}",
           f"--rollout_count={opts['rollout_count']}"]
    for tag, p in (("p1", a), ("p2", b)):
        cmd.append(f"--{tag}_type={p['type']}")
        cmd.append(f"--{tag}_max_simulations={p.get('sims', opts['sims'])}")
        cmd.append(f"--{tag}_uct_c={p.get('uct_c', opts['uct_c'])}")
        cmd.append(f"--{tag}_solve={'true' if p.get('solve', p['type'] != 'az') else 'false'}")
        if p["type"] == "az":
            cmd.append(f"--{tag}_az_path={p['path']}")
            cmd.append(f"--{tag}_az_checkpoint={p['checkpoint']}")
            cmd.append(f"--{tag}_az_device={p.get('device', opts['device'])}")
            cmd.append(f"--{tag}_az_graph_def={p.get('graph_def', 'vpnet.pb')}")
    return cmd


WINS_RE = re.compile(r"Overall wins:\s*(\d+),\s*(\d+)")
DRAWS_RE = re.compile(r"Overall draws:\s*(\d+)")


def run_match(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = r.stdout + r.stderr
    w, d = WINS_RE.search(out), DRAWS_RE.search(out)
    if not w or not d:
        sys.stderr.write(out[-2000:] + "\n")
        raise RuntimeError("解析不到 Overall wins / draws，對局可能失敗")
    return int(w.group(1)), int(w.group(2)), int(d.group(1))


# ---------------------------------------------------------------- 擬合

def fit_elo(names, matches, anchor, ridge=1e-6):
    """matches: {(i,j): (wins_i, wins_j, draws)}。回傳 (elo, se)。

    Bradley-Terry / Elo 最大概似，錨點固定在 0，用 Newton 解。
    和局算 0.5 分(和局率 ~3%，不值得上 Rao-Kupper)。
    """
    m = len(names)
    S = np.zeros((m, m))      # S[i,j] = i 對 j 的得分
    N = np.zeros((m, m))
    for (i, j), (wi, wj, d) in matches.items():
        n = wi + wj + d
        if n == 0:
            continue
        N[i, j] += n; N[j, i] += n
        S[i, j] += wi + 0.5 * d
        S[j, i] += wj + 0.5 * d

    free = [k for k in range(m) if k != anchor]
    r = np.zeros(m)
    for _ in range(200):
        D = r[:, None] - r[None, :]
        E = 1.0 / (1.0 + np.exp(-SCALE * D))          # i 對 j 的期望得分
        g = SCALE * (S - N * E).sum(axis=1)
        W = N * E * (1 - E) * SCALE ** 2
        H = np.diag(W.sum(axis=1)) - W                 # 負 Hessian
        Hf = H[np.ix_(free, free)] + ridge * np.eye(len(free))
        step = np.linalg.solve(Hf, g[free])
        r[free] += step
        if np.max(np.abs(step)) < 1e-9:
            break
    cov = np.linalg.inv(H[np.ix_(free, free)] + ridge * np.eye(len(free)))
    se = np.zeros(m)
    se[free] = np.sqrt(np.diag(cov))
    return r, se


# ---------------------------------------------------------------- 主程式

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", default="results/ladder_matches.jsonl")
    ap.add_argument("--out", default="results/ladder.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fit-only", action="store_true")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    P = cfg["participants"]
    names = [p["name"] for p in P]
    idx = {n: i for i, n in enumerate(names)}
    opts = cfg["options"]
    anchor = idx[cfg["anchor"]]

    os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
    cached = {}
    if os.path.exists(args.cache):
        for line in open(args.cache):
            rec = json.loads(line)
            cached[(rec["a"], rec["b"], rec["n"], rec["seed"])] = rec

    matches = {}
    todo = []
    for a, b in itertools.combinations(range(len(P)), 2):
        pa, pb = P[a], P[b]
        # 同一個 process 要同時載入兩個網路 -> 架構必須相同
        if pa.get("arch") and pb.get("arch") and pa["arch"] != pb["arch"]:
            sys.stderr.write(f"skip {pa['name']} vs {pb['name']}: 架構不同({pa['arch']} / {pb['arch']}),"
                             f"同一個 binary 建不出兩種 head\n")
            continue
        n, seed = opts["games_per_pair"], opts["seed"]
        key = (pa["name"], pb["name"], n, seed)
        if key in cached:
            rec = cached[key]
            matches[(a, b)] = (rec["wa"], rec["wb"], rec["draws"])
        else:
            todo.append((a, b, key))

    if args.fit_only and todo:
        sys.stderr.write(f"--fit-only:還有 {len(todo)} 組沒跑,先用已有的擬合\n")
    elif todo:
        for k, (a, b, key) in enumerate(todo, 1):
            cmd = build_cmd(opts["binary"], opts["game"], P[a], P[b],
                            opts["games_per_pair"], opts["seed"], opts)
            print(f"[{k}/{len(todo)}] {P[a]['name']}  vs  {P[b]['name']}", flush=True)
            if args.dry_run:
                print("   " + " ".join(cmd)); continue
            wa, wb, d = run_match(cmd)
            print(f"   {wa} - {wb} (draw {d})", flush=True)
            matches[(a, b)] = (wa, wb, d)
            with open(args.cache, "a") as f:
                f.write(json.dumps({"a": key[0], "b": key[1], "n": key[2], "seed": key[3],
                                    "wa": wa, "wb": wb, "draws": d}) + "\n")
        if args.dry_run:
            return

    if not matches:
        sys.stderr.write("沒有任何對局結果\n"); return

    elo, se = fit_elo(names, matches, anchor)
    order = np.argsort(-elo)
    print(f"\n=== Elo ladder (錨點 {names[anchor]} := 0, {opts['sims']} sims) ===")
    print(f"{'':<22}{'Elo':>7}{'95% CI':>16}   對局")
    for i in order:
        played = sum(v[0] + v[1] + v[2] for (x, y), v in matches.items() if x == i or y == i)
        ci = "" if i == anchor else f"±{1.96*se[i]:.0f}"
        print(f"{names[i]:<22}{elo[i]:>+7.0f}{ci:>16}   {played}")

    json.dump({"names": names, "elo": elo.tolist(), "se": se.tolist(),
               "anchor": names[anchor], "sims": opts["sims"],
               "matches": {f"{names[a]}|{names[b]}": v for (a, b), v in matches.items()}},
              open(args.out, "w"), indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
