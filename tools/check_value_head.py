#!/usr/bin/env python3
"""check_value_head.py -- value head 的健康檢查

自動判斷 checkpoint 是哪一種 value head：

舊架構 (1 filter -> flatten(H*W) -> Linear)，回答三個問題：
  1. ReLU 有沒有在砍掉格子？        -> beta / |gamma|，要 >> 1 才代表沒砍
  2. head 有沒有在做空間求和？      -> cos(有效權重, 全1向量)，要接近 1.0
  3. 每個位置的權重有沒有被訓練到？ -> W1 per-position std vs 占用率

  checkpoint-193 (nn_width=32, nn_depth=8) 的實測：
    beta/|gamma| = 0.07     -> P(ReLU 前 < 0) ~= 47%，將近一半的格子被截斷
    cos(eff, 1)  = 0.048    -> 跟均勻求和幾乎正交
    占用率 <=5% 的 93 格 per-pos std = 0.0345（初始化是 0.0385）-> 沒被訓練過
    corr(占用率, per-pos std) = +0.607

新架構 (32 filters -> global mean ++ max pooling -> Linear)，求和與位置無關已被
硬編碼，上面 (2)(3) 失去意義。改看：
  1. value_conv 各通道的 BN gamma：大部分塌到 0 表示通道用不完，可以減
  2. 各通道被讀出的程度（W1 在 mean / max 兩半的欄範數）與線性化的正負號：
     有號求和需要「加分」與「扣分」兩種通道同時存在

用法:
    ./diag_occupancy            # 產生 occupancy.csv（舊架構的 (3) 需要，可省略）
    python3 readckpt.py checkpoint-193.pt ck.npz
    python3 check_value_head.py ck.npz [occupancy.csv]
"""
import os
import sys
from math import erf, sqrt

import numpy as np

COLLAPSED_GAMMA = 0.1   # |gamma| 小於「中位數 x 這個比例」視為塌掉的通道


def phi(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def head_prefix(d):
    keys = [k for k in d.files if k.endswith('value_conv.weight')]
    if not keys:
        sys.exit('找不到 value_conv.weight —— 這不是 resnet 的 checkpoint？')
    return keys[0][:-len('value_conv.weight')]


def flatten_head_checks(d, P, occ_path):
    g = d[P + 'value_batch_norm.weight'][0]
    b = d[P + 'value_batch_norm.bias'][0]
    mu = d[P + 'value_batch_norm.running_mean'][0]
    var = d[P + 'value_batch_norm.running_var'][0]
    W1 = d[P + 'value_linear_1.weight']            # (hidden, H*W)
    W2 = d[P + 'value_linear_2.weight'][0]

    print('=== (1) ReLU 有沒有在砍掉格子 ===')
    print(f'  gamma={g:+.4f}  beta={b:+.4f}  running_mean={mu:+.4f}  running_std={np.sqrt(var):.4f}')
    print(f'  BN 後的場: mean=beta={b:+.4f}, std=|gamma|={abs(g):.4f}  ->  beta/|gamma| = {b/abs(g):+.3f}')
    print(f'  推估 P(ReLU 前 < 0) = {phi(-b/abs(g))*100:.1f}%   (要讓 ReLU 失效需要 beta/|gamma| >> 1)')

    print('\n=== (2) 有效的每格權重 (線性區近似 W2 @ W1) ===')
    eff = W2 @ W1
    n = eff.size
    side = int(round(np.sqrt(n)))
    cos = eff.sum() / (np.linalg.norm(eff) * np.sqrt(n))
    print(f'  mean={eff.mean():+.4f}  std={eff.std():.4f}  std/|mean|={eff.std()/abs(eff.mean()):.2f}')
    print(f'  與均勻求和的 cosine = {cos:+.4f}   (1.0 = 完美的平移不變求和)')
    print(f'  範圍 {eff.min():+.4f} ~ {eff.max():+.4f}')
    if side * side == n:
        print(f'  有效權重 {side}x{side} (x1000):')
        for row in eff.reshape(side, side):
            print('   ', ' '.join(f'{v*1000:7.1f}' for v in row))

    if not os.path.exists(occ_path):
        print(f'\n(找不到 {occ_path}，跳過 (3)。先跑 ./diag_occupancy)')
        return
    occ = np.loadtxt(occ_path, delimiter=',').ravel()
    if occ.size != n:
        print(f'\n({occ_path} 大小 {occ.size} 與 head 的 {n} 不符，跳過 (3))')
        return

    print('\n=== (3) 每個位置的訓練痕跡 vs 該格被使用的頻率 ===')
    per_pos = W1.std(axis=0)
    fan_in = W1.shape[1]
    init_std = (1 / np.sqrt(fan_in)) / np.sqrt(3)   # libtorch Linear: U(-1/sqrt(fan_in), +1/sqrt(fan_in))
    print(f'  初始化 std ~= {init_std:.4f}')
    for lo, hi, lab in [(50, 101, '占用>=50%'), (20, 50, '占用20-50%'),
                        (5, 20, '占用5-20%'), (-1, 5, '占用<=5%')]:
        m = (occ > lo) & (occ <= hi)
        if not m.any():
            continue
        print(f'  {lab:12s} n={m.sum():3d}   mean|eff|={np.abs(eff[m]).mean():.4f}   '
              f'W1 per-pos std={per_pos[m].mean():.4f}')
    print(f'  corr(占用率, W1 per-pos std) = {np.corrcoef(occ, per_pos)[0,1]:+.3f}   '
          f'(不含 W2，不受線性區近似影響)')
    print(f'  corr(占用率, |有效權重|)      = {np.corrcoef(occ, np.abs(eff))[0,1]:+.3f}')


def pooled_head_checks(d, P):
    g = d[P + 'value_batch_norm.weight']
    b = d[P + 'value_batch_norm.bias']
    W1 = d[P + 'value_linear_1.weight']            # (hidden, 2C): [mean 的 C 欄 | max 的 C 欄]
    W2 = d[P + 'value_linear_2.weight'][0]
    C = g.size

    print(f'=== value head: {C} filters -> global mean ++ max pooling -> '
          f'Linear({W1.shape[1]}->{W1.shape[0]}) -> Linear({W2.size}->1) ===')
    print('    (位置相依參數 = 0；舊架構的「cos(eff,1)」與「每格權重 vs 占用率」不再適用)\n')

    print('=== (1) value_conv 各通道的 BN gamma：通道有沒有塌掉 ===')
    ag = np.abs(g)
    med = np.median(ag)
    collapsed = ag < COLLAPSED_GAMMA * med
    print(f'  |gamma| min={ag.min():.4f}  median={med:.4f}  max={ag.max():.4f}')
    print(f'  塌掉的通道 (|gamma| < {COLLAPSED_GAMMA} x median): {collapsed.sum()}/{C}'
          + ('   <- 大部分塌掉：通道用不完，可以減' if collapsed.sum() > C // 2 else ''))
    cut = np.array([phi(-bb / aa) if aa > 0 else 1.0 for aa, bb in zip(ag, b)])
    print(f'  推估 P(ReLU 前 < 0) 平均 = {cut.mean()*100:.1f}%   '
          f'(多通道下這是正常的：A=max(0,s)、B=max(0,-s) 各砍一半)')

    print('\n=== (2) 各通道怎麼被讀出 ===')
    col_mean = np.linalg.norm(W1[:, :C], axis=0)
    col_max = np.linalg.norm(W1[:, C:], axis=0)
    eff = W2 @ W1                                   # 線性區近似，(2C,)
    eff_mean, eff_max = eff[:C], eff[C:]
    fan_in = W1.shape[1]
    init_col = np.sqrt(W1.shape[0] / (3 * fan_in))  # libtorch Linear 初始化時每欄的期望範數
    print(f'  W1 每欄範數初始化 ~= {init_col:.4f}')
    print(f'  mean 分支: 平均欄範數 {col_mean.mean():.4f}   有效權重 正/負 = '
          f'{int((eff_mean > 0).sum())}/{int((eff_mean < 0).sum())}')
    print(f'  max  分支: 平均欄範數 {col_max.mean():.4f}   有效權重 正/負 = '
          f'{int((eff_max > 0).sum())}/{int((eff_max < 0).sum())}')
    if (eff_mean > 0).all() or (eff_mean < 0).all():
        print('  !! mean 分支的通道全部同號 —— 沒有「加分／扣分」兩種通道，還不能表示有號求和')
    print(f'\n  {"ch":>3} {"gamma":>8} {"beta":>8} {"P(<0)":>6} {"|W1 mean|":>10} '
          f'{"|W1 max|":>9} {"eff mean":>9} {"eff max":>9}')
    for c in np.argsort(-ag):
        print(f'  {c:3d} {g[c]:+8.4f} {b[c]:+8.4f} {cut[c]*100:5.1f}% {col_mean[c]:10.4f} '
              f'{col_max[c]:9.4f} {eff_mean[c]:+9.4f} {eff_max[c]:+9.4f}'
              + ('   collapsed' if collapsed[c] else ''))


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    d = np.load(sys.argv[1])
    occ_path = sys.argv[2] if len(sys.argv) > 2 else 'occupancy.csv'

    P = head_prefix(d)
    filters = d[P + 'value_conv.weight'].shape[0]
    in_features = d[P + 'value_linear_1.weight'].shape[1]
    if filters > 1 and in_features == 2 * filters:
        pooled_head_checks(d, P)
    elif filters == 1:
        flatten_head_checks(d, P, occ_path)
    else:
        sys.exit(f'不認得的 value head：value_conv 有 {filters} 個 filter，'
                 f'value_linear_1 輸入 {in_features} 維')

    print('\n=== 全網路 BatchNorm gamma ===')
    for k in sorted(d.files):
        if k.endswith('batch_norm.weight') or ('batch_norm_' in k and k.endswith('.weight')):
            v = d[k]
            print(f'  {k:52s} |gamma| mean={np.abs(v).mean():.4f} min={np.abs(v).min():.4f}')


if __name__ == '__main__':
    main()
