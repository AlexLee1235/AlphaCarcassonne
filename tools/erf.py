#!/usr/bin/env python3
"""erf.py -- trunk 的有效感受野 (Effective Receptive Field) 與端到端敏感度

理論感受野 != 有效感受野。8 blocks x 2 conv(3x3) 的理論半徑是 16，覆蓋 15x15，
但卷積沒有「求和」運算子，只有「局部加權平均」：把 k 個 3x3 核卷在一起得到的是
路徑數決定的離散高斯，sigma ~= sqrt(2k/3)。k=16 -> sigma ~= 3.3。
(Luo et al. 2016, Understanding the Effective Receptive Field in Deep CNNs)

這支把 17 個卷積核用 FFT 卷在一起，直接量出來。線性化的假設：
  - BatchNorm 折進 conv (gamma / sqrt(var + eps))
  - residual block 視為 I + W2' o W1'
  - ReLU 視為恆等
ReLU 只會削弱傳遞（約一半的路徑歸零），所以這是**上界**。

三個輸出：
  (1) trunk 有效感受野的徑向衰減 —— 若在做均勻求和，「相對中心」應一路都是 1.00
  (2) 端到端（trunk x head 讀出）對每個輸入格的敏感度
  (3) 有號探針：擾動「元件 meeple 數」對 value 的影響方向

(2)(3) 的 head 讀出依架構自動判斷：
  - 舊架構 flatten(H*W) -> Linear：每格一個位置權重 (W2 @ W1)
  - 新架構 global pooling：每格權重都是 1/(H*W)，通道權重取 W2 @ W1 的 mean 那一半。
    max 分支不可線性化，沒有計入。棋盤邊長取 occupancy.csv 的大小，沒有就用 15。

checkpoint-193（舊架構）的實測：
  (1) r=7 -> 0.29,  r=10 -> 0.047,  r=14 -> 0.0023
  (2) 只看空間平面 (0-24): max/min = 10.7x，占用率 <=5% 的格子只有中心區的 0.45
      corr(占用率, 敏感度) = +0.613
  (3) plane15 - plane20 只有 89/225 格符號是正的，mean 還是負的
      -> 網路連「我的 meeple 多是好事」這個方向都沒學到
換成 global pooling 之後，(3) 的正號格數應該大幅上升（CLAUDE.md §8 第 4 項）。

用法:
    python3 readckpt.py checkpoint-193.pt ck.npz
    ./diag_occupancy                     # 產生 occupancy.csv（(2) 的分組需要）
    python3 erf.py ck.npz [occupancy.csv]
"""
import os
import sys

import numpy as np

N = 64          # FFT 網格，要大於最終支撐 (input 3 + 8 blocks x 4 + 1 = 35)
BN_EPS = 1e-3   # model.cc 裡 BatchNorm2dOptions.eps(0.001)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    d = np.load(sys.argv[1])
    occ_path = sys.argv[2] if len(sys.argv) > 2 else 'occupancy.csv'

    def fold_bn(w, prefix):
        g = d[prefix + '.weight']
        var = d[prefix + '.running_var']
        return w * (g / np.sqrt(var + BN_EPS))[:, None, None, None]

    def emb(k):
        """空間核 -> FFT，中心置於 index 0（後續合成都保持這個約定）。"""
        s = k.shape[-1]
        c = s // 2
        o = np.zeros(k.shape[:2] + (N, N))
        o[..., :s, :s] = k
        return np.fft.fft2(np.roll(o, (-c, -c), axis=(-2, -1)))

    def compose(FA, FB):
        """先套用 B 再套用 A。"""
        return np.einsum('acuv,cbuv->abuv', FA, FB)

    # --- 合成整個 trunk + value_conv ---
    F = emb(fold_bn(d['layers.0.input_conv.weight'], 'layers.0.input_batch_norm'))
    C = F.shape[0]
    ident = np.zeros((C, C, 1, 1))
    ident[np.arange(C), np.arange(C), 0, 0] = 1
    F_ident = emb(ident)
    nblocks = sum(1 for k in d.files if k.endswith('_conv_1.weight'))
    for i in range(nblocks):
        L = f'layers.{i+1}.res_{i}_'
        W1 = fold_bn(d[L + 'conv_1.weight'], L + 'batch_norm_1')
        W2 = fold_bn(d[L + 'conv_2.weight'], L + 'batch_norm_2')
        F = compose(F_ident + compose(emb(W2), emb(W1)), F)
    head = f'layers.{nblocks + 1}.'
    F = compose(emb(fold_bn(d[head + 'value_conv.weight'], head + 'value_batch_norm')), F)

    K = np.roll(np.real(np.fft.ifft2(F)), (N // 2, N // 2), axis=(-2, -1))   # [Cv, Cin, N, N]
    M = np.sqrt((K ** 2).sum((0, 1)))
    c = N // 2

    print('=== (1) trunk 有效感受野：value_conv 輸出對「距離 r 的輸入格」的敏感度 ===')
    print('    (線性化，ReLU 當恆等 -> 這是上界。若在做均勻求和，「相對中心」應一路 1.00)\n')
    yy, xx = np.mgrid[0:N, 0:N]
    r = np.maximum(np.abs(yy - c), np.abs(xx - c))       # Chebyshev
    tot = M.sum()
    cum = 0.0
    print(f"{'r':>3} {'環平均敏感度':>16} {'相對中心':>10} {'累積佔比':>10}")
    for rr in range(0, 20):
        m = (r == rr)
        if not m.any():
            break
        cum += M[m].sum()
        print(f'{rr:>3} {M[m].mean():16.4e} {M[m].mean()/M[c,c]:10.4f} {100*cum/tot:9.1f}%')

    # --- 端到端：加上 head 的讀出 ---
    occ = np.loadtxt(occ_path, delimiter=',').ravel() if os.path.exists(occ_path) else None
    W1h = d[head + 'value_linear_1.weight']
    W2h = d[head + 'value_linear_2.weight'][0]
    eff = W2h @ W1h
    Cv = K.shape[0]
    if Cv == 1:
        # 舊架構：flatten(H*W) -> Linear，每格一個位置權重
        n = eff.size
        side = int(round(np.sqrt(n)))
        if side * side != n:
            print('\n(value head 不是 flatten(H*W) 形式，(2)(3) 不適用)')
            return
        pos_w = eff.reshape(side, side)
        Kh = K[0]
        readout = 'head 位置權重'
    elif eff.size == 2 * Cv:
        # 新架構：global mean ++ max pooling。mean 分支每格權重 1/(H*W)，
        # 通道權重 eff[:Cv]；max 分支不可線性化，不計入。
        side = int(round(np.sqrt(occ.size))) if occ is not None else 15
        n = side * side
        pos_w = np.full((side, side), 1.0 / n)
        Kh = np.tensordot(eff[:Cv], K, axes=1)      # [Cin, N, N]
        readout = 'global mean pooling，不含 max 分支'
        print(f'\n(global pooling 架構，棋盤邊長 {side}'
              f'{"（取自 " + occ_path + "）" if occ is not None else "（預設）"})')
    else:
        print(f'\n(不認得的 value head：{Cv} filters、value_linear_1 輸入 {W1h.shape[1]} 維，(2)(3) 略過)')
        return

    T = np.zeros((Kh.shape[0], side, side))
    for py in range(side):
        for px in range(side):
            w = pos_w[py, px]
            if w == 0:
                continue
            T += w * Kh[:, c + py - (side - 1): c + py + 1,
                           c + px - (side - 1): c + px + 1][:, ::-1, ::-1]

    print(f'\n\n=== (2) 端到端 (trunk x {readout}) 對每個輸入格的敏感度 ===')
    print('    只取真正的空間平面 0-24（地形/盾/修道院/連通旗標/元件 meeple 圖），')
    print('    排除 25-79 的常數廣播平面（它們每格都一樣，per-cell 敏感度沒有意義）\n')
    E = np.sqrt((T[0:25] ** 2).sum(0))
    flat = E.ravel()
    mid = side // 2
    print(f'  中心({mid},{mid})={E[mid,mid]*1e3:.0f}   邊({mid},1)={E[mid,1]*1e3:.0f}   '
          f'角(1,1)={E[1,1]*1e3:.0f}')
    print(f'  max/min = {E.max()/E.min():.1f}x    std/mean = {E.std()/E.mean():.2f}')

    if occ is not None:
        if occ.size == n:
            base = flat[occ > 50].mean()
            for lo, hi, lab in [(50, 101, '占用>=50%'), (20, 50, '占用20-50%'),
                                (5, 20, '占用5-20%'), (-1, 5, '占用<=5%')]:
                m = (occ > lo) & (occ <= hi)
                if not m.any():
                    continue
                print(f'  {lab:12s} n={m.sum():3d}  平均 = {flat[m].mean()*1e3:8.0f}  '
                      f'(相對占用>=50% = {flat[m].mean()/base:.2f})')
            print(f'  corr(占用率, 敏感度) = {np.corrcoef(occ, flat)[0,1]:+.3f}')
    else:
        print(f'  (找不到 {occ_path}，跳過占用率分組。先跑 ./diag_occupancy)')

    print('\n=== (3) 有號探針：擾動「元件 meeple 數」對 value 的影響 ===')
    print('    plane 15 = 我方 meeple 數(N 邊，元件層級)，plane 20 = 對手')
    print('    若網路在算 pending，(15 - 20) 方向應該在每一格都讓 value 上升\n')
    for pl, lab in [(15, 'plane15 我方 meeple(N 邊)'), (20, 'plane20 對手 meeple(N 邊)')]:
        S = T[pl]
        print(f'  {lab}: 正號格數={int((S>0).sum())}/{n}  mean={S.mean()*1e3:+9.0f}  '
              f'std={S.std()*1e3:9.0f}')
    D = T[15] - T[20]
    print(f'\n  (15 - 20) 方向: 正號格數={int((D>0).sum())}/{n}  mean={D.mean()*1e3:+.0f}  '
          f'std/|mean|={D.std()/abs(D.mean()):.2f}')
    print('  有號敏感度 (x1e3)，若在做均勻求和應處處同號且數值接近：')
    for row in D:
        print('   ', ' '.join(f'{v*1e3:7.0f}' for v in row))


if __name__ == '__main__':
    main()
