# S-penalty v2：归一化削弱 S 引导分辨率的证明

v2 评分（covered）：

$$
q(c,S)=c^{\,1-\beta}S^{\,\beta},\qquad \lambda=\frac{\beta}{1-\beta}.
$$

归一化形式 $\tilde S=\dfrac{S}{S+m}$，$m>0$。以下 $\lg$ 为任意固定底的对数，所有结论与底无关。


## 1. 局部弹性：归一化是严格的 log 压缩

$$
\frac{\mathrm d\lg\tilde S}{\mathrm d\lg S}
=\frac{S\,\tilde S'(S)}{\tilde S}
=\frac{m}{S+m}\in(0,1),\qquad \forall\,0<S<\infty .
$$

未归一化的弹性恒为 1。任意 $S_B>S_A\ge0$：

$$
\lg\frac{\tilde S_B}{\tilde S_A}
=\int_{S_A}^{S_B}\frac{m}{S+m}\,\mathrm d\lg S
<\lg\frac{S_B}{S_A}.
$$

压缩比

$$
\kappa=\frac{\lg(\tilde S_B/\tilde S_A)}{\lg(S_B/S_A)}
=1+\frac{\lg\frac{S_A+m}{S_B+m}}{\lg(S_B/S_A)}
\in(0,1).
$$

例：$S_A=0.1$、$S_B=0.3$、$m=0.15$ 时 $\kappa=\lg 0.6/\lg 3=0.465$，S 的对比度只剩 46%。


## 2. 有效主导权随 S 漂移

$$
\frac{\mathrm d\lg q}{\mathrm d\lg S}=\beta\cdot\frac{m}{S+m}\equiv\beta_{\text{eff}}(S)\le\beta .
$$

未归一化的 S 指数恒为 $\beta$，$\beta$ 是真正的可控主导权。归一化后 $\beta_{\text{eff}}$ 随 S 变化：低 S 区接近 $\beta$，高 S 区趋于 0，同一 $\beta$ 在不同区间代表不同主导权。


## 3. 重排门槛被同等比例抬高

取 $c_A>c_B$、$S_A<S_B$。B 越过 A 的条件：

$$
\lg\frac{c_A}{c_B}<\lambda\,\lg\frac{S_B}{S_A}
\quad\Longleftrightarrow\quad
\lambda>\rho,\qquad
\rho=\frac{\lg(c_A/c_B)}{\lg(S_B/S_A)}.
$$

归一化后 S 比值乘上 $\kappa$，条件变为

$$
\lambda>\frac{\rho}{\kappa}.
$$

$\kappa<1$，归一化后需要更大的 $\lambda$ 才能让高 S 候选反超。


## 4. 边界斜率饱和

在 $(\lg c,\lg S)$ 平面，等 $q$ 线的斜率：

$$
\text{未归一化：}\ \frac{\mathrm d\lg c}{\mathrm d\lg S}=-\lambda,
\qquad
\text{归一化：}\ -\lambda\frac{m}{S+m}.
$$

归一化在 $S\to\infty$ 时斜率趋于 0，S 再增加也无法换取更低的 $c$，高 S 区饱和；这与 v1 中 $S\ge t_S$ 后 $w_S\equiv1$ 是同一缺陷，而 covered-high 正是需要引导分辨率的区间。

两端极限给出插值关系：

$$
m\to0:\ q\to c^{\,1-\beta}\ (\text{S 失效}),
\qquad
m\to\infty:\ q\to c^{\,1-\beta}S^{\,\beta}\ (\text{未归一化}).
$$

任何有限 $m$ 都严格弱于未归一化。


## 5. 判别式族：未归一化是无约束形式

$q$ 的单调等价判别分：

$$
s=\lg c+\lambda\lg S .
$$

若 TP/FP 在 $(\lg c,\lg S)$ 上的对数似然比线性，该族可精确取到任一正系数线性判别，最优 $\lambda^{*}=w_S/w_c$。归一化形式为

$$
s=\lg c+\lambda\lg S-\lambda\lg(S+m),
$$

多出的非线性项把判别面限制为曲线族；只有 $\lg(S+m)$ 在数据支撑上近似常数（$m\gg S$）时不损失，而该区间正是 $m$ 不可辨识的区间。因此在幂融合族内

$$
\mathrm{AUC}_{\text{norm}}\le \mathrm{AUC}_{\text{unnorm}},
$$

除退化情形外严格成立。


## 6. 结论与验证口径

- 未归一化保留 S 全区间弹性 1 与恒定主导权 $\beta$，任意两个候选的 S 对数比不被压缩；归一化按 $\kappa<1$ 压缩，且在 $S\to\infty$ 取消 S 的区分作用。
- 上述为几何与均值层证明，不是 AUROC 本身；类内方差决定最终可分性。
- 最终判定以日志中 $(c,S,\text{TP})$ 的 $\lambda$ 一维扫描（AUROC、AUPRC、FPR@TPR95）为准。


## 7. v2 参数离线搜索（P0/P1 十场景重放）

### 7.1 初始范围结论

- $\beta$：主范围 `0.10–0.25`，重点观察 `0.15–0.25`；`0.30` 仅作强 S 边界。
- $q_{\text{admit}}$：总体约 `0.45–0.56`，必须随 $\beta$ 联动下降。
- 当前默认 `q_admit=0.60` 不适用。

### 7.2 参数职责

$$
q=c^{1-\beta}S^{\beta}
$$

- $\beta$ 控制 TSP3D 与价值分数的相对主导权。
- $q_{\text{admit}}$ 只控制准入强度，不控制主导权。
- 两者不适合做完整笛卡尔积；同一准入强度下，$\beta$ 越大，$q_{\text{admit}}$ 必须越低。
- P1 十场景中，全部 15,869 个 covered 候选均满足 $S<c$，因此增大 $\beta$ 会系统性压低 $q$ 的数值尺度。

### 7.3 $\beta$ 的数据约束

| 数据口径 | AUROC 最优 $\beta$ | AUPRC 最优 $\beta$ |
|---|---|---|
| P0 十场景 | 约 0.194 | 约 0.078 |
| P1 十场景 | 约 0.187 | 约 0.088 |
| P1 十场景 covered-only | 约 0.25–0.30 | 约 0.20 |

整体候选支持 $\beta\approx0.10–0.20$，covered 候选内部的 S 引导则支持 $\beta\approx0.20–0.30$。

高分低 S 假阳性也要求稍大的 $\beta$。在固定 TP recall 约 40% 时，111 个 `c≥0.70 且 S<0.14` 的 covered FP 保留量为：

| $\beta$ | 保留 FP |
|---|---|
| 0.20 | 104/111 |
| 0.25 | 51/111 |
| 0.30 | 19/111 |
| 0.35 | 8/111 |

但 $\beta$ 过大后，uncovered 候选的 $q=c$ 会相对 covered 候选取得数值优势。按与 v1 相同的 2,244 个准入规模计算：

| $\beta$ | precision |
|---|---|
| v1 | 72.99% |
| 0.10 | 72.28% |
| 0.20 | 65.55% |
| 0.25 | 60.61% |
| 0.30 | 55.93% |

- `0.10–0.20` 偏整体区分能力。
- `0.25` 是反高分幻觉的重要边界。
- `0.30` 以上暂不适合作为主范围。

### 7.4 $q_{\text{admit}}$ 的联动范围

P1 十场景离线候选上，保持 TP recall 约 `40%–60%` 时：

| $\beta$ | 60% TP recall 对应 $q_{\text{admit}}$ | 40% TP recall 对应 $q_{\text{admit}}$ | 定位 |
|---|---|---|---|
| 0.10 | 0.507 | 0.542 | 弱 S、较高召回 |
| 0.15 | 0.479 | 0.512 | 主范围 |
| 0.20 | 0.450 | 0.482 | 主范围 |
| 0.25 | 0.422 | 0.454 | 强惩罚边界 |
| 0.30 | 0.393 | 0.427 | 过强边界 |

近似关系为：

$$
q_{\text{admit}}\approx[\,0.564-0.57\beta,\;0.599-0.57\beta\,]
$$

当前 `sigma_prop=0.45`，当 $q_{\text{admit}}\le0.45$ 时，所有 uncovered 候选都会直接通过准入。P1 十场景的 uncovered 候选 precision 只有 45.41%，因此低于 `0.45` 的工作点风险较大。

### 7.5 初始参数范围

| $\beta$ | 对应 $q_{\text{admit}}$ 范围 | 地位 |
|---|---|---|
| 0.10 | 0.50–0.56 | TSP3D 主导下界 |
| 0.15 | 0.47–0.53 | 主范围 |
| 0.20 | 0.45–0.50 | 主范围 |
| 0.25 | 0.45–0.47 | 强幻觉抑制边界 |
| 0.30 | 约 0.45 | 仅作上界诊断 |

`q_admit=0.60` 应排除：在 $\beta=0.20$ 时 TP recall 仅 3.74%；当 $\beta\ge0.35$ 时，没有 covered 候选能够通过，准入候选几乎只剩 uncovered。

### 7.6 口径限制

以上结论来自 v1 闭环轨迹上的离线重放，只能用于收缩 v2 参数范围，不能直接视为 v2 的闭环导航结果。
