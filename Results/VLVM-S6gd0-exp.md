# S6 GD 0 前置检测实验

## 1. 口径

| 项 | 值 |
| :--- | :--- |
| run dir | `outputs/s5_gdcal_4scene_2026-10-26_191657` |
| 场景 | `5cdEh9F2hJL` / `TEEsavR23oF` / `mv2HUxq3B53` / `wcojb4TFT35` (4 × 99 = 396) |
| provisional | `provisional_sigma=0.55`，`1.0 ≤ dist ≤ 4.0 m` |
| 升级 | `num_obs≥2` 且质心距离 `≤4.0 m`；无 `upgrade_sigma_min` |
| 停止 | 仅 `confirmed` 可停；`pointnav_stop_radius=0.90 m` |
| stable | `stale≥60` / 连续 miss≥5 / 累计 miss≥8；复活 `0.5 m / 10 步 / sigma_c≥0.50` |
| GD 配置 | log-only；`every_n=5`、`record_every_n=1`、`box_thr=0.4`、`max_boxes=1`、`caption_style=vocab`、`min_dist=1.0` |
| policy GD 调用 | 否 |
| TSP3D checkpoint | `data/vlvm_v1/vlvm_v1.pth` |
| 完成标记 / 耗时 | `COMPLETED`（2026-10-27 04:39）/ 9:22:39 |

## 2. 导航总体

| 配置 | SR | Oracle | SPL | SoftSPL | steps |
| :--- | ---: | ---: | ---: | ---: | ---: |
| VLFM (2D 基线) | 45.71% (181/396) | 52.53% (208/396) | 0.2487 | 0.3020 | 174.7250 |
| Raw VLVM ScanRefer | 44.44%（176/396） | 50.76%（201/396） | 0.1812 | 0.2477 | 143.4000 |
| vlvm base (`sigma_tar=0.65`) | 31.31% | 57.58% | 0.1275 | 0.2176 | 237.4343 |
| vlvm v1 (`sigma_tar=0.60`) | 36.61% | 54.55% | 0.1607 | 0.2622 | 188.7449 |
| vlvm p0 | 43.69%（173/396） | 63.13% (250/396) | 0.1737 | 0.2508 | 220.0750 |
| vlvm p1 | 42.68%（169/396） | 61.87%（245/396） | 0.1689 | 0.2495 | 217.4975 |
| S5 final（provisional 0.50） | 41.41% (164 / 396) | 58.08% | 0.1803 | 0.2795 | 184.78 |
| S6gd0（provisional 0.55，GD log-only） | 41.92% (166 / 396) | 61.87% | 0.1721 | 0.2678 | 204.01 |

## 3. 逐场景

| 场景 | S6gd0 success | S6gd0 SR | S6gd0 Oracle | S6gd0 SPL | S6gd0 steps | S5 final success | S5 final SR | S5 final SPL | S5 final steps |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `5cdEh9F2hJL` | 42 | 42.42% | 60.61% | 0.1593 | 230.92 | 39 | 39.39% | 0.1572 | 220.09 |
| `TEEsavR23oF` | 35 | 35.35% | 53.54% | 0.1496 | 214.14 | 37 | 37.37% | 0.1655 | 209.56 |
| `mv2HUxq3B53` | 46 | 46.46% | 71.72% | 0.1833 | 243.56 | 50 | 50.51% | 0.2157 | 193.52 |
| `wcojb4TFT35` | 43 | 43.43% | 61.62% | 0.1962 | 127.43 | 38 | 38.38% | 0.1828 | 115.95 |

## 4. 逐类别

| 类别 | 集数 | S6gd0 success | S6gd0 SR | S5 final success | S5 final SR |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `bed` | 76 | 40 | 52.63% | 41 | 53.95% |
| `chair` | 86 | 59 | 68.60% | 59 | 68.60% |
| `couch` | 82 | 38 | 46.34% | 43 | 52.44% |
| `potted plant` | 21 | 0 | 0.00% | 0 | 0.00% |
| `toilet` | 71 | 10 | 14.08% | 9 | 12.68% |
| `tv` | 60 | 19 | 31.67% | 12 | 20.00% |

## 5. 失败构成与错误锁 / 停

| `failure_cause` | S6gd0 | S5 final |
| :--- | ---: | ---: |
| `did_not_fail` | 166 | 164 |
| `false_positive` | 110 | 132 |
| `false_negative` | 34 | 25 |
| `never_saw_target_did_not_travel_stairs_feasible` | 40 | 32 |
| `never_saw_target_did_not_travel_stairs_likely_infeasible` | 44 | 43 |
| `timeout_true_positive` | 2 | 0 |

| 指标 | S6gd0 | S5 final |
| :--- | ---: | ---: |
| `wrong_stop_before_target_coverage` | 60 | 82 |
| `first_wrong_lock_step` 非空 | 163 | 213 |

## 6. 覆盖与记忆

| 项 | 数值 |
| :--- | ---: |
| candidates | 7,703 |
| candidates TP / FP | 2,240 / 5,463 |
| `s_penalty_admitted` | 1,090 |
| `memory_written` | 2,212 |
| `memory_merged` | 1,636 |

| provisional gate 结果 | n | TP | FP |
| :--- | ---: | ---: | ---: |
| `confirmed` | 247 | 146 | 101 |
| `provisional` | 329 | 123 | 206 |
| `merge_existing` | 1,636 | 1,184 | 452 |
| `missing` | 374 | 41 | 333 |
| `rejected` | 5,117 | 746 | 4,371 |

| 记忆事件 | n | TP |
| :--- | ---: | ---: |
| `new` | 576 | — |
| `merge` | 1,636 | — |
| `upgrade` | 221 | 107 |
| `transition` | 158 | — |
| `delete` | 93 | 18 |

| 状态 / 事件细分 | 数值 |
| :--- | ---: |
| new `confirmed`（TP / FP） | 247（146 / 101） |
| new `provisional`（TP / FP） | 329（123 / 206） |
| upgrade `close_reobservation` / `frame_upgrade` | 189 / 32 |
| transition `confirmed→stable` / `provisional→stable` | 72 / 86 |
| delete `stable_stale_purge` / `stable_miss_delete` / `stable_total_miss` | 63 / 29 / 1 |

| TP 覆盖口径 | S6gd0 | S5 final |
| :--- | ---: | ---: |
| 有 post-occ TP 的集数 | 240 / 396 | 228 / 396 |
| 有 admitted TP 的集数 | 198 / 396 | 202 / 396 |
| 无 post-occ TP 的集数 | 156 / 396 | 168 / 396 |
| 无 admitted TP 的集数 | 198 / 396 | 194 / 396 |

## 7. GD 日志与成本

| 项 | 数值 |
| :--- | ---: |
| `gd_frames.jsonl` 行数（逐步） | 76,037 |
| GD 实际调用帧 | 15,102 |
| GD 调用率 | 19.86% |
| `gd_proposals.jsonl` 行数 | 1,780 |
| 每调用 proposals 的 p50 / p90 / max | 0 / 1 / 1 |
| 延迟 p10 / p50 / p90 / max（ms） | 95.74 / 97.77 / 135.15 / 1007.40 |
| `max_buffer_size`（每集提案缓冲最大值） | 5 |
| `max_tracked_keys`（每集 GD 命中轨道数最大值） | 3 |
| GD 空结果帧 | 4,399 |

| GD 框级决策 | n |
| :--- | ---: |
| `below_thr` | 7,633 |
| `other_class` | 14,770 |
| `picked`（进入 2D→3D 链） | 2,162 |

## 8. GD 提案 precision

| 分组 | n | TP | FP | precision |
| :--- | ---: | ---: | ---: | ---: |
| 全部 | 1,780 | 776 | 1,004 | 43.60% |

| 类别 | n | TP | FP | precision |
| :--- | ---: | ---: | ---: | ---: |
| `bed` | 229 | 176 | 53 | 76.86% |
| `chair` | 206 | 172 | 34 | 83.50% |
| `couch` | 324 | 171 | 153 | 52.78% |
| `potted plant` | 209 | 0 | 209 | 0.00% |
| `toilet` | 140 | 100 | 40 | 71.43% |
| `tv` | 672 | 157 | 515 | 23.36% |

| GD conf | n | TP | FP | precision |
| :--- | ---: | ---: | ---: | ---: |
| `[0.40,0.50)` | 576 | 203 | 373 | 35.24% |
| `[0.50,0.60)` | 378 | 164 | 214 | 43.39% |
| `[0.60,0.70)` | 311 | 139 | 172 | 44.69% |
| `[0.70,1.00)` | 515 | 270 | 245 | 52.43% |

| 距机器人（m） | n | TP | FP | precision |
| :--- | ---: | ---: | ---: | ---: |
| `[0,1)` | 2 | 0 | 2 | 0.00% |
| `[1,2)` | 356 | 173 | 183 | 48.60% |
| `[2,3)` | 423 | 240 | 183 | 56.74% |
| `[3,4)` | 313 | 163 | 150 | 52.08% |
| `[4,5)` | 284 | 146 | 138 | 51.41% |
| `[5,10)` | 402 | 54 | 348 | 13.43% |

| 条件 | n | TP | FP | precision |
| :--- | ---: | ---: | ---: | ---: |
| `depth_ok` | 1,293 | 679 | 614 | 52.51% |
| `depth_far` | 487 | 97 | 390 | 19.92% |
| `gd_pos` | 337 | 297 | 40 | 88.13% |
| `gd_neg` | 1,443 | 479 | 964 | 33.19% |
| `suspect` | 407 | 183 | 224 | 44.96% |
| `clean` | 1,373 | 593 | 780 | 43.19% |

| 累计 `gd_hits` | n | TP | FP | precision |
| :--- | ---: | ---: | ---: | ---: |
| 0 | 1,419 | 469 | 950 | 33.05% |
| 1 | 202 | 159 | 43 | 78.71% |
| 2 | 90 | 83 | 7 | 92.22% |
| 3 | 44 | 41 | 3 | 93.18% |
| 4 | 16 | 15 | 1 | 93.75% |
| ≥5 | 9 | 9 | 0 | 100.00% |

## 9. GD 覆盖

| 口径 | episodes | GD hit | hit rate | 当前 success |
| :--- | ---: | ---: | ---: | ---: |
| 无 post-occ TP | 156 | 18 | 11.54% | 7 |
| 无 admitted TP | 198 | 55 | 27.78% | 18 |

| 无 post-occ TP：类别 | episodes | GD hit | hit rate | 当前 success |
| :--- | ---: | ---: | ---: | ---: |
| `bed` | 16 | 1 | 6.25% | 0 |
| `chair` | 18 | 7 | 38.89% | 3 |
| `couch` | 26 | 3 | 11.54% | 3 |
| `potted plant` | 21 | 0 | 0.00% | 0 |
| `toilet` | 50 | 5 | 10.00% | 0 |
| `tv` | 25 | 2 | 8.00% | 1 |

| 无 admitted TP：类别 | episodes | GD hit | hit rate | 当前 success |
| :--- | ---: | ---: | ---: | ---: |
| `bed` | 23 | 7 | 30.43% | 2 |
| `chair` | 24 | 12 | 50.00% | 4 |
| `couch` | 38 | 15 | 39.47% | 8 |
| `potted plant` | 21 | 0 | 0.00% | 0 |
| `toilet` | 57 | 10 | 17.54% | 1 |
| `tv` | 35 | 11 | 31.43% | 3 |

## 10. 轨道仲裁

| 规则 | n | TP | FP | precision | recall |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 全部 | 284 | 177 | 107 | 62.32% | 100.00% |
| `gd_pos` | 183 | 145 | 38 | 79.23% | 81.92% |
| `gd_hits≥2` | 87 | 78 | 9 | 89.66% | 44.07% |
| `gd_hits≥3` | 43 | 38 | 5 | 88.37% | 21.47% |
| `no gd_pos` | 101 | 32 | 69 | 31.68% | 18.08% |

| AUC | 值 |
| :--- | ---: |
| `gd_conf_max` | 0.5157 |
| `gd_hits_max` | 0.7755 |
| `gd_pos` | 0.7320 |
| `n_proposals` | 0.5560 |

| 比率 | 值 |
| :--- | ---: |
| TP 轨道 `gd_pos` 率 | 81.92%（145 / 177） |
| FP 轨道 `gd_pos` 率 | 35.51%（38 / 107） |
| 差值 | 46.41 pp |

## 11. 停止回放

| 规则 | allowed | TP kept | TP lost | FP kept | FP avoided | precision |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 292 | 171 | 0 | 121 | 0 | 58.56% |
| `state=confirmed` | 275 | 166 | 5 | 109 | 12 | 60.36% |
| `confirmed & gd_pos` | 112 | 91 | 80 | 21 | 100 | 81.25% |
| `confirmed & gd_hits≥2` | 57 | 50 | 121 | 7 | 114 | 87.72% |
| `confirmed & gd_conf≥0.4` | 144 | 111 | 60 | 33 | 88 | 77.08% |

| 停止点比率 | 值 |
| :--- | ---: |
| `confirmed & gd_pos` 的 TP 保留率 | 53.22%（91 / 171） |
| `confirmed & gd_pos` 的 FP 保留率 | 17.36%（21 / 121） |
| 差值 | 35.86 pp |

## 12. 选择上下文

| 项 | 数值 |
| :--- | ---: |
| nav goal 行数 | 6,175 |
| 存在 GD 正票替代轨道的 nav goal | 194 |
| 占比 | 3.14% |

## 13. S6 判定门对照

| 门 | S6 阈值 | 本次数据 | 口径 |
| :--- | :--- | :--- | :--- |
| 覆盖门 | 224 集可行无 TP 命中 ≥50% | 无 post-occ TP 156 集，命中 18（11.54%） | 本次 4 场景 396 集，非 990 集口径 |
| 区分门 | TP 与 FP 的 GD 正票率差 ≥0.3 | 81.92% − 35.51% = 46.41 pp | 轨道级，284 条 |
| 停止门 | TP/FP 正票率差 ≥0.4 且 precision ≥85% | 53.22% − 17.36% = 35.86 pp；precision 81.25% | `confirmed & gd_pos` |
| 停止门（更严） | — | TP 保留 50、FP 保留 7、precision 87.72% | `confirmed & gd_hits≥2` |
| 成本 | — | p50 97.77 ms、p90 135.15 ms、1,780 proposals | GD 调用 15,102 次 |

## 14. 运行信息

| 项 | 值 |
| :--- | :--- |
| `episodes.jsonl` | 396 |
| `gd_frames.jsonl` | 76,037 |
| `gd_proposals.jsonl` | 1,780 |
| `track_frames.jsonl` | 13,094 |
| `memory_events.jsonl` | 2,684 |
| `nav_goals.jsonl` | 6,175 |
| `gd_offline_cal.json` | 278 KB |
| 启动脚本 | `scripts/_nav_s5_gdcal.sh` |
| 完成时间 | 2026-10-27 04:39 |
| 总耗时 | 9:22:39 |
