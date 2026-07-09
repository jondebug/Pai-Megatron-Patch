# 235B checkpoint eval-status log
Generated: 2026-06-24 06:22 PDT

Scope: every checkpoint on disk with iter in [1000,3000].

## Totals

- relevant checkpoints: **246**
- inf-evaluated (OK): **230**
- **PENDING eval: 13**  (HF-ready: 2, need convert: 11)
- incomplete (<32 shards, skipped): 3
- in diverged cells (excluded): 0

## PENDING checkpoints (created, not yet inf-evaluated)

- 235bv11e16_critical_path_rlc0.5_aux0.01 @ iter 1000  — PENDING-needConvert
- 235bv5a_norl_aux0.001_r24 @ iter 3000  — PENDING-needConvert
- 235bv5a_norl_aux0.003_r25 @ iter 3000  — PENDING-needConvert
- 235bv5a_norl_aux0.005_r26 @ iter 3000  — PENDING-needConvert
- 235bv5a_norl_aux0.015_r28 @ iter 2500  — PENDING-needConvert
- 235bv5a_norl_aux0.015_r28 @ iter 2927  — PENDING-needConvert
- 235bv5a_norl_aux0.01_r27 @ iter 3000  — PENDING-needConvert
- 235bv5a_norl_aux0.02_r29 @ iter 3000  — PENDING-needConvert
- 235bv5b_rlc1.0_aux0.005_basemean_r72 @ iter 1000  — PENDING-HFready
- 235bv5b_rlc1.0_aux0.015_kl0.001_r63 @ iter 1000  — PENDING-HFready
- 235bv6_ppo_kl0.001_r02 @ iter 3000  — PENDING-needConvert
- 235bv6_ppo_lm1.0_kl0.001_r03 @ iter 2500  — PENDING-needConvert
- 235bv6_ppo_lm1.0_kl0.001_r03 @ iter 2749  — PENDING-needConvert

## Full per-cell status

**235bv11e16_critical_path_rlc0.5_aux0.01**
    1000:PENDING-needConvert  1524:OK(72.13@7145)
**235bv12corner_per_token_load_weighted_rlc0.03_aux0.001**
    3000:OK(74.93@7421)
**235bv12corner_per_token_load_weighted_rlc0.05_aux0.0005**
    3000:OK(75.02@7798)
**235bv12corner_rlc0.03_aux0.001**
    3000:OK(75.31@7346)
**235bv12corner_rlc0.05_aux0.001**
    2842:OK(75.22@7425)  3000:INCOMPLETE(27/32)
**235bv5a_norl_aux0.001_r24**
    1500:OK(75.47@8741)  2000:OK(75.71@8691)  2200:OK(75.46@8691)  2500:OK(76.01@8691)  2800:OK(75.68@8691)  3000:PENDING-needConvert
**235bv5a_norl_aux0.003_r25**
    1500:OK(75.06@7702)  2000:OK(74.52@8237)  2205:OK(75.16@8237)  2500:OK(74.95@8237)  2827:OK(74.83@8237)  3000:PENDING-needConvert
**235bv5a_norl_aux0.005_r26**
    1500:OK(74.34@7132)  2000:OK(74.31@7172)  2240:OK(74.86@7172)  2500:OK(74.89@7172)  2845:OK(74.76@7172)  3000:PENDING-needConvert
**235bv5a_norl_aux0.015_r28**
    1500:OK(73.98@5780)  2000:OK(73.79@6684)  2277:OK(74.33@6684)  2500:PENDING-needConvert  2927:PENDING-needConvert
**235bv5a_norl_aux0.01_r27**
    1500:OK(74.30@6272)  2000:OK(74.38@6510)  2245:OK(74.38@6510)  2500:OK(74.35@6510)  2881:OK(73.91@6510)  3000:PENDING-needConvert
**235bv5a_norl_aux0.02_r29**
    1500:OK(73.64@5404)  2000:OK(73.57@6385)  2258:OK(73.91@6385)  2500:OK(74.08@6385)  2946:OK(73.33@6385)  3000:PENDING-needConvert
**235bv5a_rlc0.1_ppo_aux0.001_r00**
    3000:OK(75.00@7421)
**235bv5a_rlc0.1_ppo_aux0.003_r01**
    3000:OK(74.78@7372)
**235bv5a_rlc0.1_ppo_aux0.005_r02**
    2000:OK(75.18@6791)  2500:OK(74.99@6791)  3000:OK(75.12@6791)
**235bv5a_rlc0.1_ppo_aux0.015_r04**
    3000:OK(73.22@5563)
**235bv5a_rlc0.1_ppo_aux0.01_r03**
    3000:OK(74.03@6034)
**235bv5a_rlc0.1_ppo_aux0.02_r05**
    1500:OK(73.47@5200)  2000:OK(72.88@4247)  2307:OK(73.26@4669)  2500:OK(73.16@3797)  2995:OK(72.55@3797)  3000:OK(72.26@3797)
**235bv5a_rlc0.5_ppo_aux0.001_r06**
    2190:OK(75.89@8217)  2500:OK(75.51@8217)  2807:OK(75.77@8217)  3000:OK(75.21@8217)
**235bv5a_rlc0.5_ppo_aux0.003_r07**
    1500:OK(74.80@7262)  2209:OK(75.22@7262)  2500:OK(75.29@7262)  2858:OK(74.86@7262)  3000:OK(74.81@7262)
**235bv5a_rlc0.5_ppo_aux0.005_r08**
    1500:OK(74.56@6766)  3000:OK(74.61@5533)
**235bv5a_rlc0.5_ppo_aux0.015_r10**
    2500:OK(73.33@4205)
**235bv5a_rlc0.5_ppo_aux0.01_r09**
    3000:OK(73.28@4643)
**235bv5a_rlc0.5_ppo_aux0.02_r11**
    3000:OK(72.60@3815)
**235bv5a_rlc1.0_ppo_aux0.001_r12**
    2000:OK(75.54@8021)  2200:OK(75.23@8021)  2500:OK(75.47@8021)  2804:OK(75.31@8021)  3000:OK(75.06@8021)
**235bv5a_rlc1.0_ppo_aux0.003_r13**
    1500:OK(75.08@7171)  2000:OK(74.87@6102)  2232:OK(74.58@6102)  2500:OK(74.97@6102)  2824:OK(74.62@6102)  3000:OK(74.38@6102)
**235bv5a_rlc1.0_ppo_aux0.005_r14**
    1500:OK(74.52@6703)  2247:OK(74.83@5806)  2500:OK(74.35@5806)  2890:OK(74.69@5806)  3000:OK(74.23@5806)
**235bv5a_rlc1.0_ppo_aux0.015_r16**
    3000:OK(72.89@4144)
**235bv5a_rlc1.0_ppo_aux0.01_r15**
    1500:OK(74.20@6054)  3000:OK(74.44@4682)
**235bv5a_rlc1.0_ppo_aux0.02_r17**
    3000:OK(71.67@3812)
**235bv5a_rlc2.0_ppo_aux0.005_r20**
    3000:OK(74.36@5481)
**235bv5a_rlc2.0_ppo_aux0.015_r22**
    3000:OK(72.77@5690)
**235bv5a_rlc2.0_ppo_aux0.01_r21**
    3000:OK(73.53@6175)
**235bv5a_rlc2.0_ppo_aux0.02_r23**
    3000:OK(72.42@3899)
**235bv5b_norl_aux0.005_seed1_r130005**
    1000:OK(74.28)  1500:OK(74.99)  1932:OK(74.86)  2000:OK(74.42)  2496:OK(74.24)
**235bv5b_norl_aux0.015_seed1_r130015**
    1000:OK(73.41)  1400:OK(73.79)  1500:OK(73.71)  2000:OK(74.06)  2003:OK(73.85)  2500:OK(74.01)  2805:OK(73.20)  3000:OK(73.61)
**235bv5b_rlc0.1_aux0.003_basecritic_r75**
    3000:OK(75.68@6130)
**235bv5b_rlc0.1_aux0.003_basemean_r74**
    3000:OK(74.66@6054)
**235bv5b_rlc0.1_aux0.003_seed1_r44**
    1000:OK(74.76@7119)
**235bv5b_rlc0.1_aux0.003_seed2_r45**
    1000:OK(74.26@7288)
**235bv5b_rlc0.5_aux0.001_basecritic_g0.3_r110**
    1000:OK(75.61@7964)  1306:OK(75.61@7964)  1500:OK(75.65@7964)  1850:OK(75.91@7964)  2000:OK(75.40@7964)  2373:OK(75.50@7964)  2500:OK(75.00@7964)  2893:OK(74.93@7964)  3000:OK(75.71@7964)
**235bv5b_rlc0.5_aux0.001_basecritic_g0.5_r111**
    1000:OK(74.90@8024)  1305:OK(76.19@8024)  1500:OK(75.79@8024)  1845:OK(75.98@8024)  2000:OK(75.50@8024)  2377:OK(75.70@8024)  2500:OK(76.03@8024)  2905:OK(75.03@8024)  3000:OK(75.61@8024)
**235bv5b_rlc0.5_aux0.001_basecritic_r71**
    1000:OK(75.45@7929)  1500:OK(75.52)  2000:OK(75.39@7549)  2500:OK(75.34)  2690:OK(74.86)  3000:OK(75.15@7549)
**235bv5b_rlc0.5_aux0.001_basemean_r70**
    1500:OK(75.54)  2000:OK(75.38@7845)  2200:OK(75.15@7845)  2500:OK(75.03@7845)  2819:OK(74.89@7845)  3000:OK(75.02@7845)
**235bv5b_rlc0.5_aux0.001_seed1_r42**
    2213:OK(75.91@7405)  2500:INCOMPLETE(31/32)
**235bv5b_rlc0.5_aux0.001_seed2_r43**
    2696:OK(75.03)  3000:OK(75.12@7290)
**235bv5b_rlc0.5_aux0.015_kl0.0003_r60**
    1000:OK(73.57@5407)  1130:OK(73.52@5173)  1500:OK(74.11)  1668:OK(73.74@4898)  2000:OK(73.76@4610)  2095:OK(73.74)  2500:OK(73.63)  2659:OK(73.39@4316)  2661:OK(73.64@4316)  3000:OK(73.46@4316)
**235bv5b_rlc0.5_aux0.02_lm0.3_r62**
    1500:OK(72.32)  2289:OK(72.55)  2500:OK(72.46@4089)  2934:OK(72.48@4089)  3000:OK(72.12@4089)
**235bv5b_rlc0.5_g0.3_mean_ppo_r04**
    1500:OK(73.81@5982)
**235bv5b_rlc0.5_g0_mean_ppo_gumbel_t0.3_r01**
    3000:OK(73.81@6024)
**235bv5b_rlc0.5_g0_mean_ppo_kl0.0001_r05**
    3000:OK(73.99@6414)
**235bv5b_rlc0.5_g0_mean_ppo_kl0.0003_r06**
    3000:OK(74.35@6427)
**235bv5b_rlc0.5_g0_mean_ppo_kl0.001_r07**
    3000:OK(73.76@6443)
**235bv5b_rlc0.5_g0_mean_ppo_kl0.01_r08**
    3000:OK(73.91@6580)
**235bv5b_rlc0.5_g0_mean_ppo_lm0.3_r02**
    2500:OK(74.11@6034)  2912:OK(74.09@6034)  3000:OK(73.65@6034)
**235bv5b_rlc0.5_g0_mean_ppo_lm1.0_r03**
    2500:OK(73.99@6103)  3000:OK(73.66@6103)
**235bv5b_rlc0.5_g0_mean_ppo_r00**
    1500:INCOMPLETE(31/32)
**235bv5b_rlc1.0_aux0.001_seed1_r40**
    2000:OK(61.18@7809)
**235bv5b_rlc1.0_aux0.001_seed2_r41**
    1500:OK(74.65)  2000:OK(75.07@7452)  2500:OK(74.20)
**235bv5b_rlc1.0_aux0.005_basecritic_r73**
    2000:OK(63.69@6758)
**235bv5b_rlc1.0_aux0.005_basemean_r72**
    1000:PENDING-HFready  1736:OK(74.25)  2000:OK(74.11)  2370:OK(73.91)  2500:OK(74.00)  2961:OK(74.15)  3000:OK(74.25)
**235bv5b_rlc1.0_aux0.005_rwdcritical_path_r31**
    1000:OK(74.81)  1500:OK(74.86)  2000:OK(74.87)  2237:OK(75.04)  2500:OK(74.83)  3000:OK(74.73)
**235bv5b_rlc1.0_aux0.005_rwdentropy_r33**
    1000:OK(75.17)  1500:OK(75.34)  1736:OK(75.33)  2000:OK(74.99)  2392:OK(74.86)  2500:OK(74.95)  3000:OK(75.40)
**235bv5b_rlc1.0_aux0.005_rwdper_token_load_weighted_r30**
    1500:OK(74.47)  2000:OK(73.93)  2255:OK(74.44)  2500:OK(74.48)  2887:OK(74.44)
**235bv5b_rlc1.0_aux0.005_rwdtopn_binary_r32**
    1500:OK(74.57)  2000:OK(74.59)  2242:OK(74.90)  2500:OK(74.15)  2880:OK(74.86)
**235bv5b_rlc1.0_aux0.015_kl0.001_r63**
    1000:PENDING-HFready  1500:OK(73.66)  1593:OK(73.64)  2000:OK(73.60)  2166:OK(74.04)
**235bv5b_rlc1.0_aux0.01_seed1_r46**
    1000:OK(73.79)  1500:OK(73.96)  2000:OK(73.56)  2261:OK(73.41)  2500:OK(73.55)  2938:OK(73.60)  3000:OK(73.33)
**235bv5b_rlc1.0_aux0.01_seed2_r47**
    1500:OK(73.71)  2000:OK(73.62)  2279:OK(73.54)  2500:OK(73.92)  2934:OK(73.70)  3000:OK(73.63)
**235bv5b_rlc1.0_aux0.02_kl0.001_r61**
    1000:OK(73.37)  1500:OK(73.63)  1599:OK(73.04)  2000:OK(73.80)  2149:OK(73.66)  2500:OK(73.79)  2689:OK(73.64)
**235bv6_ppo_kl0.001_r02**
    1500:OK(73.97@6045)  2000:OK(74.32@6045)  2102:OK(74.38@6045)  2500:OK(74.13@6045)  2651:OK(74.15@6045)  3000:PENDING-needConvert
**235bv6_ppo_lm1.0_kl0.001_r03**
    1494:OK(73.99@6037)  1500:OK(74.20@6037)  1860:OK(74.26@6037)  2000:OK(74.15@6037)  2287:OK(74.33@6037)  2500:PENDING-needConvert  2749:PENDING-needConvert
**235bv6_ppo_lm1.0_r01**
    2000:OK(74.10@6022)  2243:OK(73.88@6022)
