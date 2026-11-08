# MI300X vs MI325X: op-test MoE của GLM-5.3-Flash (decode)

Repo này ghi kết quả op-test cho kernel MoE (routed experts) của GLM-5.3-Flash khi decode. Mục tiêu là giải thích vì sao decode trên MI300X chậm hơn MI325X: TPOT chênh khoảng ×1.4 ở 1P1D, DP4-EP4 × 2 replica, MTP k=1.

Cập nhật lần cuối: **2026-11-09 01:53 KST**. README được cập nhật mỗi khi có kết quả mới; xem [TODO](#todo).

## Bối cảnh: gap đo e2e

**Setup**
- GLM-5.3-Flash, 1P1D. Node decode gồm 2 replica, mỗi replica DP4-EP4 (4 GPU); MTP k=1.
- Bench ISL 10k. Hai máy dùng cùng image, cùng preset, cùng cách bench.
- Node decode: 8 × MI300X (làm mát khí, cap 750 W/GPU) và 8 × MI325X (bản làm mát lỏng, cap 1000 W/GPU).

**TPOT, clock và power theo conc** (TPOT phía client; sclk và power là trung bình 8 GPU node decode, lấy mẫu mỗi 1 s)

| conc | ≈ request / DP rank | MI300X TPOT (ms) | MI325X TPOT (ms) | **gap** | sclk 300X / 325X (GHz) | sclk 325X/300X | power 300X / 325X (W) |
|---|---|---|---|---|---|---|---|
| 8 | 1 | 7.81 | 6.92 | **×1.13** | 1.73 / 2.08 | 1.20 | 530 / 595 |
| 16 | 2 | 9.88 | 7.96 | **×1.24** | 1.56 / 2.07 | 1.33 | 560 / 660 |
| 32 | 4 | 11.96 | 8.86 | **×1.35** | 1.31 / 2.06 | 1.57 | 547 / 759 |
| 64 | 8 | 13.68 | 10.58 | **×1.29** | 1.22 / 2.06 | 1.69 | 563 / 809 |
| 128 | 16 | 15.09 | 11.37 | **×1.33** | 1.16 / 2.01 | 1.73 | 606 / 919 |
| 512 | 64 | 23.84 | 16.59 | **×1.44** | 1.10 / 1.81 | 1.64 | 651 / 956 |
| 1536 | 192 | 43.62 | 31.39 | **×1.39** | 1.09 / 1.71 | 1.57 | 669 / 940 |

Các điểm chính:
- **MI300X không chạm 750 W nhưng sclk vẫn tụt theo tải:** 1.73 GHz ở c8, còn 1.10 GHz ở c512, trong khi power chỉ 530–670 W.
  - PPT chỉ active 1.5–10 % thời gian từ c8 đến c128, và 30 % ở c512.
  - Thermal và PROCHOT bằng 0. Hotspot MI300X 82–88 °C, MI325X 46–52 °C.
- **MI325X giữ khoảng 2.06 GHz đến c64.** Nó chỉ xuống 1.81 GHz ở c512, khi đã chạm cap 1000 W (PPT 95 %).
- **HBM clock:** 1300 MHz (MI300X) và 1500 MHz (MI325X), tức ×1.15, giữ nguyên từ c8 đến c512.
- **Gap tăng nhanh từ c8 đến c32** (khoảng 1 → 4 request/rank), đúng đoạn sclk MI300X rơi từ 1.73 xuống 1.31 GHz. Sau đó gap đi ngang ở ×1.29–1.44.

**Breakdown một step decode ở c512** (torch profile 180 s trên 1 replica, rank 0; cả hai máy cùng batch 66 request/rank và cùng binary kernel)

| mỗi forward | MI300X | MI325X | ratio |
|---|---|---|---|
| step | 48.86 ms | 36.74 ms | **×1.33** |
| Σ thời gian kernel | 46.33 ms | 35.78 ms | ×1.29 |
| GPU idle | 4.69 ms (9.6 %) | 2.76 ms (7.5 %) | ×1.70 |

Phần chênh 12.1 ms của step gồm 84 % là kernel GPU chậm hơn, 16 % là GPU idle nhiều hơn (phía host/CPU).

| nhóm kernel | MI300X µs/step | MI325X µs/step | **ratio** | % thời gian kernel (300X) | % phần chênh kernel |
|---|---|---|---|---|---|
| **MoE** (routed experts: CK a8w4 2-stage GEMM, sorting, quant) | 18713 | 13455 | **×1.39** | 40.4 % | **50 %** |
| GEMM dense (fp8 CK preshuffle, hipBLASLt, batched a8w8) | 9294 | 7229 | ×1.29 | 20.1 % | 20 % |
| attention KDA (linear attention) | 4441 | 3692 | ×1.20 | 9.6 % | 7 % |
| attention sparse MLA | 2264 | 1754 | ×1.29 | 4.9 % | 5 % |
| indexer | 583 | 476 | ×1.22 | 1.3 % | 1 % |
| Mori dispatch / combine (EP) | 4790 | 3826 | ×1.25 | 10.3 % | 9 % |
| mHC | 2523 | 2106 | ×1.20 | 5.4 % | 4 % |
| quant / copy / norm / sampling / khác | 3722 | 3243 | ×1.15 | 8.1 % | 5 % |
| **Σ kernel** | 46329 | 35781 | **×1.29** | | 100 % |

- **MoE mỗi lời gọi trong e2e:**
  - stage1: 262.5 µs (MI300X) / 180.0 µs (MI325X), ×1.46;
  - stage2: 146.2 / 107.6 µs, ×1.36;
  - sorting: 9.0 / 7.5 µs;
  - tổng ≈ 418 / 295 µs, ×1.41.
- Chính tỉ lệ MoE 40 : GEMM 20 : KDA 10 : MLA 5 (% thời gian kernel của MI300X) được dùng để dựng chế độ interleave của op-test bên dưới.
- **Lưu ý:**
  - Profiler làm step chậm hơn ở cả hai máy: 48.9 so với 41.4 ms, và 36.7 so với 31.2 ms khi không profile. Ratio thì không đổi (×1.33).
  - TPOT tổng ở c512 là ×1.44, cao hơn ×1.33, vì replica thứ hai của MI300X chậm hơn replica thứ nhất khoảng 17 %, còn MI325X thì đối xứng. Phần lệch replica đang được tách riêng, không thuộc phạm vi op-test.
- **Câu hỏi mà op-test cần trả lời:** vì sao MoE, nhóm chiếm một nửa phần chênh, lại chậm ×1.39–1.41? Do sclk, do HBM, hay do yếu tố khác?

## Bandwidth: HBM và kernel MoE

| máy | HBM spec | BabelStream đo được (TB/s) | BabelStream / spec | kernel MoE, e2e (TB/s) | kernel MoE, op-test interleave (TB/s) | kernel MoE, op-test stress (TB/s) |
|---|---|---|---|---|---|---|
| **MI300X** (HBM3) | **5.3 TB/s** | **4.23** | 80 % | **2.23** = 53 % BabelStream / 42 % spec | **2.44** = 58 % / 46 % | **2.53** = 60 % / 48 % |
| **MI325X** (HBM3E) | **6.0 TB/s** | **4.7** | 78 % | **3.15** = 67 % BabelStream / 53 % spec | **3.24** = 69 % / 54 % | **3.10** = 66 % / 52 % |
| MI325X / MI300X | ×1.13 | ×1.11 | | **×1.41** | ×1.33 | ×1.23 |

- **BabelStream:** `hip-stream`, kiểu double, mảng 2 GiB, 100 lần mỗi kernel, đo trên cả 8 GPU của mỗi node khi node trống.
- **Kernel MoE:** ở c512 với input dump (routing thật), trọng số thật. Bandwidth = số byte một lời gọi ÷ latency một lời gọi.
  - Số byte một lời gọi là **930 MB** (stage1 + stage2 + sorting + quant), đo bằng `rocprofv3` FETCH_SIZE + WRITE_SIZE. Đây là lượng L2 đọc từ fabric (HBM hoặc MALL) cộng lượng ghi, và như nhau trên hai máy.
  - Latency: e2e lấy từ trace (418 / 295 µs), op-test lấy từ bảng ở mục Kết quả.
- **Đọc bảng:** kernel MoE chỉ khai thác 42–48 % HBM spec trên MI300X, so với 52–54 % trên MI325X. Peak HBM chỉ chênh ×1.11–1.13, nhưng bandwidth thực tế của kernel chênh ×1.23–1.41. Như vậy kernel không bị giới hạn bởi peak HBM; xem thí nghiệm clock ở dưới (sclk là yếu tố chính).

## Thuật ngữ

| term | nghĩa |
|---|---|
| **MoE call** | một lời gọi `aiter.fused_moe` của một lớp MoE trên một EP rank, tức rank 0, giữ expert 0–71 trên tổng 288. Một step decode có **43** lời gọi: lớp 3–44 và lớp MTP. Latency trong bảng là **µs cho một lời gọi**, lấy trung bình 43 lời gọi |
| **stage1 / stage2** | hai kernel GEMM chính của MoE (CK `kernel_moe_gemm`, fp8 × int4). stage1 = gate+up kèm SiLU (`MulABScaleWint4`), stage2 = down (`ExpertWeightWin4`). Latency một lời gọi = stage1 + stage2 + sorting + quant |
| **conc** | số request đồng thời của cả node decode (2 replica × DP4) |
| **m_toks / rank** | số token mỗi DP rank trong một step = 2 × ceil(conc / 8). Hệ số 2 là do MTP k=1: mỗi request có 2 token mỗi step |
| **MoE rows (valid) / pairs** | số dòng rank 0 nhận được sau dispatch EP. Ngoặc là số dòng thật, phần còn lại là pad. *pairs* là số cặp (token, expert) rơi vào 72 expert của rank 0 |
| **sclk** | clock lõi GPU (gfx clock), MHz. Lấy trung bình mẫu 50 ms trong 18 s cuối của cửa sổ đo |
| **W** | socket power trung bình của GPU trong cùng cửa sổ. Cap của mỗi GPU: MI300X 750 W, MI325X 1000 W |
| **PPT %** | phần trăm thời gian GPU bị giới hạn vì chạm cap power (bộ đếm throttle của amd-smi) |
| **HBM clock / fclk** | clock bộ nhớ: MI300X dùng HBM3, tối đa 1300 MHz; MI325X dùng HBM3E, tối đa 1500 MHz. fclk là clock của data fabric |
| **weights real / random** | *real*: expert thật từ checkpoint (8 lớp), lượng tử int4 đúng như server làm (scale khớp bit-exact). *random*: int4 ngẫu nhiên. Hai loại cho latency như nhau (±4 %); bảng dưới dùng real |
| **dataset** | cách sinh routing (token chọn expert nào): **uniform** chọn 8/288 expert đều; **skew** theo Zipf α = 0.45, expert nóng nhất ≈ 4.3× trung bình, giống dữ liệu thật; **dump** là đúng 43 lời gọi MoE ghi lại từ bench e2e ở c512 (chỉ có c512) |

## Cách đo

### Stress test (MoE 100 %)
- Một step decode chỉ gồm 43 MoE call chạy liền nhau, mỗi lớp có routing và trọng số riêng (xoay vòng 8 bản).
- Step được capture thành một **HIP graph**, rồi replay liên tục **20 s**.
- GPU chỉ chạy MoE, không có kernel nào khác, nên đây là tải nặng và đều nhất có thể. Cả hai GPU thường chạm cap power.

### Interleave (giống step decode thật)
- Mô phỏng đúng một step decode của **một DP rank**: giữa các MoE call có các kernel khác của model, theo đúng thứ tự từng lớp.
- Với mỗi lớp MoE:
  1. **attention của lớp đó**, dùng kernel thật của model:
     - lớp linear-attention dùng **KDA decode** (`kda_decode_spec`);
     - lớp DSA và lớp MTP dùng **sparse-MLA decode** (2048 key mỗi query, ngữ cảnh khoảng 10.7k token mỗi request).
  2. **GEMM dense, đúng shape của GLM:**
     - in/out projection của KDA và MLA, indexer;
     - router;
     - shared expert;
     - MLP của 3 lớp dense đầu.
     - fp8 per-token đi qua CK `gemm_a8w8_bpreshuffle` như server, phần còn lại dùng hipBLASLt.
  3. **MoE call**, giống stress test.
  4. **idle**, thay cho phần còn lại của step (Mori dispatch/combine, mHC, quant, copy, GPU nghỉ).
- Số lần gọi GEMM/KDA/MLA và độ dài idle **được hiệu chỉnh theo từng máy và từng conc**, để thời gian GPU của step chia đúng như profile e2e ở c512: **MoE 40 : GEMM 20 : KDA 10 : MLA 5 : còn lại 25 %**.
  - Hiệu chỉnh hai bước: lần đầu với clock nguội, rồi hai lần chỉnh lại dưới tải liên tục. Mỗi điểm đo đều ra đúng 40/20/10/5/25.
  - Kiểm tra mức giống model: nếu chạy đúng số kernel tự nhiên của model (34 KDA, 12 MLA, 355 GEMM mỗi step), tỉ lệ tự ra là 42 : 18 : 12 : 4 (sau khi chừa 25 % phần còn lại), gần với trace e2e.
- Mục đích: tái tạo đúng **môi trường power/clock** mà MoE gặp trong e2e. Khi chạy stress, GPU luôn chạm cap power. Khi chạy interleave, MI300X chỉ khoảng 600 W, MI325X khoảng 975 W, giống e2e (651 W và 956 W).

### Thu số liệu
- **Cách chính, CUDA events:** replay graph của một step liên tục 20 s, đo bằng CUDA event. aiter `GpuMonitor` lấy mẫu sclk và power mỗi 50 ms, bỏ 2 s đầu. Thời gian từng kernel (stage1, stage2, GEMM, KDA, MLA, idle) lấy từ profile một lần replay ngay sau cửa sổ đo, khi clock vẫn như lúc đo.
- **Cách phụ, aiter `run_perftest(testGraph=True)`:** cách aiter tự đo, một đợt replay ngắn khoảng 0.4 s. Latency gần như bằng events (trung vị 0.985) và tỉ lệ giữa hai máy cũng như vậy.
- Mỗi máy chạy 8 GPU song song, mỗi GPU một điểm đo. Mỗi GPU có cap power riêng; đã kiểm tra rằng chạy 8 GPU cho cùng clock như chạy 1 GPU.

## Kết quả: 2 bảng × 3 bảng con

Ghi chú:
- Cột **µs 300X/325X** là MI300X chậm hơn bao nhiêu lần. Cột **sclk 325X/300X** là MI325X có clock cao hơn bao nhiêu lần.
- Tham chiếu e2e ở c512: MoE một lời gọi ≈ 418 µs (MI300X) và ≈ 295 µs (MI325X), tức ×1.41.
- Mục xổ ra của mỗi bảng con có: perftest, power cao nhất, PPT %, HBM clock, fclk.

### Bảng 1: stress (MoE 100 % thời gian GPU)

**uniform** (events: sustained 20 s; W = socket power trung bình, cap MI300X 750 W / MI325X 1000 W)

| conc | m_toks / rank | MoE rows (valid) / pairs | MI300X µs | sclk MHz | W | MI325X µs | sclk MHz | W | µs 300X/325X | sclk 325X/300X | stage1 µs 300X / 325X (×) | stage2 µs 300X / 325X (×) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 16 (2) / 4 | 67.6 | 1945 | 439 | 63.0 | 2085 | 476 | 1.07 | 1.07 | 36.9 / 35.1 (1.05) | 13.4 / 12.8 (1.05) |
| 2 | 2 | 16 (2) / 3 | 68.4 | 1943 | 413 | 63.2 | 2088 | 468 | 1.08 | 1.07 | 37.8 / 35.8 (1.06) | 13.5 / 12.6 (1.07) |
| 4 | 2 | 16 (4) / 8 | 77.3 | 1843 | 611 | 68.8 | 2086 | 678 | 1.12 | 1.13 | 41.8 / 38.9 (1.08) | 18.0 / 14.7 (1.22) |
| 8 | 2 | 16 (7) / 16 | 113.5 | 1602 | 646 | 89.9 | 2074 | 834 | 1.26 | 1.29 | 66.9 / 53.8 (1.24) | 28.7 / 21.1 (1.36) |
| 16 | 4 | 16 (14) / 32 | 161.0 | 1291 | 653 | 118.2 | 2032 | 995 | 1.36 | 1.57 | 93.0 / 68.3 (1.36) | 45.4 / 33.4 (1.36) |
| 32 | 8 | 32 (30) / 65 | 209.4 | 1046 | 733 | 163.3 | 1732 | 996 | 1.28 | 1.66 | 123.8 / 97.6 (1.27) | 68.0 / 52.0 (1.31) |
| 64 | 16 | 64 (58) / 126 | 273.4 | 1001 | 750 | 220.8 | 1653 | 1000 | 1.24 | 1.65 | 166.0 / 135.2 (1.23) | 89.9 / 71.1 (1.27) |
| 128 | 32 | 125 (115) / 254 | 320.5 | 979 | 750 | 264.0 | 1663 | 997 | 1.21 | 1.70 | 196.8 / 162.8 (1.21) | 106.2 / 83.4 (1.27) |
| 256 | 64 | 239 (231) / 511 | 336.7 | 940 | 750 | 273.6 | 1696 | 997 | 1.23 | 1.80 | 205.8 / 170.3 (1.21) | 111.8 / 87.3 (1.28) |
| 512 | 128 | 469 (462) / 1023 | 364.2 | 981 | 750 | 310.0 | 1701 | 996 | 1.17 | 1.73 | 218.8 / 192.4 (1.14) | 120.4 / 96.5 (1.25) |
| 768 | 192 | 701 (693) / 1536 | 389.4 | 1055 | 747 | 330.3 | 1700 | 1000 | 1.18 | 1.61 | 227.4 / 193.4 (1.18) | 131.2 / 111.6 (1.18) |
| 1024 | 256 | 933 (925) / 2047 | 442.8 | 1062 | 740 | 351.8 | 1644 | 992 | 1.26 | 1.55 | 255.5 / 194.5 (1.31) | 151.0 / 124.4 (1.21) |

<details><summary>uniform: perftest, power, PPT, HBM clock, fclk</summary>

| conc | perftest µs 300X / 325X (×) | perftest sclk MHz 300X / 325X | W max 300X / 325X | PPT % 300X / 325X | HBM clk MHz 300X / 325X | fclk MHz 300X / 325X |
|---|---|---|---|---|---|---|
| 1 | 68.8 / 62.8 (1.10) | 1952 / 2093 | 444 / 477 | 0 / 0 | 1300 / 1500 | 1800 / 1800 |
| 2 | 69.8 / 63.4 (1.10) | 1948 / 2097 | 416 / 470 | 0 / 0 | 1300 / 1500 | 1800 / 1800 |
| 4 | 78.0 / 68.8 (1.13) | 1922 / 2096 | 632 / 679 | 10 / 1 | 1300 / 1500 | 1800 / 1800 |
| 8 | 111.4 / 90.5 (1.23) | 1741 / 2084 | 654 / 839 | 11 / 0 | 1300 / 1500 | 1800 / 1800 |
| 16 | 158.1 / 118.3 (1.34) | 1523 / 2050 | 662 / 999 | 8 / 100 | 1300 / 1385 | 1800 / 1800 |
| 32 | 205.6 / 165.9 (1.24) | 1257 / 1848 | 741 / 1001 | 66 / 100 | 1300 / 1317 | 1800 / 1800 |
| 64 | 270.1 / 223.9 (1.21) | 1262 / 1800 | 755 / 1000 | 99 / 100 | 1300 / 1500 | 1800 / 1800 |
| 128 | 316.3 / 258.4 (1.22) | 1272 / 1762 | 752 / 1001 | 100 / 100 | 1280 / 1499 | 1800 / 1800 |
| 256 | 333.0 / 275.0 (1.21) | 1186 / 1775 | 752 / 1002 | 100 / 100 | 1280 / 1384 | 1800 / 1800 |
| 512 | 371.7 / 309.8 (1.20) | 1196 / 1764 | 755 / 1002 | 99 / 100 | 1300 / 1385 | 1800 / 1800 |
| 768 | 383.9 / 330.0 (1.16) | 1340 / 1777 | 755 / 1000 | 93 / 100 | 1100 / 1301 | 1550 / 1800 |
| 1024 | 441.3 / 349.3 (1.26) | 1300 / 1747 | 746 / 1000 | 71 / 99 | 1100 / 1303 | 1550 / 1800 |

</details>

**skew** (events: sustained 20 s; W = socket power trung bình, cap MI300X 750 W / MI325X 1000 W)

| conc | m_toks / rank | MoE rows (valid) / pairs | MI300X µs | sclk MHz | W | MI325X µs | sclk MHz | W | µs 300X/325X | sclk 325X/300X | stage1 µs 300X / 325X (×) | stage2 µs 300X / 325X (×) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 16 (2) / 4 | 67.6 | 1939 | 432 | 67.0 | 2086 | 483 | 1.01 | 1.08 | 37.0 / 35.5 (1.05) | 13.5 / 13.8 (0.98) |
| 2 | 2 | 16 (2) / 4 | 68.6 | 1942 | 420 | 63.3 | 2096 | 476 | 1.08 | 1.08 | 37.7 / 35.7 (1.05) | 13.6 / 12.6 (1.07) |
| 4 | 2 | 16 (4) / 8 | 76.5 | 1861 | 566 | 68.2 | 2083 | 668 | 1.12 | 1.12 | 41.6 / 38.7 (1.07) | 17.3 / 14.5 (1.19) |
| 8 | 2 | 16 (7) / 17 | 110.2 | 1635 | 648 | 89.2 | 2064 | 819 | 1.23 | 1.26 | 64.7 / 53.3 (1.21) | 27.8 / 20.4 (1.36) |
| 16 | 4 | 16 (14) / 32 | 152.5 | 1338 | 657 | 118.3 | 2041 | 988 | 1.29 | 1.53 | 87.6 / 68.7 (1.28) | 43.3 / 33.0 (1.31) |
| 32 | 8 | 32 (29) / 65 | 193.0 | 1080 | 719 | 157.0 | 1762 | 997 | 1.23 | 1.63 | 113.0 / 93.8 (1.20) | 62.9 / 49.4 (1.27) |
| 64 | 16 | 64 (58) / 127 | 261.6 | 987 | 750 | 218.9 | 1721 | 1000 | 1.20 | 1.74 | 158.4 / 133.3 (1.19) | 85.8 / 70.4 (1.22) |
| 128 | 32 | 124 (116) / 258 | 311.0 | 989 | 750 | 247.4 | 1627 | 996 | 1.26 | 1.65 | 190.4 / 152.3 (1.25) | 103.1 / 78.6 (1.31) |
| 256 | 64 | 239 (232) / 517 | 340.3 | 962 | 750 | 279.2 | 1707 | 994 | 1.22 | 1.78 | 208.6 / 173.0 (1.21) | 113.5 / 87.8 (1.29) |
| 512 | 128 | 470 (464) / 1037 | 377.9 | 1050 | 747 | 310.8 | 1691 | 993 | 1.22 | 1.61 | 227.0 / 190.4 (1.19) | 126.0 / 96.4 (1.31) |
| 768 | 192 | 702 (696) / 1556 | 435.4 | 1007 | 750 | 325.7 | 1598 | 1000 | 1.34 | 1.59 | 252.7 / 186.4 (1.36) | 152.7 / 107.5 (1.42) |
| 1024 | 256 | 935 (928) / 2078 | 486.1 | 982 | 750 | 358.2 | 1566 | 990 | 1.36 | 1.59 | 283.8 / 201.7 (1.41) | 167.2 / 119.7 (1.40) |

<details><summary>skew: perftest, power, PPT, HBM clock, fclk</summary>

| conc | perftest µs 300X / 325X (×) | perftest sclk MHz 300X / 325X | W max 300X / 325X | PPT % 300X / 325X | HBM clk MHz 300X / 325X | fclk MHz 300X / 325X |
|---|---|---|---|---|---|---|
| 1 | 69.3 / 62.6 (1.11) | 1947 / 2097 | 438 / 484 | 0 / 0 | 1300 / 1500 | 1800 / 1800 |
| 2 | 70.2 / 63.5 (1.11) | 1947 / 2067 | 423 / 477 | 0 / 0 | 1300 / 1500 | 1800 / 1800 |
| 4 | 77.5 / 68.3 (1.14) | 1925 / 2094 | 581 / 670 | 7 / 0 | 1300 / 1500 | 1800 / 1800 |
| 8 | 107.8 / 89.8 (1.20) | 1816 / 2076 | 658 / 823 | 13 / 0 | 1300 / 1500 | 1800 / 1800 |
| 16 | 149.4 / 114.6 (1.30) | 1621 / 1410 | 665 / 994 | 9 / 74 | 1300 / 1500 | 1800 / 1800 |
| 32 | 190.7 / 157.7 (1.21) | 1288 / 1886 | 727 / 1001 | 47 / 100 | 1300 / 1491 | 1800 / 1800 |
| 64 | 258.4 / 217.5 (1.19) | 1266 / 1829 | 752 / 1000 | 100 / 100 | 1300 / 1500 | 1800 / 1800 |
| 128 | 310.4 / 248.9 (1.25) | 1235 / 1769 | 753 / 1001 | 100 / 100 | 1300 / 1499 | 1800 / 1800 |
| 256 | 341.1 / 279.9 (1.22) | 1199 / 1777 | 752 / 1002 | 100 / 99 | 1300 / 1305 | 1800 / 1800 |
| 512 | 378.8 / 309.4 (1.22) | 1238 / 1360 | 755 / 1001 | 100 / 99 | 1300 / 1303 | 1800 / 1800 |
| 768 | 412.2 / 324.5 (1.27) | 1307 / 1691 | 754 / 1000 | 100 / 100 | 1300 / 1300 | 1800 / 1800 |
| 1024 | 472.1 / 350.2 (1.35) | 1255 / 1693 | 753 / 1001 | 100 / 99 | 1300 / 1302 | 1800 / 1800 |

</details>

**dump** (events: sustained 20 s; W = socket power trung bình, cap MI300X 750 W / MI325X 1000 W)

| conc | m_toks / rank | MoE rows (valid) / pairs | MI300X µs | sclk MHz | W | MI325X µs | sclk MHz | W | µs 300X/325X | sclk 325X/300X | stage1 µs 300X / 325X (×) | stage2 µs 300X / 325X (×) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 512 | 128 | 512 (462) / 1022 | 367.7 | 1061 | 741 | 300.1 | 1659 | 996 | 1.23 | 1.56 | 218.1 / 180.5 (1.21) | 121.7 / 93.7 (1.30) |

<details><summary>dump: perftest, power, PPT, HBM clock, fclk</summary>

| conc | perftest µs 300X / 325X (×) | perftest sclk MHz 300X / 325X | W max 300X / 325X | PPT % 300X / 325X | HBM clk MHz 300X / 325X | fclk MHz 300X / 325X |
|---|---|---|---|---|---|---|
| 512 | 385.8 / 302.4 (1.28) | 1226 / 1746 | 751 / 1001 | 80 / 100 | 1100 / 1301 | 1550 / 1800 |

</details>

### Bảng 2: interleave (MoE 40 : GEMM 20 : KDA 10 : MLA 5 : còn lại 25)

**uniform** (events: sustained 20 s; W = socket power trung bình, cap MI300X 750 W / MI325X 1000 W)

| conc | m_toks / rank | MoE rows (valid) / pairs | MI300X µs | sclk MHz | W | MI325X µs | sclk MHz | W | µs 300X/325X | sclk 325X/300X | stage1 µs 300X / 325X (×) | stage2 µs 300X / 325X (×) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 16 (2) / 4 | 67.9 | 1906 | 415 | 62.6 | 2097 | 456 | 1.08 | 1.10 | 37.1 / 34.9 (1.06) | 13.5 / 12.6 (1.07) |
| 2 | 2 | 16 (2) / 3 | 68.9 | 1911 | 396 | 63.0 | 2095 | 450 | 1.09 | 1.10 | 37.8 / 35.4 (1.07) | 13.6 / 12.7 (1.07) |
| 4 | 2 | 16 (4) / 8 | 77.0 | 1877 | 469 | 69.3 | 2089 | 524 | 1.11 | 1.11 | 41.6 / 39.0 (1.07) | 17.9 / 14.8 (1.21) |
| 8 | 2 | 16 (7) / 16 | 112.0 | 1789 | 497 | 91.0 | 2081 | 595 | 1.23 | 1.16 | 65.4 / 54.3 (1.20) | 28.3 / 21.2 (1.33) |
| 16 | 4 | 16 (14) / 32 | 159.8 | 1670 | 554 | 122.8 | 2074 | 705 | 1.30 | 1.24 | 95.4 / 68.2 (1.40) | 45.2 / 33.7 (1.34) |
| 32 | 8 | 32 (30) / 65 | 206.7 | 1428 | 546 | 160.0 | 2073 | 808 | 1.29 | 1.45 | 124.3 / 96.4 (1.29) | 67.9 / 50.9 (1.33) |
| 64 | 16 | 64 (58) / 126 | 273.9 | 1229 | 540 | 220.7 | 2054 | 834 | 1.24 | 1.67 | 167.2 / 136.6 (1.22) | 89.9 / 70.3 (1.28) |
| 128 | 32 | 125 (115) / 254 | 316.0 | 1202 | 556 | 247.3 | 2013 | 904 | 1.28 | 1.67 | 194.3 / 154.1 (1.26) | 104.9 / 79.5 (1.32) |
| 256 | 64 | 239 (231) / 511 | 338.2 | 1155 | 587 | 265.9 | 2020 | 947 | 1.27 | 1.75 | 207.4 / 167.0 (1.24) | 112.2 / 84.3 (1.33) |
| 512 | 128 | 469 (462) / 1023 | 387.1 | 1125 | 618 | 287.2 | 1900 | 978 | 1.35 | 1.69 | 233.9 / 176.5 (1.33) | 128.9 / 91.3 (1.41) |
| 768 | 192 | 701 (693) / 1536 | 400.4 | 1146 | 601 | 306.8 | 1857 | 985 | 1.30 | 1.62 | 234.8 / 179.9 (1.31) | 135.5 / 102.7 (1.32) |
| 1024 | 256 | 933 (925) / 2047 | 461.7 | 1144 | 616 | 325.7 | 1782 | 977 | 1.42 | 1.56 | 268.5 / 187.5 (1.43) | 158.2 / 109.3 (1.45) |

<details><summary>uniform: perftest, power, PPT, HBM clock, fclk</summary>

| conc | perftest µs 300X / 325X (×) | perftest sclk MHz 300X / 325X | W max 300X / 325X | PPT % 300X / 325X | HBM clk MHz 300X / 325X | fclk MHz 300X / 325X |
|---|---|---|---|---|---|---|
| 1 | 69.8 / 62.9 (1.11) | 1951 / – | 423 / 458 | 0 / 0 | 1300 / 1500 | 1800 / 1800 |
| 2 | 69.0 / 63.7 (1.08) | 1923 / – | 401 / 452 | 0 / 0 | 1300 / 1500 | 1800 / 1800 |
| 4 | 78.3 / 69.3 (1.13) | 1934 / – | 478 / 526 | 0 / 0 | 1300 / 1500 | 1800 / 1800 |
| 8 | 110.9 / 90.6 (1.22) | 1841 / 1702 | 505 / 598 | 0 / 0 | 1300 / 1500 | 1800 / 1800 |
| 16 | 154.1 / 118.3 (1.30) | 1806 / 2081 | 560 / 709 | 0 / 0 | 1300 / 1500 | 1800 / 1800 |
| 32 | 202.1 / 160.2 (1.26) | 1703 / 2081 | 552 / 812 | 0 / 0 | 1300 / 1500 | 1800 / 1800 |
| 64 | 270.1 / 223.0 (1.21) | 1548 / 1683 | 545 / 840 | 0 / 4 | 1300 / 1500 | 1800 / 1800 |
| 128 | 311.0 / 249.1 (1.25) | 1500 / 1537 | 567 / 915 | 1 / 31 | 1300 / 1500 | 1800 / 1800 |
| 256 | 330.2 / 265.7 (1.24) | 1488 / 2043 | 601 / 963 | 1 / 52 | 1300 / 1500 | 1800 / 1800 |
| 512 | 370.6 / 290.6 (1.28) | 1416 / 1983 | 625 / 1001 | 2 / 93 | 1300 / 1499 | 1800 / 1800 |
| 768 | 393.5 / 321.1 (1.23) | 1358 / 1980 | 607 / 1000 | 1 / 91 | 1300 / 1499 | 1800 / 1800 |
| 1024 | 458.9 / 342.4 (1.34) | 1394 / 1896 | 622 / 1001 | 9 / 97 | 1300 / 1499 | 1800 / 1800 |

</details>

**skew** (events: sustained 20 s; W = socket power trung bình, cap MI300X 750 W / MI325X 1000 W)

| conc | m_toks / rank | MoE rows (valid) / pairs | MI300X µs | sclk MHz | W | MI325X µs | sclk MHz | W | µs 300X/325X | sclk 325X/300X | stage1 µs 300X / 325X (×) | stage2 µs 300X / 325X (×) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 16 (2) / 4 | 68.1 | 1906 | 413 | 62.7 | 2088 | 460 | 1.09 | 1.10 | 37.1 / 35.0 (1.06) | 13.7 / 12.6 (1.08) |
| 2 | 2 | 16 (2) / 4 | 69.0 | 1907 | 408 | 63.8 | 2097 | 454 | 1.08 | 1.10 | 37.9 / 35.5 (1.07) | 13.7 / 12.8 (1.07) |
| 4 | 2 | 16 (4) / 8 | 76.4 | 1871 | 471 | 68.8 | 2090 | 518 | 1.11 | 1.12 | 41.5 / 38.9 (1.07) | 17.5 / 14.6 (1.20) |
| 8 | 2 | 16 (7) / 17 | 111.5 | 1779 | 497 | 89.3 | 2084 | 582 | 1.25 | 1.17 | 65.1 / 53.3 (1.22) | 28.3 / 20.4 (1.39) |
| 16 | 4 | 16 (14) / 32 | 153.6 | 1689 | 553 | 114.0 | 2082 | 692 | 1.35 | 1.23 | 90.7 / 65.2 (1.39) | 43.9 / 32.4 (1.36) |
| 32 | 8 | 32 (29) / 65 | 194.9 | 1416 | 551 | 150.6 | 2027 | 853 | 1.29 | 1.43 | 115.3 / 89.5 (1.29) | 63.8 / 47.9 (1.33) |
| 64 | 16 | 64 (58) / 127 | 253.8 | 1298 | 548 | 212.2 | 2002 | 879 | 1.20 | 1.54 | 154.5 / 130.5 (1.18) | 83.4 / 67.4 (1.24) |
| 128 | 32 | 124 (116) / 258 | 311.1 | 1188 | 571 | 246.1 | 1958 | 864 | 1.26 | 1.65 | 190.6 / 154.0 (1.24) | 102.9 / 77.9 (1.32) |
| 256 | 64 | 239 (232) / 517 | 338.3 | 1160 | 587 | 258.4 | 1901 | 938 | 1.31 | 1.64 | 207.5 / 159.9 (1.30) | 112.4 / 83.7 (1.34) |
| 512 | 128 | 470 (464) / 1037 | 390.1 | 1157 | 602 | 290.6 | 1876 | 996 | 1.34 | 1.62 | 235.5 / 178.3 (1.32) | 130.4 / 92.9 (1.40) |
| 768 | 192 | 702 (696) / 1556 | 439.6 | 1143 | 596 | 311.3 | 1843 | 989 | 1.41 | 1.61 | 255.4 / 182.7 (1.40) | 153.4 / 103.8 (1.48) |
| 1024 | 256 | 935 (928) / 2078 | 484.9 | 1143 | 612 | 335.5 | 1758 | 998 | 1.45 | 1.54 | 282.9 / 195.8 (1.44) | 166.3 / 110.5 (1.51) |

<details><summary>skew: perftest, power, PPT, HBM clock, fclk</summary>

| conc | perftest µs 300X / 325X (×) | perftest sclk MHz 300X / 325X | W max 300X / 325X | PPT % 300X / 325X | HBM clk MHz 300X / 325X | fclk MHz 300X / 325X |
|---|---|---|---|---|---|---|
| 1 | 69.9 / 64.8 (1.08) | – / – | 421 / 463 | 0 / 0 | 1300 / 1500 | 1800 / 1800 |
| 2 | 70.7 / 64.0 (1.10) | – / 2106 | 414 / 456 | 0 / 0 | 1300 / 1500 | 1800 / 1800 |
| 4 | 77.2 / 68.9 (1.12) | 1934 / – | 481 / 521 | 0 / 0 | 1300 / 1500 | 1800 / 1800 |
| 8 | 108.1 / 89.6 (1.21) | 1883 / – | 511 / 585 | 0 / 0 | 1300 / 1500 | 1800 / 1800 |
| 16 | 148.5 / 115.3 (1.29) | 1814 / – | 561 / 698 | 0 / 0 | 1300 / 1500 | 1800 / 1800 |
| 32 | 189.5 / 151.8 (1.25) | 1684 / 2083 | 560 / 1000 | 0 / 29 | 1300 / 1500 | 1800 / 1800 |
| 64 | 251.5 / 217.3 (1.16) | 1477 / 2085 | 557 / 1001 | 0 / 51 | 1300 / 1499 | 1800 / 1800 |
| 128 | 308.3 / 242.9 (1.27) | 1413 / 2070 | 578 / 1002 | 1 / 35 | 1300 / 1484 | 1800 / 1800 |
| 256 | 329.9 / 267.8 (1.23) | 1474 / 2049 | 595 / 1001 | 1 / 54 | 1300 / 1486 | 1800 / 1800 |
| 512 | 387.8 / 292.1 (1.33) | 1370 / 1410 | 611 / 1000 | 1 / 93 | 1300 / 1500 | 1800 / 1800 |
| 768 | 436.4 / 328.4 (1.33) | 1406 / 1980 | 601 / 1001 | 2 / 92 | 1300 / 1500 | 1800 / 1800 |
| 1024 | 457.7 / 345.8 (1.32) | 1440 / 1919 | 620 / 1001 | 2 / 100 | 1300 / 1500 | 1800 / 1800 |

</details>

**dump** (events: sustained 20 s; W = socket power trung bình, cap MI300X 750 W / MI325X 1000 W)

| conc | m_toks / rank | MoE rows (valid) / pairs | MI300X µs | sclk MHz | W | MI325X µs | sclk MHz | W | µs 300X/325X | sclk 325X/300X | stage1 µs 300X / 325X (×) | stage2 µs 300X / 325X (×) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 512 | 128 | 512 (462) / 1022 | 381.6 | 1155 | 611 | 287.0 | 1874 | 974 | 1.33 | 1.62 | 228.0 / 175.5 (1.30) | 126.6 / 89.7 (1.41) |

<details><summary>dump: perftest, power, PPT, HBM clock, fclk</summary>

| conc | perftest µs 300X / 325X (×) | perftest sclk MHz 300X / 325X | W max 300X / 325X | PPT % 300X / 325X | HBM clk MHz 300X / 325X | fclk MHz 300X / 325X |
|---|---|---|---|---|---|---|
| 512 | 366.1 / 290.7 (1.26) | 1439 / 1977 | 619 / 1005 | 1 / 92 | 1300 / 1499 | 1800 / 1800 |

</details>


## Cập nhật: thí nghiệm giới hạn clock (2026-11-09 ~01:05 KST)

**Cách làm**
- Đặt giới hạn trên (max) cho sclk và/hoặc HBM clock bằng `amd-smi set --clk-limit`.
- Mỗi cấu hình chạy trên một GPU riêng. Đo stress và interleave, dataset uniform, trọng số thật.
- Đo xong trả giới hạn về đúng giá trị cũ; so lại toàn bộ cài đặt trên cả hai máy và đã xác nhận **giống hệt** lúc trước.

**Cách đọc ô:** `µs @ sclk / HBM`, tức latency MoE một lời gọi @ sclk trung bình (MHz) / HBM clock (MHz). "gốc" là không đặt giới hạn.

| mode | conc | MI300X gốc | MI300X HBM ≤1100 | MI300X sclk ≤1100 | MI325X gốc | MI325X HBM ≤1300 | MI325X sclk ≤1100 | MI325X sclk ≤1100 + HBM ≤1300 | MI325X sclk ≤1950 + HBM ≤1300 | 300X sclk≤1100 / 325X sclk≤1100+HBM≤1300 | 300X gốc / 325X sclk≤1950+HBM≤1300 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stress | 16 | 158.9 @ 1317 / 1300 | 158.6 @ 1386 / 1100 | 187.8 @ 1080 / 1300 | 118.5 @ 1993 / 1500 | 123.3 @ 2056 / 1300 | 178.2 @ 1077 / 1500 | 183.7 @ 1075 / 1300 | 124.7 @ 1912 / 1300 | ×1.02 | ×1.27 |
| stress | 64 | 275.3 @ 997 / 1300 | 274.6 @ 1053 / 1100 | 286.8 @ 942 / 1300 | 222.3 @ 1661 / 1300 | 233.5 @ 1789 / 1300 | 251.3 @ 1060 / 1500 | 259.8 @ 1065 / 1300 (đo lại; lần đầu 395.4) | 243.2 @ 1795 / 1300 | ×1.10 | ×1.13 |
| stress | 256 | 337.1 @ 938 / 1288 | 335.1 @ 987 / 1100 | 343.7 @ 905 / 1300 | 274.2 @ 1711 / 1329 | 276.5 @ 1654 / 1300 | 293.1 @ 1055 / 1499 | 304.1 @ 1060 / 1300 | 284.7 @ 1680 / 1300 | ×1.13 | ×1.18 |
| stress | 512 | 371.8 @ 1058 / 1100 | 369.1 @ 1070 / 1100 | 397.1 @ 901 / 1300 | 308.6 @ 1676 / 1303 | 310.5 @ 1687 / 1300 | 336.7 @ 1049 / 1499 | 343.1 @ 1054 / 1300 | 312.6 @ 1686 / 1300 | ×1.16 | ×1.19 |
| interleave | 16 | 162.0 @ 1638 / 1300 | 160.1 @ 1694 / 1100 | 188.0 @ 1100 / 1300 | 117.6 @ 2076 / 1500 | 122.0 @ 2083 / 1300 | 179.9 @ 1089 / 1500 | 181.2 @ 1088 / 1300 | 127.4 @ 1932 / 1300 | ×1.04 | ×1.27 |
| interleave | 64 | 278.1 @ 1249 / 1300 | 274.7 @ 1310 / 1100 | 283.5 @ 1043 / 1300 | 216.6 @ 2048 / 1500 | 229.4 @ 2072 / 1300 | 251.4 @ 1077 / 1500 | 261.5 @ 1078 / 1300 | 236.2 @ 1924 / 1300 | ×1.08 | ×1.18 |
| interleave | 256 | 340.3 @ 1145 / 1300 | 334.2 @ 1178 / 1100 | 347.0 @ 994 / 1300 | 258.4 @ 1985 / 1500 | 276.0 @ 2043 / 1300 | 293.6 @ 1068 / 1500 | 305.6 @ 1068 / 1300 | 284.6 @ 1912 / 1300 | ×1.14 | ×1.20 |
| interleave | 512 | 390.0 @ 1130 / 1300 | 382.7 @ 1162 / 1100 | 399.6 @ 986 / 1300 | 287.1 @ 1894 / 1500 | 303.7 @ 1993 / 1300 | 339.1 @ 1062 / 1500 | 346.2 @ 1064 / 1300 | 306.8 @ 1892 / 1300 | ×1.15 | ×1.27 |

**Đọc kết quả**
- **Cùng sclk thực tế và cùng HBM clock thì gap gần như hết.** Ở c16–c64, khi cả hai máy giữ được khoảng 1.05–1.1 GHz, MI300X/MI325X chỉ còn ×1.02–1.08. Ở c256–512, MI300X không giữ nổi 1100 MHz (bị kéo xuống 900–990 MHz), nên gap còn ×1.13–1.16, đúng theo phần clock thấp hơn.
  - stress c64 của MI325X ở cấu hình sclk ≤ 1100 + HBM ≤ 1300: lần đầu ra 395 µs, bất thường. Đo lại (01:17 KST) ra 259.8 µs, trong khi cấu hình sclk ≤ 1100 cho 252.8 µs, bằng lần đầu. Tỉ lệ đúng ở dòng này là ×1.10.
- **HBM clock chỉ đóng góp nhỏ.**
  - MI325X hạ HBM từ 1500 xuống 1300 MHz thì chậm thêm 4–7 %.
  - MI300X hạ HBM từ 1300 xuống 1100 MHz thì latency không đổi (0.98–1.00).
- **sclk là yếu tố chính.** Giới hạn sclk ở 1100 MHz thì MI325X chậm 16–53 %, MI300X chậm 16–18 % ở c16.
- **Cho MI325X cùng giới hạn clock với MI300X (sclk ≤ 1950, HBM ≤ 1300) thì gap vẫn còn ×1.13–1.27.** Lý do là MI325X còn dư power (cap 1000 W) nên vẫn chạy khoảng 1.9 GHz. MI300X thì chạy MoE ở khoảng 1.0–1.3 GHz ngay cả khi chưa chạm 750 W. Ví dụ c16 stress: 1.32 GHz ở 650 W, PPT 8 %.
- **Lưu ý về sclk trung bình.** Trên MI300X, sclk trung bình có thể đánh lừa. Ở c16 interleave, trung bình là 1.64 GHz nhưng latency MoE bằng lúc stress ở 1.32 GHz. Như vậy clock *trong lúc MoE chạy* thấp hơn mức trung bình; phần chênh do các đoạn GEMM/idle kéo trung bình lên.

## Kết luận

- Gap MoE MI300X/MI325X (×1.2–1.4) đến **chủ yếu từ sclk thực tế lúc kernel MoE chạy**: MI300X khoảng 1.0–1.3 GHz, MI325X khoảng 1.7–2.0 GHz.
- Phần còn lại, khoảng 4–7 %, đến từ HBM3E 1500 MHz so với HBM3 1300 MHz.
- Khi hai máy có cùng sclk và cùng HBM clock, kernel MoE chạy gần như nhau (×1.02–1.08).
- Trên MI300X, sclk trung bình (GpuMonitor/amd-smi) không phản ánh clock lúc MoE chạy. Trong interleave, các đoạn GEMM/idle kéo trung bình lên, nhưng MoE vẫn chạy ở khoảng 1.3 GHz.

## Cập nhật: giới hạn clock của MI300X khi chạy MoE

**Câu hỏi:** MI300X chạy MoE chỉ ở khoảng 1.0–1.3 GHz, kể cả khi chưa chạm cap 750 W. Đây là lựa chọn của bộ điều khiển clock, hay là giới hạn cứng?

**Thí nghiệm:** nâng sclk tối thiểu (min clock) lên 1500 hoặc 1900 MHz bằng `amd-smi set --clk-limit sclk min`, rồi đo lại. Đo xong trả về như cũ và đã kiểm tra lại toàn bộ cài đặt.

| điểm | gốc: µs @ sclk / W / PPT | min 1500 MHz | min 1900 MHz |
|---|---|---|---|
| stress c4 | 77.6 @ 1845 / 597 W / 9 % | 77.3 @ 1855 / 584 W / 8 % | 76.6 @ 1855 / 581 W / 8 % |
| stress c16 | 160.2 @ 1304 / 651 W / 7 % | 157.4 @ 1323 / 654 W / 8 % | 157.8 @ 1321 / 657 W / 9 % |
| stress c64 | 276.7 @ 989 / 749 W / 98 % | 272.5 @ 1013 / 750 W / 100 % | 272.7 @ 1008 / 750 W / 100 % |
| stress c512 | 372.1 @ 1049 / 739 W / 72 % | 365.6 @ 1074 / 735 W / 64 % | 368.6 @ 1060 / 744 W / 82 % |
| interleave c16 | 163.8 @ 1635 / 552 W / 0 % | 160.7 @ 1673 / 554 W / 0 % | 160.1 @ 1659 / 556 W / 0 % |

**Kết luận:** firmware **bỏ qua** min clock.
- Ở c16 stress, GPU vẫn chỉ chạy 1.32 GHz ở 655 W, PPT 8 %, hotspot khoảng 67 °C.
- Vậy đây là **giới hạn cứng trong firmware**, ưu tiên hơn cả cài đặt của người dùng. Nó không phải PPT trung bình và không phải nhiệt.
- Ứng viên còn lại là giới hạn dòng điện (EDC/TDC) hoặc PPT tính trên cửa sổ rất ngắn. amd-smi trên MI300X không có counter cho các loại này: điện áp, `GFX_CLK_BELOW_HOST_LIMIT_*` đều báo N/A.

## Cập nhật: interleave sát e2e hơn

**Thay đổi:**
- 25 % "còn lại" giờ không còn là idle hoàn toàn mà gồm: **mHC thật** (`rocm_aiter_ops.mhc_pre` + `mhc_post`) 5 %, **rmsnorm + copy** 8 %, và idle 12 %. Idle đại diện cho Mori dispatch/combine (cần các GPU khác trong nhóm EP) và GPU nghỉ thật.
- **shared expert chạy trên stream phụ**, song song với MoE, như server.
- Tỉ lệ step đo được: MoE 39 : GEMM 20 : KDA 10 : MLA 5 : mHC 5 : norm/copy 8 : idle 12.

| uniform / dump, trọng số thật | MI300X: bản cũ → bản mới (µs @ sclk, W) | MI325X: bản cũ → bản mới | gap cũ → mới |
|---|---|---|---|
| dump c512 | 381.6 @ 1155, 611 W → **379.7** @ 1202, 593 W | 287.0 @ 1874, 974 W → **292.5** @ 1890, 999 W | ×1.33 → ×1.30 |
| uniform c16 | 159.8 → 163.0 | 122.8 → 116.5 | ×1.30 → ×1.40 |
| uniform c64 | 273.9 → 275.3 | 220.7 → 220.9 | ×1.24 → ×1.25 |
| uniform c256 | 338.2 → 338.2 | 265.9 → 261.8 | ×1.27 → ×1.29 |
| uniform c512 | 387.1 → 394.1 | 287.2 → 295.8 | ×1.35 → ×1.33 |
| uniform c1024 | 461.7 → 479.9 | 325.7 → 332.2 | ×1.42 → ×1.44 |

- **MI325X khớp e2e:** 292.5 µs so với 295 µs, lệch −1 %.
- **MI300X vẫn nhanh hơn e2e khoảng 9 %:** 379.7 so với 418 µs.
  - Trong e2e, MI300X chạy ở clock thấp hơn (1.10 so với 1.20 GHz) dù power cao hơn (651 so với 593 W).
  - Phần chênh này nằm ở môi trường của cả node khi chạy e2e. Hai khả năng: cả 8 GPU chạy cùng lúc nên nóng hơn (hotspot 85 °C so với khoảng 70 °C), hoặc Mori trao đổi dữ liệu thật giữa các GPU. Đang kiểm tra khả năng thứ nhất.

## TODO

- [x] Op-test một file, 3 dataset (uniform, skew, dump), 2 mode (stress, interleave), trọng số real và random, 12 mức conc, trên MI300X và MI325X.
- [x] Thêm cột power vào bảng mục Kết quả.
- [x] Thí nghiệm giới hạn clock (sclk / HBM) trên cả hai máy, đã trả clock về như cũ và kiểm tra lại.
- [x] Viết lại kết luận "gap đến từ đâu" theo thí nghiệm clock. Bản trước dựa trên sclk trung bình và kết luận "không do sclk"; kết luận đó sai. Xem mục [Kết luận](#kết-luận).
- [x] Đo lại điểm bất thường MI325X stress c64, sclk ≤ 1100 + HBM ≤ 1300: 395 → 259.8 µs.
- [x] Thêm bối cảnh gap e2e (TPOT, clock, power, breakdown c512) vào README.
- [x] Bandwidth: đo lại BabelStream trên 8 GPU mỗi máy, đo số byte mỗi lời gọi MoE bằng `rocprofv3`, thêm bảng bandwidth của kernel so với HBM.
- [x] Cơ chế giữ clock MI300X ở 1.0–1.3 GHz: đã thử nâng min clock lên 1500/1900 MHz và firmware bỏ qua, nên đây là giới hạn cứng trong firmware. Không phải PPT trung bình, không phải nhiệt; không có counter nào báo ra (xem mục trên).
- [x] Interleave sát e2e hơn (mHC thật, norm/copy, shared expert overlap): MI325X khớp e2e (−1 %), MI300X vẫn nhanh hơn e2e khoảng 9 %.
- [ ] **Đang làm:** chạy cùng một điểm (dump c512, interleave) trên **cả 8 GPU cùng lúc**, để node nóng và tốn power như khi chạy e2e. Xem clock của MI300X có xuống 1.10 GHz và MoE có lên khoảng 418 µs như e2e không.
- [ ] Mori dispatch/combine thật giữa các GPU (op-test nhiều GPU), nếu bước trên chưa giải thích được phần chênh 9 %.
