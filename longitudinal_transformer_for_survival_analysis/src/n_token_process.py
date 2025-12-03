import os

spatial_ks = [2, 4, 6, 8, 10]
temporal_kt = [2, 4, 6, 8, 10]

base_path = "/home/lin01231/song0760/longitudinal_transformer_for_survival_analysis/src/results/surv_AREDS_SF_step-ahead_50-ep_deform-spatial_deform-temporal_nps-{}_npt-{}"

# 保存结果矩阵
c_index_table = []

for ks in spatial_ks:
    row = []
    for kt in temporal_kt:
        folder = base_path.format(ks, kt)
        summary_file = os.path.join(folder, "test_summary.txt")
        try:
            with open(summary_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if "Mean C-index" in line:
                        value = float(line.strip().split(":")[1])
                        row.append(value)
                        break
                else:
                    row.append("N/A")  # fallback if not found
        except FileNotFoundError:
            row.append("N/A")
    c_index_table.append(row)

# 生成 LaTeX 表格
latex = "\\begin{table}[h!]\n\\centering\n"
latex += "\\caption{Concordance index $C(t, \\Delta t)$ for combinations of spatial ($K_s$) and temporal ($K_t$) token sampling numbers.}\n"
latex += "\\begin{tabular}{c|" + "c" * len(temporal_kt) + "}\n"
latex += "\\toprule\n"
latex += "$K_s \\backslash K_t$ & " + " & ".join(map(str, temporal_kt)) + " \\\\\n"
latex += "\\midrule\n"

for i, ks in enumerate(spatial_ks):
    row_str = f"{ks} " + " & " + " & ".join([f"{v:.3f}" if isinstance(v, float) else v for v in c_index_table[i]])
    latex += row_str + " \\\\\n"

latex += "\\bottomrule\n\\end{tabular}\n\\end{table}"

print(latex)
