#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-1 recompute for paper 2 (2026-10-31), implementing review items A1/A2/A5/A7.2 + CSV fix.
All numbers are recomputed from the 64 calibration_suite.json + 44 alarm_cost.json artifacts.
The 5 bit-identical duplicate records (trans_medical_p19_split1-5 == e0_base_s1_p19_split1-5,
verified bit-for-bit) are flagged and excluded from the 59-cell main analysis.

Outputs:
  - stdout: all paper-facing numbers (verdict matrix, classwise/class1 counts, B1 boundary,
    final-checkpoint discrimination, alarm tau=0.30 rows + cross-check of all taus)
  - results/master/master_table_v2_20261031.csv  (properly quoted, + is_duplicate, class1 cols)
  - results/master/master_summary_v2_20261031.md  (59-cell summary + alarm incl. tau=0.30)
Zero training; read-only on all raw artifacts.
"""
import json, os, csv, re
from collections import defaultdict

R = "results/"
OUT = R + "master/"

METHODS = ["temperature", "vector", "platt", "isotonic"]

def parse_tag(tag):
    if tag.startswith(("e0_base", "trans_medical")):
        backbone = "coformer"
    elif tag.startswith("grud"):
        backbone = "grud"
    elif tag.startswith("talstm"):
        backbone = "talstm"
    else:
        backbone = "unknown"
    dataset = "p19" if "p19" in tag else ("p12" if "p12" in tag else "pam")
    seed = ""
    for p in tag.split("_"):
        if re.fullmatch(r"s\d+", p):
            seed = p
    return backbone, dataset, seed

DUP_TAGS = set("trans_medical_p19_split%d" % i for i in range(1, 6))

def load_cal(tag):
    with open(R + "cal2_" + tag + "/calibration_suite.json") as f:
        return json.load(f)

def verdict(m, meth):
    e = m[meth]
    if e["ece_rel_drop_pct"] < 10:
        return False
    if not e["brier_improved"]:
        return False
    if meth == "temperature":
        if not e.get("params", {}).get("T_ci_excludes_1", False):
            return False
    return True

def c1(e):
    d = e.get("classwise_ece_detail")
    return d["class1"] if d else None

# ---------- load all cells ----------
tags = sorted(x[5:] for x in os.listdir(R) if x.startswith("cal2_"))
cells = {}
for t in tags:
    d = load_cal(t)
    bb, ds, sd = parse_tag(t)
    cells[t] = dict(raw=d, methods=d["methods"], backbone=bb, dataset=ds, seed=sd,
                    dup=(t in DUP_TAGS))
uniq = [t for t in tags if t not in DUP_TAGS]
binary_uniq = [t for t in uniq if cells[t]["dataset"] != "pam"]
print("total cal2 cells: %d | unique: %d | binary unique: %d" % (len(tags), len(uniq), len(binary_uniq)))

groups = defaultdict(list)
for t in uniq:
    c = cells[t]
    groups[(c["backbone"], c["dataset"])].append(t)

# ---------- A. verdict matrix ----------
print("\n===== A. VERDICT MATRIX (59 unique cells) =====")
print("%-10s %-5s %3s %-16s %s" % ("backbone", "data", "n", "T mean+-sd", "temp/vector/platt/iso GAIN"))
tot = defaultdict(int)
for key in sorted(groups, key=lambda k: (["coformer","grud","talstm"].index(k[0]), ["p19","p12","pam"].index(k[1]))):
    ts = groups[key]
    Ts = [cells[t]["methods"]["temperature"]["params"]["T"] for t in ts]
    tmean = sum(Ts)/len(Ts)
    tvar = sum((x-tmean)**2 for x in Ts)/(len(Ts)-1) if len(Ts) > 1 else 0.0
    tmean, tsd = tmean, tvar ** 0.5
    cnts = [sum(1 for t in ts if verdict(cells[t]["methods"], m)) for m in METHODS]
    for i, m in enumerate(METHODS):
        tot[m] += cnts[i]
    print("%-10s %-5s %3d %.3f+-%.3f      %d/%d  %d/%d  %d/%d  %d/%d" % (
        key[0], key[1], len(ts), tmean, tsd,
        cnts[0], len(ts), cnts[1], len(ts), cnts[2], len(ts), cnts[3], len(ts)))
print("TOTAL(59): temp %d  vector %d  platt %d  isotonic %d" % (tot["temperature"], tot["vector"], tot["platt"], tot["isotonic"]))

# ---------- B. classwise / class1 ----------
print("\n===== B. CLASSWISE / CLASS1 =====")
div_macro = sum(1 for t in uniq
                if cells[t]["methods"]["temperature"]["ece_rel_drop_pct"] > 0
                and cells[t]["methods"]["temperature"]["classwise_ece"] > cells[t]["methods"]["none"]["classwise_ece"])
div_c1 = sum(1 for t in binary_uniq
             if cells[t]["methods"]["temperature"]["ece_rel_drop_pct"] > 0
             and c1(cells[t]["methods"]["temperature"]) > c1(cells[t]["methods"]["none"]))
cw_imp = {m: sum(1 for t in uniq if cells[t]["methods"][m]["classwise_ece"] < cells[t]["methods"]["none"]["classwise_ece"]) for m in METHODS}
c1_imp = {m: sum(1 for t in binary_uniq if c1(cells[t]["methods"][m]) < c1(cells[t]["methods"]["none"])) for m in METHODS}
print("macro: aggregate-ECE improves while classwise-ECE worsens (temp): %d/59" % div_macro)
print("class1 (binary only): aggregate improves while class1-ECE worsens (temp): %d/%d" % (div_c1, len(binary_uniq)))
print("classwise-ECE improved counts /59: " + "  ".join("%s %d" % (m, cw_imp[m]) for m in METHODS))
print("class1-ECE improved counts /%d: " % len(binary_uniq) + "  ".join("%s %d" % (m, c1_imp[m]) for m in METHODS))

# ---------- C. isotonic NLL ----------
iso_nll_bad = sum(1 for t in uniq if cells[t]["methods"]["isotonic"]["nll"] > cells[t]["methods"]["none"]["nll"])
print("\n===== C. ISOTONIC NLL worsening: %d/59 =====" % iso_nll_bad)

# ---------- D. B1 boundary ----------
print("\n===== D. B1 BOUNDARY (group mean |T-1| >= 0.2) =====")
strong, mild = [], []
for key, ts in groups.items():
    Tm = sum(cells[t]["methods"]["temperature"]["params"]["T"] for t in ts) / len(ts)
    (strong if abs(Tm - 1) >= 0.2 else mild).extend(ts)
sG = sum(1 for t in strong if verdict(cells[t]["methods"], "temperature"))
mG = sum(1 for t in mild if verdict(cells[t]["methods"], "temperature"))
print("strong-overconfidence cells: %d, temperature GAIN %d (%.0f%%)" % (len(strong), sG, 100.0*sG/len(strong)))
print("mild cells: %d, temperature GAIN %d (%.0f%%)" % (len(mild), mG, 100.0*mG/len(mild)))

# ---------- E. final-checkpoint discrimination ----------
print("\n===== E. FINAL-CHECKPOINT DISCRIMINATION (from cal2 JSONs) =====")
print("%-10s %-5s %3s  %-18s %-18s %s" % ("backbone", "data", "n", "AUROC", "AUPRC", "acc(PAM)"))
for key in sorted(groups, key=lambda k: (["coformer","grud","talstm"].index(k[0]), ["p19","p12","pam"].index(k[1]))):
    ts = groups[key]
    if key[1] == "pam":
        accs = [cells[t]["methods"]["none"]["acc"] * 100 for t in ts]
        m = sum(accs)/len(accs); s = (sum((x-m)**2 for x in accs)/(len(accs)-1))**0.5
        print("%-10s %-5s %3d  %-18s %-18s %.1f+-%.1f" % (key[0], key[1], len(ts), "-", "-", m, s))
    else:
        aurocs = [cells[t]["raw"]["auroc"] for t in ts]
        auprcs = [cells[t]["raw"]["auprc"] for t in ts]
        def ms(v):
            m = sum(v)/len(v); s = (sum((x-m)**2 for x in v)/(len(v)-1))**0.5
            return "%.2f+-%.2f" % (m, s)
        print("%-10s %-5s %3d  %-18s %-18s" % (key[0], key[1], len(ts), ms(aurocs), ms(auprcs)))

# flagship class1
f = cells["e0_base_s1_p19_split1"]["methods"]
print("flagship class1-ECE: %.6f -> %.6f (%+.1f%%)" % (
    c1(f["none"]), c1(f["temperature"]), 100.0*(c1(f["temperature"])/c1(f["none"])-1)))

# ---------- F. alarm ----------
print("\n===== F. ALARM (44 cells: P19 35 / P12 9, no duplicates) =====")
alarm_tags = sorted(x[6:] for x in os.listdir(R) if x.startswith("alarm_"))
def load_alarm(tag):
    with open(R + "alarm_" + tag + "/alarm_cost.json") as f:
        return json.load(f)
TAUS = ["tau=0.05", "tau=0.10", "tau=0.20", "tau=0.30", "tau=0.50"]
for dsname, dsel in [("P19", "p19"), ("P12", "p12")]:
    ats = [t for t in alarm_tags if dsel in t]
    print("--- %s (%d cells) ---" % (dsname, len(ats)))
    for tau in TAUS:
        row = []
        for m in METHODS:
            dcs = [load_alarm(t)["fixed_tau"][tau][m]["dcost"] for t in ats]
            imp = sum(1 for x in dcs if x < 0)
            row.append("%+.4f (%d/%d)" % (sum(dcs)/len(dcs), imp, len(dcs)))
        print("  %-8s " % tau + "  ".join(row))
# flip rate range at tau=0.20 (P19): per-cell overall and per-method
flips_all = [load_alarm(t)["fixed_tau"]["tau=0.20"][m]["flip_rate"] for t in alarm_tags if "p19" in t for m in METHODS]
print("flip-rate per-cell range at tau=0.20 (P19, all methods): %.3f - %.3f" % (min(flips_all), max(flips_all)))
for m in METHODS:
    fl = [load_alarm(t)["fixed_tau"]["tau=0.20"][m]["flip_rate"] for t in alarm_tags if "p19" in t]
    print("  %-12s %.3f - %.3f" % (m, min(fl), max(fl)))
# transfer: mean |sens_drift| (paper convention) + count of |temp| < |none|
def absdrift(tsel, meth):
    return [abs(load_alarm(t)["transfer"][meth]["sens_drift"]) for t in alarm_tags if tsel in t]
for dsname, dsel in [("P19", "p19"), ("P12", "p12")]:
    an, at = absdrift(dsel, "none"), absdrift(dsel, "temperature")
    smaller = sum(1 for t in alarm_tags if dsel in t
                  and abs(load_alarm(t)["transfer"]["temperature"]["sens_drift"]) < abs(load_alarm(t)["transfer"]["none"]["sens_drift"]))
    print("%s transfer |drift|: none %.3f vs temperature %.3f (temperature smaller in %d/%d)" % (
        dsname, sum(an)/len(an), sum(at)/len(at), smaller, len(an)))

# ---------- G. master table v2 ----------
os.makedirs(OUT, exist_ok=True)
rows = []
alarm_map = {t: load_alarm(t) for t in alarm_tags}
for t in tags:
    c = cells[t]; m = c["methods"]; d = c["raw"]
    for meth in ["none"] + METHODS:
        e = m[meth]
        a = alarm_map.get(t)
        dcost = flip = drift = ""
        if a and meth in METHODS:
            dcost = a["fixed_tau"]["tau=0.20"][meth]["dcost"]
            flip = a["fixed_tau"]["tau=0.20"][meth]["flip_rate"]
            drift = a["transfer"][meth]["sens_drift"]
        pr = e.get("params", {})
        T = pr.get("T", "")
        ci = pr.get("T_ci95", "")
        if isinstance(ci, list):
            ci = "[%s, %s]" % (ci[0], ci[1])
        rows.append({
            "tag": t, "backbone": c["backbone"], "dataset": c["dataset"], "seed": c["seed"],
            "method": meth,
            "verdict": ("GAIN" if verdict(m, meth) else "FAIL") if meth in METHODS else "",
            "is_duplicate": t in DUP_TAGS,
            "T": T, "T_ci95": ci, "T_ci_excludes_1": pr.get("T_ci_excludes_1", ""),
            "ece_before": m["none"]["ece"], "ece_after": e["ece"], "ece_rel_drop_pct": e.get("ece_rel_drop_pct", ""),
            "cwECE_before": m["none"]["classwise_ece"], "cwECE_after": e["classwise_ece"],
            "class1_before": c1(m["none"]) if c1(m["none"]) is not None else "",
            "class1_after": c1(e) if c1(e) is not None else "",
            "brier_before": m["none"]["brier"], "brier_after": e["brier"],
            "nll_before": m["none"]["nll"], "nll_after": e["nll"],
            "auroc": d.get("auroc", d.get("auroc_macro_ovr", "")),
            "auprc": d.get("auprc", d.get("auprc_macro_ovr", "")),
            "alarm_dcost_tau0.2": dcost, "alarm_fliprate_tau0.2": flip,
            "transfer_sens_drift": drift,
        })
csv_path = OUT + "master_table_v2_20261031.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), quoting=csv.QUOTE_MINIMAL)
    w.writeheader()
    w.writerows(rows)
print("\nwrote %s (%d rows)" % (csv_path, len(rows)))

# sanity: re-read with csv reader and verify quoting on a temperature row
with open(csv_path, newline="") as f:
    rr = list(csv.DictReader(f))
temp_row = next(r for r in rr if r["tag"] == "e0_base_s1_p19_split1" and r["method"] == "temperature")
assert len(rr) == len(rows) and temp_row["T_ci95"] == "[1.1129, 1.274366]", "CSV roundtrip failed"
print("CSV roundtrip OK (quoting verified)")

# ---------- master summary v2 ----------
md = ["# 主表汇总 v2（2026-10-31，59 唯一格口径；5 个逐位重复格已标记于 master_table_v2 的 is_duplicate 列）", ""]
md.append("## 判定计数（按 骨干 × 数据集，59 唯一格）")
md.append("| backbone | dataset | n | temp | vector | platt | isotonic | T 均值±sd |")
md.append("|---|---|---|---|---|---|---|---|")
for key in sorted(groups, key=lambda k: (["coformer","grud","talstm"].index(k[0]), ["p19","p12","pam"].index(k[1]))):
    ts = groups[key]
    Ts = [cells[t]["methods"]["temperature"]["params"]["T"] for t in ts]
    tmean = sum(Ts)/len(Ts)
    tsd = (sum((x-tmean)**2 for x in Ts)/(len(Ts)-1))**0.5 if len(Ts) > 1 else 0.0
    cnts = [sum(1 for t in ts if verdict(cells[t]["methods"], m)) for m in METHODS]
    md.append("| %s | %s | %d | %d/%d | %d/%d | %d/%d | %d/%d | %.3f±%.3f |" % (
        key[0], key[1], len(ts), cnts[0], len(ts), cnts[1], len(ts), cnts[2], len(ts), cnts[3], len(ts), tmean, tsd))
md.append("| **总计** | | **59** | **%d/59** | **%d/59** | **%d/59** | **%d/59** | |" % (
    tot["temperature"], tot["vector"], tot["platt"], tot["isotonic"]))
md.append("")
md.append("## 关键计数（59 格口径）")
md.append("- macro classwise 分叉（总体 ECE 改善而 classwise-ECE 恶化，temperature）：%d/59" % div_macro)
md.append("- class1（阳性类）分叉（二分类 %d 格）：%d/%d" % (len(binary_uniq), div_c1, len(binary_uniq)))
md.append("- classwise-ECE 改善计数：temp %d / vector %d / platt %d / isotonic %d（/59）" % (
    cw_imp["temperature"], cw_imp["vector"], cw_imp["platt"], cw_imp["isotonic"]))
md.append("- isotonic NLL 恶化：%d/59" % iso_nll_bad)
md.append("- B1：强过置信组（组均值 |T−1|≥0.2，%d 格）temperature GAIN %d；其余 %d 格 GAIN %d" % (
    len(strong), sG, len(mild), mG))
md.append("")
md.append("## 告警代价（τ 全五档；P19 35 格 / P12 9 格，均不含重复格，与论文口径一致）")
for dsname, dsel in [("P19", "p19"), ("P12", "p12")]:
    ats = [t for t in alarm_tags if dsel in t]
    md.append("### %s（%d 格）" % (dsname, len(ats)))
    md.append("| τ | temperature | vector | platt | isotonic |")
    md.append("|---|---|---|---|---|")
    for tau in TAUS:
        cellsrow = []
        for m in METHODS:
            dcs = [load_alarm(t)["fixed_tau"][tau][m]["dcost"] for t in ats]
            imp = sum(1 for x in dcs if x < 0)
            cellsrow.append("%+.4f (%d/%d)" % (sum(dcs)/len(dcs), imp, len(dcs)))
        md.append("| %s | %s |" % (tau.replace("tau=", ""), " | ".join(cellsrow)))
    md.append("")
md_path = OUT + "master_summary_v2_20261031.md"
with open(md_path, "w") as f:
    f.write("\n".join(md) + "\n")
print("wrote %s" % md_path)
