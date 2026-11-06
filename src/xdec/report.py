"""REPORT.md (Korean) from results/analysis.json + tables.md. Verdicts are rule-based (protocol.md): a difference counts only
when the 95% paired-bootstrap CI excludes 0; R2 is primary, R0 secondary. Observations and interpretation are separated.

  python -m src.xdec.report --run output/2026-11-05/xdec
"""
import argparse
import json
import os


def ci(d):
    if not d or d.get("point") is None:
        return "n/a"
    return f"{d['point']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}]"


def sig(d):
    if not d or d.get("lo") is None or d["lo"] != d["lo"]:
        return "n/a"
    if d["lo"] > 0:
        return "개선(CI>0)"
    if d["hi"] < 0:
        return "악화(CI<0)"
    return "차이 확인 안 됨"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    R = a.run
    A = json.load(open(os.path.join(R, "results", "analysis.json")))
    C = {**A.get("contrasts_main", {}), **A.get("contrasts_refs", {})}
    T = A.get("twobytwo", {})
    M = A.get("main", {})
    st = {}
    if os.path.exists(os.path.join(R, "queue_status.tsv")):
        for ln in open(os.path.join(R, "queue_status.tsv")).read().splitlines()[1:]:
            p = ln.split("\t")
            st[p[0]] = p[1]

    def c(x, y, rd="r2", m="m0", col="Overall"):
        return C.get(f"{x}-{y}/{rd}/{m}", {}).get(col)

    def line(x, y, label=None):
        return (f"| {label or f'{x} − {y}'} | {ci(c(x, y))} | {sig(c(x, y))} | {ci(c(x, y, col='Obj D5-16'))} | "
                f"{ci(c(x, y, m='m2'))} | {ci(c(x, y, rd='r0'))} |")

    L = ["# Decoder Architecture Pilot (E1–E7) 결과 보고서", "",
         "> 자동 생성 초안(규칙 기반 판정). 수치 원본은 [`tables.md`](tables.md), [`results/analysis.json`](results/analysis.json)에 있다. "
         "프로토콜은 [`protocol.md`](protocol.md), 구조 감사는 [`architecture_audit.md`](architecture_audit.md)에 있다. PACO **Val만** 평가했고 Test는 평가하지 않았다. "
         "학습 seed는 0 하나이므로 CI는 평가 표본의 불확실성만 나타낸다.", ""]
    L += ["## 실행 상태", "", "| item | status |", "|---|---|"] + [f"| {k} | {v} |" for k, v in st.items()] + [""]
    L += ["## 주요 표 (R2, val)", "", "| Method | ckpt | COCO-dev Feature Loss | COCO-dev Token Cos | PACO-val Token Cos | Val M0 R2 | Obj D5–16 | Val M2 R2 |", "|---|---|---|---|---|---|---|---|"]
    f = lambda v, nd=2: "–" if v is None else f"{v:.{nd}f}"  # noqa: E731
    for m, r in M.items():
        L.append(f"| {m} | {r.get('step') or '–'} | {f(r.get('dev_loss'), 4)} | {f(r.get('dev_cos'), 4)} | {f(r.get('paco_val_tok_cos'), 4)} | "
                 f"{f(r.get('M0'))} | {f(r.get('obj516'))} | {f(r.get('M2'))} |")
    L += ["", "## 구성 요소별 대비 (paired bootstrap)", "",
          "| 대비 | M0 Overall R2 | 판정 | Obj D5–16 R2 | M2 Overall R2 | M0 Overall R0 |", "|---|---|---|---|---|---|"]
    for x, y in (("E1", "Bicubic"), ("E3", "E2"), ("E4", "E3"), ("E5", "E3"), ("E6", "E4"), ("E6", "E5"), ("E7", "E3")):
        L.append(line(x, y))
    L += ["", "## 2×2 (E3/E4/E5/E6)", "", "| 효과 | M0 Overall R2 | 판정 | M0 Overall R0 |", "|---|---|---|---|"]
    for k in ("Query effect, no skip (E4 − E3)", "Query effect, with skip (E6 − E5)", "Skip effect, Mask query (E5 − E3)",
              "Skip effect, Bicubic query (E6 − E4)", "Interaction (E6 − E5) − (E4 − E3)"):
        d2, d0 = T.get(f"{k}/r2/m0", {}).get("Overall"), T.get(f"{k}/r0/m0", {}).get("Overall")
        L.append(f"| {k} | {ci(d2)} | {sig(d2)} | {ci(d0)} |")
    L += ["", "## 각 decoder vs 기준선 (M0 Overall R2)", "", "| Decoder | − Direct Full | − Bicubic | − R00 | − Full Multi-layer | − C0 |", "|---|---|---|---|---|---|"]
    for e in ("E1", "E2", "E3", "E4", "E5", "E6", "E7", "R00"):
        L.append(f"| {e} | " + " | ".join(ci(c(e, r)) for r in ("Direct Full", "Bicubic", "R00", "Full Multi-layer", "C0 Oracle")) + " |")
    L += ["", "## 질문별 답 (규칙 기반 초안)", ""]
    q = [("Q1. Bicubic spatial refinement(E1)이 도움이 되었는가?", [("E1", "Bicubic")]),
         ("Q2. 원본 token을 cross-attention으로 읽는 것이 R00보다 유리했는가?", [("E2", "R00"), ("E3", "R00")]),
         ("Q3. ROI memory보다 full memory가 유리했는가?", [("E3", "E2")]),
         ("Q4. Mask query vs Bicubic query", [("E4", "E3"), ("E6", "E5")]),
         ("Q5. Bicubic을 query에 vs output skip에", [("E5", "E3"), ("E6", "E4"), ("E4", "E5")]),
         ("Q6. Query self-attention의 추가 이득", [("E7", "E3")])]
    for title, pairs in q:
        L.append(f"**{title}** " + "; ".join(f"{x} − {y}: {ci(c(x, y))} → {sig(c(x, y))} (R0 {sig(c(x, y, rd='r0'))})" for x, y in pairs))
        L.append("")
    link = A.get("recon_paco_link", {}).get("spearman", {})
    L.append("**Q7. Teacher 복원 → PACO?** 방법 간 rank 상관(PACO val, R2): " + ", ".join(f"{k.split('/')[0]} ρ={v:+.2f}" for k, v in link.items() if k.endswith("/r2"))
             + ". 방법 수가 적어 CI는 없다. 판정은 본문 해석을 참조.")
    L.append("")
    L.append("**Q8. 경쟁력:** " + "; ".join(f"{e}: vs Direct {sig(c(e, 'Direct Full'))}, vs Bicubic {sig(c(e, 'Bicubic'))}, vs Full ML {sig(c(e, 'Full Multi-layer'))}"
                                           for e in ("E1", "E2", "E3", "E4", "E5", "E6", "E7")))
    L.append("")
    cost = A.get("cost", {})
    L.append("**Q9. 비용:** " + "; ".join(f"{m} {v.get('params') or 0:,} params / {v.get('gflops_roi') or 0:.2f} GFLOPs/ROI" for m, v in cost.items()))
    L.append("")
    L.append("**Q10. S-series와의 차이:** architecture_audit.md §2 참조(입력 L12만, raw feature loss, mask/bicubic query, ROI vs full memory, direct vs skip). "
             "S-series 수치는 다른 프로토콜이라 직접 차이를 계산하지 않았다.")
    L.append("")
    L += ["## 해석 시 주의", "",
          "- Feature reconstruction 개선만으로는 downstream 성공을 주장하지 않는다.",
          "- R0만 개선되고 R2가 개선되지 않으면 reader 차이로 구분한다.",
          "- Skip 대비(E5−E3, E6−E4)에는 skip 연산과 '초기 출력이 Bicubic에서 시작하는 효과'(zero-init)가 함께 들어 있다.",
          "- E7−E3은 SA 추가분(+0.53M params, FLOPs 증가)을 포함한 이번 구성의 효과이고, self-attention 일반의 필요성을 뜻하지 않는다.",
          "- Hidden 256 / 2 block 고정이다. 모두 실패해도 decoder capacity 한계는 배제할 수 없다.",
          "- Seed 1개 결과다. 비슷하게 나오면 '이번 설정에서 아키텍처 변경의 이득이 확인되지 않음'으로 보고한다.", ""]
    open(os.path.join(R, "REPORT.md"), "w").write("\n".join(L) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
