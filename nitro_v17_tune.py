"""
v17 Cap / Mean-Rev DIAGNOSTIC (no-margin world).

Runs the no-margin v17 (PMP ceiling 1.00/0.85/0.50) in production mode 7 and
dissects the new setups so we know which lever to pull:
  - per-setup trade stats (n, win rate, mean/median/worst/best return)
  - exit-reason breakdown  (whipsaw-stops vs riding-losses vs TP)
  - worst 6 trades with dates
  - marginal attribution: Cap-off, MR-off, both-off vs baseline (CAGR/Sharpe/MDD)
"""
import os, sys
import numpy as np
from collections import Counter

_base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _base)
import nitro_v16
import nitro_v17_backtest as v17b
from nitro_v16 import load_data, compute_dew, trade_stats, fmt_eq


def make_pmp(a, b, c):
    def f(vix):
        v = vix if not np.isnan(vix) else 15.0
        if v < 20:  return a
        if v < 32:  return b
        return c
    return f

PMP_NOMARGIN = make_pmp(1.00, 0.85, 0.50)
nitro_v16.get_pmp = PMP_NOMARGIN
v17b.get_pmp = PMP_NOMARGIN

MODE = "7"


def reset_knobs():
    v17b.CAP_ON = True;  v17b.MR_ON = True
    v17b.CAP_ATR_MULT = 1.0; v17b.MR_ATR_MULT = 1.0
    v17b.CAP_MAXHOLD = 0; v17b.CAP_VIX_MAX = 0.0
    v17b.CAP_CONFIRM = False; v17b.CAP_TP = 0.05


def trade_diag(trades, kind):
    rows = [t for t in trades if len(t) > 7 and t[7] == kind]
    if not rows:
        return None, []
    rets = np.array([t[2] for t in rows])
    exits = Counter(t[3] for t in rows)
    worst = sorted(rows, key=lambda t: t[2])[:6]
    return dict(n=len(rows), win=float((rets > 0).mean()),
                mean=float(rets.mean()), med=float(np.median(rets)),
                worst=float(rets.min()), best=float(rets.max()),
                exits=dict(exits)), worst


def st(r):
    return trade_stats(r["trades"], r["eq"], r["dret"])


def main():
    print("Loading data ...")
    df, df_full, qqq_raw = load_data()
    dew = compute_dew(qqq_raw)
    df = df.merge(dew, on="Date", how="left")
    df["DEW_Signal"] = df["DEW_Signal"].fillna("")
    df = v17b.add_v17_features(df)

    print("\n" + "=" * 92)
    print("  v17 Cap / Mean-Rev DIAGNOSTIC  ·  no-margin (PMP<=1.00)  ·  MODE 7")
    print("=" * 92)

    reset_knobs()
    r_base = v17b.run_engine_v17(df, mode=MODE)
    s_base = st(r_base)
    print(f"\n  BASELINE v17 no-margin: n={s_base['n']} final={fmt_eq(s_base['final'])} "
          f"CAGR={s_base['cagr']*100:.2f}% Sharpe={s_base['sharpe']:.3f} MDD={s_base['mdd']*100:.2f}%  "
          f"(Cap={r_base['n_cap']} MR={r_base['n_mr']})")

    for kind, name in [("cap", "CAP (capitulation, -10% 5-day)"), ("mr", "MEAN-REV (stretch<-2, vol20<0.30)")]:
        d, worst = trade_diag(r_base["trades"], kind)
        print(f"\n  ── {name} ──")
        if d is None:
            print("    (no trades)")
            continue
        print(f"    n={d['n']}  win={d['win']*100:.1f}%  mean={d['mean']*100:+.2f}%  "
              f"median={d['med']*100:+.2f}%  worst={d['worst']*100:+.2f}%  best={d['best']*100:+.2f}%")
        print(f"    exits: {d['exits']}")
        print(f"    worst trades:")
        for t in worst:
            print(f"      {str(t[0])[:10]} -> {str(t[1])[:10]}  {t[2]*100:+7.2f}%  [{t[3]}]")

    # ── Attribution ──────────────────────────────────────────────
    print("\n  " + "-" * 88)
    print("  MARGINAL ATTRIBUTION (mode 7, no-margin)")
    print("  " + "-" * 88)

    reset_knobs(); v17b.CAP_ON = False; v17b.MR_ON = True
    s_nocap = st(v17b.run_engine_v17(df, mode=MODE))
    reset_knobs(); v17b.CAP_ON = True;  v17b.MR_ON = False
    s_nomr = st(v17b.run_engine_v17(df, mode=MODE))
    reset_knobs(); v17b.CAP_ON = False; v17b.MR_ON = False
    s_none = st(v17b.run_engine_v17(df, mode=MODE))
    s_v16 = st(nitro_v16.run_engine(df, mode=MODE))

    def line(tag, s):
        print(f"    {tag:26s} CAGR={s['cagr']*100:6.2f}%  Sharpe={s['sharpe']:6.3f}  "
              f"MDD={s['mdd']*100:7.2f}%  final={fmt_eq(s['final'])}")

    line("v16 no-margin (floor)", s_v16)
    line("v17 both OFF (VIX-gate only)", s_none)
    line("v17 MR only (Cap off)", s_nocap)
    line("v17 Cap only (MR off)", s_nomr)
    line("v17 both ON (baseline)", s_base)
    print()
    print(f"    Cap marginal:  CAGR {(s_base['cagr']-s_nocap['cagr'])*100:+.2f}%  "
          f"MDD {(s_base['mdd']-s_nocap['mdd'])*100:+.2f}pp  Sharpe {s_base['sharpe']-s_nocap['sharpe']:+.3f}")
    print(f"    MR  marginal:  CAGR {(s_base['cagr']-s_nomr['cagr'])*100:+.2f}%  "
          f"MDD {(s_base['mdd']-s_nomr['mdd'])*100:+.2f}pp  Sharpe {s_base['sharpe']-s_nomr['sharpe']:+.3f}")

    print("\n" + "=" * 92)


if __name__ == "__main__":
    main()
