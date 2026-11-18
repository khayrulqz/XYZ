"""export_snapshot.py  (PRIVATE: build.py / prep.py-r pashe rakhun)
Engine theke ekta public-safe  public_snapshot.json  banay: shudhu summary, kono trade list na.
Run:  py export_snapshot.py            (Pine Log folder theke)
"""
import json, os, sys
import numpy as np, pandas as pd

# ================= SETTINGS =================
LOG_DIR         = None            # None = build.py-r moto auto ("Pine Log" folder). Alada hole path din
OUT             = "public_snapshot.json"
SHOW_PAIR_NAMES = True            # False = pair-er nam lukano (Pair A, Pair B ...)
IS_END, OOS_START = "2020-12-31", "2021-01-01"
RISK_LEVELS     = (0.5, 1.0)      # FTMO risk % per 1R
MAX_FLOAT_DD_R  = None            # engine-er "Max floating DD" (jemon -18.2). None = skip, tokhon page closed-trade DD dekhabe
STRATEGY_TEXT   = ("Price-action swing model using pending limit orders with fixed stop-loss and take-profit. "
                   "Works on any time zone and on multiple FX pairs.")   # sweep/BOS/fib likhben na

# "How it works" (public): customer-ke ki hoy ta bolun, strategy-r logic na. Nijer moto edit korun.
HOW_IT_WORKS = [
    ("Add the indicator to your chart", "Apply the indicator to a supported FX pair on TradingView."),
    ("It maps the trade plan", "When its conditions are met, the indicator draws the pending entry level, stop-loss and take-profit on the chart."),
    ("You place the order", "Place the pending limit order at the marked level with the stop-loss and take-profit shown. Risk per trade is fixed."),
    ("The trade runs to its target or stop", "Every trade ends at the fixed take-profit or the fixed stop-loss. Results are measured in R, where 1R is the amount risked."),
    ("Track the results", "This dashboard summarises the full backtest, including the losing periods."),
]
# ============================================


def kpis(g):
    g = g.sort_values("utc_naive")
    r = g["r"].astype(float).values
    if len(r) == 0:
        return {}
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    cum = np.concatenate([[0.0], np.cumsum(r)])
    dd = float((cum - np.maximum.accumulate(cum)).min())
    run = best = 0
    for x in r:
        run = run + 1 if x < 0 else 0
        best = max(best, run)
    hd = ((g["utc_naive"] - g["open_n"]).dt.total_seconds() / 86400)
    hold = hd.mean()
    return dict(trades=int(len(r)), net_r=round(float(r.sum()), 1),
                win_rate=round(float((g["res"] == "WIN").mean() * 100), 1),
                profit_factor=round(float(pos / neg), 2) if neg > 0 else None,
                expectancy_r=round(float(r.mean()), 3), avg_hold_days=round(float(hold), 1),
                avg_win_r=round(float(r[r > 0].mean()), 2) if (r > 0).any() else None,
                avg_loss_r=round(float(r[r < 0].mean()), 2) if (r < 0).any() else None,
                median_hold_days=round(float(hd.median()), 1),
                p90_hold_days=round(float(hd.quantile(0.9)), 1),
                max_dd_r=round(dd, 1), max_consec_loss=int(best))


def run(d=None, ftmo_block=None, folder=None, out=OUT):
    from prep import load, build_timeline, build_float_timeline, PASS_MODE
    if d is None:
        if folder is None:
            folder = LOG_DIR
        if folder is None:
            import build as _b
            folder = _b.LOG_DIR
        d = load(folder, verbose=False)
    if ftmo_block is None:
        from build import ftmo_block
    d = d.sort_values("utc_naive").reset_index(drop=True)

    # equity: SHUDHU shaptahik point (trade-wise na)
    s = pd.Series(d["r"].astype(float).values, index=pd.DatetimeIndex(d["utc_naive"])).cumsum()
    w = s.resample("W").last().ffill()
    mo = d.groupby(d["utc_naive"].dt.to_period("M"))["r"].sum()
    heat = {}
    for p, v in mo.items():
        heat.setdefault(str(p.year), [None] * 12)[p.month - 1] = round(float(v), 1)
    yearly = {y: round(sum(x for x in v if x is not None), 1) for y, v in heat.items()}

    # FTMO (engine-er nijer simulator)
    tl = build_timeline(d); tlf = build_float_timeline(d, tl)
    cfg = dict(rules=dict(p1=8, p2=5, mdd=10, ddd=5, pass_mode=PASS_MODE), risk_levels_pct=list(RISK_LEVELS))
    rows, _status, pm = ftmo_block(d, tlf, cfg, "equity")
    keep = ["Risk per trade", "Phase 1 passes", "Challenges passed", "Violated (max loss)", "Violated (daily loss)",
            "Success rate", "Avg total days to full pass"]
    fr = rows[keep].copy()
    fr["Success rate"] = (fr["Success rate"] * 100).round(1)
    fr["Avg total days to full pass"] = fr["Avg total days to full pass"].round(0)
    fr = fr.astype(object).where(fr.notna(), None)

    pairs = []
    for p, g in d.groupby("pair"):
        pairs.append(dict(pair=str(p), trades=int(len(g)), net_r=round(float(g["r"].sum()), 1),
                          win_rate=round(float((g["res"] == "WIN").mean() * 100), 1)))
    pairs.sort(key=lambda x: -x["net_r"])
    if not SHOW_PAIR_NAMES:
        for i, x in enumerate(pairs):
            x["pair"] = "Pair " + chr(65 + i)

    is_m = d["open_n"] < pd.Timestamp(IS_END) + pd.Timedelta(days=1)
    oos_m = d["open_n"] >= pd.Timestamp(OOS_START)
    snap = dict(
        meta=dict(sample=False, generated=pd.Timestamp.now().strftime("%Y-%m-%d"),
                  period_from=str(d["utc_naive"].min().date()), period_to=str(d["utc_naive"].max().date()),
                  unit="R (risk multiple)", ftmo_pass_mode=pm, ftmo_rules="2-step 8% / 5%, 5% daily, 10% max loss",
                  is_window="to " + IS_END, oos_window="from " + OOS_START),
        strategy=STRATEGY_TEXT,
        how_it_works=[dict(title=a, text=b) for a, b in HOW_IT_WORKS],
        kpi={**kpis(d), "pairs": int(d["pair"].nunique()),
             **({"max_float_dd_r": MAX_FLOAT_DD_R} if MAX_FLOAT_DD_R is not None else {})},
        equity=dict(x=[t.strftime("%Y-%m-%d") for t in w.index], y=[round(float(v), 1) for v in w.values]),
        monthly=heat, yearly=yearly,
        ftmo=json.loads(fr.to_json(orient="records")),
        is_oos=dict(IS=kpis(d[is_m]), OOS=kpis(d[oos_m])),
        pairs=pairs)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=1, allow_nan=False)
    print("OK ->", os.path.abspath(out), "|", snap["kpi"]["trades"], "trades summarised (no trade list written)")
    return out


if __name__ == "__main__":
    run()
