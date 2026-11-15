"""Deterministic pre-qualification engine. Rules mirror data/lending_policy.md. Not a credit decision."""


def tier_apr(score: int) -> float:
    return 0.09 if score >= 720 else 0.14 if score >= 670 else 0.19 if score >= 620 else 0.24


def payment(p: float, apr: float, n: int) -> float:
    r = apr / 12
    return p * r / (1 - (1 + r) ** -n)


def assess(a: dict) -> dict:
    apr = tier_apr(a["credit_score"])
    pay = payment(a["amount"], apr, a["term_months"])
    dscr = a["monthly_cash_flow"] / (a["existing_monthly_debt"] + pay)
    checks = [
        ("Time in business", a["months_in_business"] >= 12, f"{a['months_in_business']} months, need 12", "business operated at least 12 months eligibility"),
        ("Annual revenue", a["annual_revenue"] >= 120000, f"${a['annual_revenue']:,.0f}, need $120,000", "annual revenue eligibility"),
        ("Credit score", a["credit_score"] >= 620, f"{a['credit_score']}, need 620", "owner credit score eligibility"),
        ("Cash flow coverage", dscr >= 1.25, f"{dscr:.2f}, need 1.25", "debt service coverage ratio"),
    ]
    room = a["monthly_cash_flow"] / 1.25 - a["existing_monthly_debt"]
    r = apr / 12
    max_amount = max(0, room * (1 - (1 + r) ** -a["term_months"]) / r)
    fails = [c[0] for c in checks if not c[1]]
    tips = []
    if "Cash flow coverage" in fails:
        tips.append(f"Try a loan up to about ${max_amount:,.0f} on this term, or a longer term to lower the payment.")
    if "Credit score" in fails:
        tips.append("Improving your credit score above 620 would unlock eligibility.")
    if "Time in business" in fails:
        tips.append("Reapply once the business has 12 months of history.")
    if "Annual revenue" in fails:
        tips.append("Annual revenue of $120,000 or more is needed.")
    verdict = "likely_eligible" if not fails else "needs_review" if len(fails) == 1 else "not_yet"
    live = verdict != "not_yet"
    offers = []
    if live:
        offers.append({"name": "Autopay discount", "apr_cut": 0.25, "query": "autopay discount"})
    if live and a["credit_score"] >= 720:
        offers.append({"name": "Excellent credit discount", "apr_cut": 0.5, "query": "excellent credit discount"})
    if live and a["amount"] >= 100000:
        offers.append({"name": "Relationship pricing", "apr_cut": 0.25, "query": "relationship pricing large loan"})
    if a["credit_score"] < 620 or not live:
        offers.append({"name": "Credit builder line", "apr_cut": 0, "query": "credit builder line"})
    apr2 = max(apr - sum(o["apr_cut"] for o in offers) / 100, 0.01)
    pay2 = payment(a["amount"], apr2, a["term_months"])
    return {"verdict": verdict, "apr": apr, "monthly_payment": round(pay, 2), "dscr": round(dscr, 2),
            "max_affordable_amount": round(max_amount),
            "checks": [{"name": n, "passed": ok, "detail": d, "policy_query": q} for n, ok, d, q in checks],
            "tips": tips, "offers": offers, "apr_after_offers": round(apr2, 4), "payment_after_offers": round(pay2, 2)}
