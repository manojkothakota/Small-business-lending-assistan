"""Growth insights from SYNTHETIC applications: where demand is, who is close to qualifying, what to launch."""
import random
from collections import Counter

from .lending import assess

INDUSTRIES = ["Restaurants", "Retail", "Construction", "Healthcare", "Logistics", "Salons and wellness"]
PLAYS = {
    "Credit score": ("Launch a credit-builder line", "Offer a small secured line to owners under 620 and graduate them to full loans."),
    "Cash flow coverage": ("Right-size every offer", "Suggest a smaller amount or longer term automatically using the maximum affordable amount."),
    "Time in business": ("Create a startup pathway", "Pre-approve young businesses for a reapply date so they come back to us."),
    "Annual revenue": ("Add a micro-loan product", "Serve businesses under $120,000 revenue with small, short loans."),
}


def synthetic_applications(n: int = 60) -> list[dict]:
    rnd = random.Random(7)
    return [dict(industry=rnd.choice(INDUSTRIES), months_in_business=rnd.randint(4, 120),
                 annual_revenue=rnd.randint(70, 600) * 1000, monthly_cash_flow=rnd.randint(2500, 14000),
                 existing_monthly_debt=rnd.randint(0, 2500), credit_score=rnd.randint(540, 790),
                 amount=rnd.randint(10, 200) * 1000, term_months=rnd.choice([12, 24, 36])) for _ in range(n)]


def build_insights() -> dict:
    rows = [(a, assess(a)) for a in synthetic_applications()]
    fails = lambda r: [c["name"] for c in r["checks"] if not c["passed"]]
    eligible = [x for x in rows if x[1]["verdict"] == "likely_eligible"]
    near = [x for x in rows if x[1]["verdict"] == "needs_review"]
    reasons = Counter(n for _, r in rows for n in fails(r))
    by_ind = []
    for ind in INDUSTRIES:
        g = [x for x in rows if x[0]["industry"] == ind]
        by_ind.append({"industry": ind, "applications": len(g),
                       "eligible_pct": round(100 * sum(x[1]["verdict"] == "likely_eligible" for x in g) / max(len(g), 1))})
    near_value = sum(a["amount"] for a, _ in near)
    top, count = reasons.most_common(1)[0]
    best = max(by_ind, key=lambda i: i["eligible_pct"])
    ideas = [
        {"title": "Win back near-miss applicants",
         "why": f"{len(near)} applicants miss exactly one rule, about ${near_value:,.0f} in requested loans.",
         "pitch": "You're one step from qualifying. Here is the single change that gets you approved, and we'll hold your file."},
        {"title": PLAYS[top][0], "why": f"{top} is the most common reason applicants fall short ({count} of {len(rows)}).",
         "pitch": PLAYS[top][1]},
        {"title": f"Partner with {best['industry'].lower()} associations",
         "why": f"{best['industry']} has the highest eligibility rate ({best['eligible_pct']}%), so referrals convert best.",
         "pitch": "Offer members a co-branded pre-qualification page and a fast-track review."},
    ]
    return {"synthetic": True, "total": len(rows), "eligible": len(eligible), "near_miss": len(near),
            "near_miss_value": near_value, "decline_reasons": [{"reason": k, "count": v} for k, v in reasons.most_common()],
            "by_industry": by_ind, "ideas": ideas}
