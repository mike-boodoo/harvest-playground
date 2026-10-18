"""
User flow: explore topic overlap until exhausted, see what each UI is
allowed to show, and leave with a model learned from every topic agent
in reach. Then benchmark every model by its peers instead of a service.

  1. JOIN       the user's own topics, and which ones they moderate
  2. EXPLORE    hop by hop through shared topics, as the user's UI shows it
  3. EXHAUSTED  how many hops until nothing new, who is unreachable
  4. TAKEAWAY   the model the user leaves with, and their stake in topic agents
  5. BENCHMARK  peer-judged arena -> EigenTrust, anchored on moderators,
                compared with an unanchored run and a naive win rate
  6. MODERATORS why first arrivals end up with the largest takeaway

Everything is simulated (see core/session_flow.py); batches are really
signed and pooled.

Run: python examples/degrees_of_separation_demo.py
"""

import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.peer_benchmark import eigentrust, peer_judgments, spearman, win_rate
from core.session_flow import build_community, run_session, topic_agent_quality


def show_session(c, ego):
    r = run_session(c, ego)
    exp, tk = r.exploration, r.takeaway
    mods = f" (moderator of {', '.join('#' + t for t in tk.moderates)})" if tk.moderates else ""
    print(f"\n=== {ego}'s session ===")
    print(f"1 JOIN       on {', '.join('#' + t for t in c.graph.topics_of(ego))}{mods}")
    for v in r.views:
        topics = ", ".join("#" + t for t in v.topics_shown) or "—"
        hidden = f" (+{v.topics_suppressed} topic(s) too small to name)" if v.topics_suppressed else ""
        who = ", ".join(v.people) if v.people else f"{v.people_count} people, not named"
        print(f"2 EXPLORE    degree {v.degree}: via {topics}{hidden}: {who}")
    print(f"3 EXHAUSTED  after {exp.exhausted_after} hop(s): reached {len(exp.user_degree) - 1} people, "
          f"{len(exp.topic_degree)} topics; {len(exp.unreachable)} people in other components")
    print(f"             smallest anonymity set if degree+topics were ever shown together: "
          f"{r.smallest_anonymity_set}")
    best = sorted(tk.per_topic.items(), key=lambda kv: -kv[1])[:3]
    print(f"4 TAKEAWAY   coverage {tk.coverage:.2f} across all topics; strongest: "
          + ", ".join(f"#{t} {q:.2f}" for t, q in best))
    print(f"             stake in topic agents: arrival-order {tk.stake['arrival']:.2f}, "
          f"Shapley {tk.stake['shapley']:.2f}")


def main():
    c = build_community(seed=0)
    g = c.graph

    for ego in ("ana", "ben", "hana"):
        show_session(c, ego)

    print("\n=== every user's takeaway ===")
    print(f"{'user':<11}{'hops':>5}{'reach':>6}{'coverage':>9}{'stake(arr)':>11}{'stake(shap)':>12}  moderates")
    for u in g.users():
        r = run_session(c, u)
        tk = r.takeaway
        print(f"{u:<11}{r.exploration.exhausted_after:>5}{len(r.exploration.user_degree) - 1:>6}"
              f"{tk.coverage:>9.2f}{tk.stake['arrival']:>11.2f}{tk.stake['shapley']:>12.2f}  "
              f"{', '.join(tk.moderates) or '—'}")
    print(f"graph diameter (most hops anyone needs to exhaust): {g.diameter()}")

    print("\n=== 5 BENCHMARK: models scored by each other ===")
    true = {u: sum(c.skill[u][t] for t in g.topics_of(u)) / len(g.topics_of(u)) for u in g.users()}
    judgments = peer_judgments(g, c.skill, random.Random(0), colluders=c.colluders)
    moderators = sorted({g.moderator(t) for t in g.topics()})
    runs = {
        "EigenTrust, moderators pre-trusted": eigentrust(g.users(), judgments, pretrusted=moderators),
        "EigenTrust, no anchor": eigentrust(g.users(), judgments),
    }
    wr = win_rate(g.users(), judgments)
    print(f"{len(judgments)} peer judgments; colluders: {', '.join(sorted(c.colluders))}")
    for name, (t, iters, reach) in runs.items():
        rank = sorted(t, key=lambda u: -t[u])
        print(f"  {name:<36} rank corr. with true skill {spearman(t, true):+.2f}; colluders ranked "
              f"{', '.join(str(rank.index(s) + 1) for s in sorted(c.colluders))} of {len(rank)}; "
              f"converged in {iters} iterations; trust spread {reach} judging hop(s) from the anchors")
    rank = sorted(wr, key=lambda u: -wr[u])
    print(f"  {'naive win rate':<36} rank corr. with true skill {spearman(wr, true):+.2f}; colluders ranked "
          f"{', '.join(str(rank.index(s) + 1) for s in sorted(c.colluders))} of {len(rank)}")

    print("\n=== 6 MODERATORS: first arrival vs. most skilled ===")
    by_ownership = topic_agent_quality(g, c.skill, c.shares["arrival"])
    print(f"{'topic':<23}{'moderator':<11}{'share(arr)':>11}{'share(shap)':>12}  {'most skilled':<13}"
          f"agent mixed by: trust  ownership")
    for topic in g.topics():
        mod = g.moderator(topic)
        best = max(g.members_of(topic), key=lambda u: c.skill[u][topic])
        print(f"#{topic:<22}{mod:<11}{c.shares['arrival'][topic][mod]:>11.2f}"
              f"{c.shares['shapley'][topic][mod]:>12.2f}  {best:<13}"
              f"{c.agent_quality[topic]:>20.2f}{by_ownership[topic]:>11.2f}")


if __name__ == "__main__":
    main()
