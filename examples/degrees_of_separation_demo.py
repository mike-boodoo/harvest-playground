"""
User flow: explore topic overlap until exhausted, see what each UI is
allowed to show, and leave with a model learned from every topic agent
in reach. Then benchmark every model by its peers instead of a service.

  1. JOIN       the user's own topics, and which ones they moderate
  2. EXPLORE    hop by hop through shared topics, as the user's UI shows it
  3. EXHAUSTED  how many hops until nothing new, who is unreachable
  4. TAKEAWAY   the model the user leaves with, and their stake in topic agents
  5. BENCHMARK  peer-judged arena -> EigenTrust anchored on the topic
                agents' own top-k picks (nobody designated), compared with
                no anchor, designated first arrivals, and a naive win rate
  6. IMPLICIT MODERATORS  who each topic agent picks, what they own, and
                how fast a member can work it out from the agent's answers

Everything is simulated (see core/session_flow.py); batches are really
signed and pooled.

Run: python examples/degrees_of_separation_demo.py
"""

import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.peer_benchmark import eigentrust, peer_judgments, spearman, win_rate
from core.session_flow import build_community, inference_accuracy, run_session


def show_session(c, ego):
    r = run_session(c, ego)
    exp, tk = r.exploration, r.takeaway
    mods = f" (picked by the agent on {', '.join('#' + t for t in tk.selected_on)})" if tk.selected_on else ""
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
    print(f"             stake in topic agents: by the agents' picks {tk.stake['selection']:.2f} "
          f"(for comparison: arrival order {tk.stake['arrival']:.2f}, Shapley {tk.stake['shapley']:.2f})")


def main():
    c = build_community(seed=0)
    g = c.graph

    for ego in ("ana", "ben", "hana"):
        show_session(c, ego)

    print("\n=== every user's takeaway ===")
    print(f"{'user':<11}{'hops':>5}{'reach':>6}{'coverage':>9}{'stake(pick)':>12}{'stake(arr)':>11}"
          f"{'stake(shap)':>12}  picked on")
    for u in g.users():
        r = run_session(c, u)
        tk = r.takeaway
        print(f"{u:<11}{r.exploration.exhausted_after:>5}{len(r.exploration.user_degree) - 1:>6}"
              f"{tk.coverage:>9.2f}{tk.stake['selection']:>12.2f}{tk.stake['arrival']:>11.2f}"
              f"{tk.stake['shapley']:>12.2f}  {', '.join(tk.selected_on) or '—'}")
    print(f"graph diameter (most hops anyone needs to exhaust): {g.diameter()}")

    print("\n=== 5 BENCHMARK: models scored by each other ===")
    true = {u: sum(c.skill[u][t] for t in g.topics_of(u)) / len(g.topics_of(u)) for u in g.users()}
    judgments = peer_judgments(g, c.skill, random.Random(0), colluders=c.colluders)
    picked = sorted({u for us in c.selection.top_k.values() for u in us})
    first = sorted({g.first_arrival(t) for t in g.topics()})
    runs = {
        "anchored on agents' picks": eigentrust(g.users(), judgments, pretrusted=picked),
        "no anchor": eigentrust(g.users(), judgments),
        "designated first arrivals": eigentrust(g.users(), judgments, pretrusted=first),
    }
    print(f"agents' picks settled after {c.selection.rounds} rounds"
          f"{'' if c.selection.converged else ' (did not converge)'}, starting from no anchor")
    wr = win_rate(g.users(), judgments)
    print(f"{len(judgments)} peer judgments; colluders: {', '.join(sorted(c.colluders))}")
    for name, (t, iters, _) in runs.items():
        rank = sorted(t, key=lambda u: -t[u])
        print(f"  EigenTrust, {name:<25} rank corr. with true skill {spearman(t, true):+.2f}; colluders ranked "
              f"{', '.join(str(rank.index(s) + 1) for s in sorted(c.colluders))} of {len(rank)}; "
              f"converged in {iters} iterations")
    rank = sorted(wr, key=lambda u: -wr[u])
    print(f"  {'naive win rate':<37} rank corr. with true skill {spearman(wr, true):+.2f}; colluders ranked "
          f"{', '.join(str(rank.index(s) + 1) for s in sorted(c.colluders))} of {len(rank)}")

    print("\n=== 6 IMPLICIT MODERATORS: nobody is designated; each agent picks its top-k ===")
    print(f"{'topic':<23}{'agent picks (share)':<30}{'first arrival':<15}{'most skilled':<14}agent quality")
    for topic in g.topics():
        picks = ", ".join(f"{u} {c.shares['selection'][topic][u]:.2f}" for u in c.selection.top_k[topic])
        best = max(g.members_of(topic), key=lambda u: c.skill[u][topic])
        print(f"#{topic:<22}{picks:<30}{g.first_arrival(topic):<15}{best:<14}{c.agent_quality[topic]:.2f}")
    print("Never shown on screen, but a member who compares the agent's answers with")
    print("everyone's on a shared topic names the agent's picks correctly:")
    print("  " + ", ".join(f"{n} prompts {inference_accuracy(c, n):.0%}" for n in (3, 5, 10, 20)))


if __name__ == "__main__":
    main()
