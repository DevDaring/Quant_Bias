from quantbias.allocate import GreedyAllocator, Objective, ranked_map, random_map, budget_for_uniform
from quantbias.quantization import PrecisionMap, account_bytes


def test_ranked_and_random_respect_budget(tiny_adapter):
    a = tiny_adapter
    budget = int(budget_for_uniform(a, 4) * 1.1)
    pm = ranked_map(a, {c.id: i for i, c in enumerate(a.components)}, budget)
    assert account_bytes(a, pm).total <= budget and any(v == 8 for v in pm.values())
    rm = random_map(a, budget, 0)
    assert account_bytes(a, rm).total <= budget


def test_greedy_finds_planted_sensitive_site(tiny_adapter):
    a = tiny_adapter
    sensitive = a.components[3].id
    budget = int(budget_for_uniform(a, 4) * 1.05)

    def score(pm: PrecisionMap):
        # harm is high unless the sensitive site is at >= 8 bits; utility flat
        H = 0.30 if pm.bits_of(sensitive) < 8 else 0.05
        return {"H": H, "A": 0.0, "gap": 0.0, "acc": 0.9, "ppl": 10.0}

    alloc = GreedyAllocator(a, score, budget, Objective(), max_steps=5, max_evals=100,
                            utility_reference={"acc": 0.9, "ppl": 10.0}, log_fn=None)
    res = alloc.run(PrecisionMap.uniform(a, 4))
    assert res.precision_map[sensitive] >= 8 and res.bytes_total <= budget and res.objective < 0.1
    assert any(s.accepted for s in res.steps)
