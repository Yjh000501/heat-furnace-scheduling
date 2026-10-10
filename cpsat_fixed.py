# cpsat_fixed.py
import pandas as pd
from ortools.sat.python import cp_model


def solve_cpsat_fixed(
    wait_times, process_times, rough_times, heat_times,
    order,
    time_limit=30, capacity=5,
    heat_furnace_count=3, heat_furnace_capacity=30,
    min_gap_in_furnace=100, max_gap_in_furnace=1800,
    max_wait_in_furnace=600,
    max_start_at_once=2,
    penalty_gap=10, penalty_wait=1, penalty_balance=20,
):
    n = len(order)
    assert sorted(order) == list(range(n))

    horizon = sum(heat_times) + sum(rough_times) + sum(wait_times) + sum(process_times)
    model = cp_model.CpModel()

    # ---------- 加热炉 ----------
    hs = [model.NewIntVar(0, horizon, f"hs_{i}") for i in range(n)]
    he = [model.NewIntVar(0, horizon, f"he_{i}") for i in range(n)]
    furnace = [model.NewIntVar(0, heat_furnace_count - 1, f"furnace_{i}") for i in range(n)]

    is_in = {}
    for i in range(n):
        for k in range(heat_furnace_count):
            is_in[(i, k)] = model.NewBoolVar(f"is_{i}_in_{k}")
            model.Add(furnace[i] == k).OnlyEnforceIf(is_in[(i, k)])
            model.Add(furnace[i] != k).OnlyEnforceIf(is_in[(i, k)].Not())

    duration = [model.NewIntVar(0, horizon, f"dur_{i}") for i in range(n)]
    for i in range(n):
        model.Add(duration[i] == he[i] - hs[i])
        model.Add(duration[i] >= heat_times[i])

    # 每炉容量
    for k in range(heat_furnace_count):
        intervals_k = []
        for i in range(n):
            interval = model.NewOptionalIntervalVar(
                hs[i], duration[i], he[i], is_in[(i, k)], f"heat_{i}_{k}"
            )
            intervals_k.append(interval)
        model.AddCumulative(intervals_k, [1] * n, heat_furnace_capacity)

    # 每炉同一时刻最多 max_start_at_once 块开始加热
    for k in range(heat_furnace_count):
        start_intervals_k = []
        for i in range(n):
            start_interval = model.NewOptionalIntervalVar(
                hs[i], 1, hs[i] + 1, is_in[(i, k)], f"start_{i}_{k}"
            )
            start_intervals_k.append(start_interval)
        model.AddCumulative(start_intervals_k, [1] * n, max_start_at_once)

    # ---------- 粗轧 ----------
    rs = [model.NewIntVar(0, horizon, f"rs_{i}") for i in range(n)]
    re = [model.NewIntVar(0, horizon, f"re_{i}") for i in range(n)]
    for i in range(n):
        model.Add(he[i] == rs[i])
    rough_interval = [
        model.NewIntervalVar(rs[i], rough_times[i], re[i], f"rough_{i}")
        for i in range(n)
    ]
    model.AddNoOverlap(rough_interval)

    # ---------- 待温 ----------
    ws = [model.NewIntVar(0, horizon, f"ws_{i}") for i in range(n)]
    we = [model.NewIntVar(0, horizon, f"we_{i}") for i in range(n)]
    for i in range(n):
        model.Add(re[i] == ws[i])
        model.Add(we[i] == ws[i] + wait_times[i])
    wait_interval = [
        model.NewIntervalVar(ws[i], wait_times[i], we[i], f"wait_{i}")
        for i in range(n)
    ]
    model.AddCumulative(wait_interval, [1] * n, capacity)

    # ---------- 精轧 ----------
    ps = [model.NewIntVar(0, horizon, f"ps_{i}") for i in range(n)]
    pe = [model.NewIntVar(0, horizon, f"pe_{i}") for i in range(n)]
    for i in range(n):
        model.Add(we[i] == ps[i])
    proc_interval = [
        model.NewIntervalVar(ps[i], process_times[i], pe[i], f"proc_{i}")
        for i in range(n)
    ]
    model.AddNoOverlap(proc_interval)

    # ---------- 固定顺序：粗轧/待温/精轧严格按顺序 ----------
    for pos in range(1, n):
        prev = order[pos - 1]
        cur = order[pos]
        model.Add(rs[cur] >= re[prev])
        model.Add(ws[cur] >= we[prev])
        model.Add(ps[cur] >= pe[prev])

    # ---------- 同炉 FIFO（硬） ----------
    for idx_i in range(n):
        for idx_j in range(idx_i + 1, n):
            i = order[idx_i]
            j = order[idx_j]
            for k in range(heat_furnace_count):
                both_in_k = model.NewBoolVar(f"both_{i}_{j}_{k}")
                model.AddBoolAnd([is_in[(i, k)], is_in[(j, k)]]).OnlyEnforceIf(both_in_k)
                model.AddBoolOr([
                    is_in[(i, k)].Not(),
                    is_in[(j, k)].Not(),
                    both_in_k
                ])

                # i 是否比 j 严格先入炉
                i_before_j = model.NewBoolVar(f"i_before_j_{i}_{j}_{k}")
                model.Add(hs[i] < hs[j]).OnlyEnforceIf([both_in_k, i_before_j])
                model.Add(hs[j] <= hs[i]).OnlyEnforceIf([both_in_k, i_before_j.Not()])

                # FIFO：i 先入炉 → i 必须先出炉、先粗轧
                model.Add(he[i] <= he[j]).OnlyEnforceIf([both_in_k, i_before_j])
                model.Add(he[j] <= he[i]).OnlyEnforceIf([both_in_k, i_before_j.Not()])

                model.Add(rs[i] <= rs[j]).OnlyEnforceIf([both_in_k, i_before_j])
                model.Add(rs[j] <= rs[i]).OnlyEnforceIf([both_in_k, i_before_j.Not()])

    # ---------- 炉内等待（软） ----------
    over_wait_all = []
    for i in range(n):
        wait_in_furnace = model.NewIntVar(0, horizon, f"wait_in_furnace_{i}")
        model.Add(wait_in_furnace == he[i] - hs[i] - heat_times[i])
        over_wait = model.NewIntVar(0, horizon, f"over_wait_{i}")
        model.Add(over_wait >= wait_in_furnace - max_wait_in_furnace)
        model.Add(over_wait >= 0)
        over_wait_all.append(over_wait)

    # ---------- 均衡（软） ----------
    furnace_load = [model.NewIntVar(0, n, f"load_{k}") for k in range(heat_furnace_count)]
    for k in range(heat_furnace_count):
        model.Add(furnace_load[k] == sum(is_in[(i, k)] for i in range(n)))
    max_load = model.NewIntVar(0, n, "max_load")
    min_load = model.NewIntVar(0, n, "min_load")
    model.AddMaxEquality(max_load, furnace_load)
    model.AddMinEquality(min_load, furnace_load)
    imbalance = model.NewIntVar(0, n, "imbalance")
    model.Add(imbalance == max_load - min_load)

    # ---------- 目标 ----------
    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, pe)
    model.Minimize(
        makespan
        + penalty_wait * sum(over_wait_all)
        + penalty_balance * imbalance
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    rows = []
    for pos, i in enumerate(order, start=1):
        hs_val = solver.Value(hs[i])
        he_val = solver.Value(he[i])
        rows.append({
            "顺序": pos,
            "板坯": f"板坯 {i + 1}",
            "加热炉": f"加热炉 {solver.Value(furnace[i]) + 1}",
            "加热时间(min)": heat_times[i] // 60,
            "炉内等待(s)": he_val - hs_val - heat_times[i],
            "粗轧时间(s)": rough_times[i],
            "待温时间(s)": wait_times[i],
            "精轧时间(s)": process_times[i],
            "加热开始(s)": hs_val,
            "加热结束(s)": he_val,
            "粗轧开始(s)": solver.Value(rs[i]),
            "粗轧结束(s)": solver.Value(re[i]),
            "待温开始(s)": solver.Value(ws[i]),
            "待温结束(s)": solver.Value(we[i]),
            "精轧开始(s)": solver.Value(ps[i]),
            "精轧结束(s)": solver.Value(pe[i]),
        })

    df = pd.DataFrame(rows)
    return {
        "status": solver.StatusName(status),
        "makespan": df["精轧结束(s)"].max(),
        "table": df,
    }