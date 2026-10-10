import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import random

from cpsat_global import solve_cpsat_global
from cpsat_fixed import solve_cpsat_fixed


# =========================================================
# 启发式排序
# =========================================================
def heuristic_order(wait_times, process_times, rough_times, heat_times):
    n = len(wait_times)
    return sorted(
        range(n),
        key=lambda i: (wait_times[i], heat_times[i], rough_times[i], process_times[i])
    )


# =========================================================
# 页面
# =========================================================
st.set_page_config(page_title="加热-粗轧-待温-精轧协同排产", layout="wide")
st.title("加热-粗轧-待温-精轧协同排产")
st.caption("启发式 + CP-SAT 全局搜索 / 启发式排序 + 固定顺序")

st.sidebar.header("参数设置")
n = st.sidebar.number_input("板坯数量", min_value=1, max_value=200, value=100, step=1)
heat_furnace_count = st.sidebar.number_input("加热炉数量", min_value=1, max_value=10, value=3, step=1)
heat_furnace_capacity = st.sidebar.number_input("每炉最多同时加热块数", min_value=1, max_value=200, value=30, step=1)
max_start_at_once = st.sidebar.number_input("每炉同一时刻最多进几块", min_value=1, max_value=10, value=2, step=1)
min_gap_in_furnace = st.sidebar.number_input("同炉相邻批次最小间隔(s)", min_value=0, max_value=600, value=100, step=10)
max_gap_in_furnace = st.sidebar.number_input("同炉相邻批次最大间隔(s)（软）", min_value=0, max_value=72000, value=1800, step=60)
max_wait_in_furnace = st.sidebar.number_input("炉内最大等待(s)（软）", min_value=0, max_value=3600, value=600, step=60)
capacity = st.sidebar.number_input("待温区容量", min_value=1, max_value=20, value=5, step=1)
time_limit = st.sidebar.number_input("CP-SAT 求解时间上限（秒）", min_value=1, max_value=600, value=30, step=1)
penalty_gap = st.sidebar.number_input("超最大间隔惩罚系数", min_value=0, max_value=100, value=10, step=1)
penalty_wait = st.sidebar.number_input("超等待惩罚系数", min_value=0, max_value=100, value=1, step=1)
penalty_balance = st.sidebar.number_input("均衡惩罚系数", min_value=0, max_value=100, value=20, step=1)

st.sidebar.markdown("---")
mode = st.sidebar.radio(
    "排产模式",
    ["启发式 + CP-SAT 全局搜索", "启发式排序 + 固定顺序"],
    index=0
)

st.subheader("1. 输入板坯数据")

random.seed(42)
default_heat_min = [random.randint(300, 400) for _ in range(200)]
default_heat = [h * 60 for h in default_heat_min]
default_rough = [150] * 200
default_wait = [random.choice([0, 100, 300, 500, 1000]) for _ in range(200)]
default_proc = [120] * 200


def get_default(lst, i, fallback):
    return lst[i] if i < len(lst) else fallback


input_data = []
for i in range(n):
    c0, c1, c2, c3, c4 = st.columns([1, 2, 2, 2, 2])
    c0.write(f"板坯 {i + 1}")
    h_min = c1.number_input(
        f"加热时间_{i} (min)", min_value=1,
        value=int(get_default(default_heat_min, i, 350)), step=10, key=f"h_{i}")
    r = c2.number_input(
        f"粗轧时间_{i}", min_value=1,
        value=int(get_default(default_rough, i, 150)), step=10, key=f"r_{i}")
    w = c3.number_input(
        f"待温时间_{i}", min_value=0,
        value=int(get_default(default_wait, i, 100)), step=10, key=f"w_{i}")
    p = c4.number_input(
        f"精轧时间_{i}", min_value=1,
        value=int(get_default(default_proc, i, 120)), step=10, key=f"p_{i}")
    input_data.append((h_min * 60, r, w, p))

st.subheader("2. 求解结果")

if st.button("开始排产"):
    heat_times = [x[0] for x in input_data]
    rough_times = [x[1] for x in input_data]
    wait_times = [x[2] for x in input_data]
    process_times = [x[3] for x in input_data]

    if mode == "启发式 + CP-SAT 全局搜索":
        with st.spinner("正在用启发式 + CP-SAT 全局搜索..."):
            result = solve_cpsat_global(
                wait_times, process_times, rough_times, heat_times,
                time_limit=time_limit, capacity=capacity,
                heat_furnace_count=heat_furnace_count,
                heat_furnace_capacity=heat_furnace_capacity,
                min_gap_in_furnace=min_gap_in_furnace,
                max_gap_in_furnace=max_gap_in_furnace,
                max_wait_in_furnace=max_wait_in_furnace,
                max_start_at_once=max_start_at_once,
                penalty_gap=penalty_gap,
                penalty_wait=penalty_wait,
                penalty_balance=penalty_balance,
            )
    else:
        init_order = heuristic_order(wait_times, process_times, rough_times, heat_times)
        with st.spinner("正在用 CP-SAT 按固定顺序求解..."):
            result = solve_cpsat_fixed(
                wait_times, process_times, rough_times, heat_times,
                init_order,
                time_limit=time_limit, capacity=capacity,
                heat_furnace_count=heat_furnace_count,
                heat_furnace_capacity=heat_furnace_capacity,
                min_gap_in_furnace=min_gap_in_furnace,
                max_gap_in_furnace=max_gap_in_furnace,
                max_wait_in_furnace=max_wait_in_furnace,
                max_start_at_once=max_start_at_once,
                penalty_gap=penalty_gap,
                penalty_wait=penalty_wait,
                penalty_balance=penalty_balance,
            )

    if result is None:
        st.error("未找到可行解，请检查输入数据或放宽容量限制。")
    else:
        st.success(f"求解状态：{result['status']}")
        col1, col2 = st.columns(2)
        col1.metric("makespan", f"{result['makespan']} s")
        col2.metric("板坯数量", n)

        st.markdown("### 排产顺序")
        st.dataframe(result["table"])

        df = result["table"].copy()

        st.markdown("### 整体时间线")
        fig_all = go.Figure()
        for _, row in df.iterrows():
            fig_all.add_trace(go.Bar(
                x=[row["加热时间(min)"] * 60], y=[row["板坯"]],
                base=[row["加热开始(s)"]], orientation="h",
                name="加热", marker_color="gold", showlegend=False,
            ))
            if row["炉内等待(s)"] > 0:
                fig_all.add_trace(go.Bar(
                    x=[row["炉内等待(s)"]], y=[row["板坯"]],
                    base=[row["加热开始(s)"] + row["加热时间(min)"] * 60],
                    orientation="h", name="炉内等待", marker_color="orange",
                    showlegend=False,
                ))
            fig_all.add_trace(go.Bar(
                x=[row["粗轧时间(s)"]], y=[row["板坯"]],
                base=[row["粗轧开始(s)"]], orientation="h",
                name="粗轧", marker_color="lightgreen", showlegend=False,
            ))
            fig_all.add_trace(go.Bar(
                x=[row["待温时间(s)"]], y=[row["板坯"]],
                base=[row["待温开始(s)"]], orientation="h",
                name="待温", marker_color="lightblue", showlegend=False,
            ))
            fig_all.add_trace(go.Bar(
                x=[row["精轧时间(s)"]], y=[row["板坯"]],
                base=[row["精轧开始(s)"]], orientation="h",
                name="精轧", marker_color="salmon", showlegend=False,
            ))
        fig_all.update_layout(
            barmode="overlay", xaxis_title="时间 (s)", yaxis_title="板坯",
            height=max(600, 20 * n),
        )
        fig_all.update_yaxes(categoryorder="array",
            categoryarray=list(df["板坯"])[::-1])
        st.plotly_chart(fig_all, use_container_width=True)

        st.markdown("### 按加热炉分组（按进炉顺序）")
        for fid in sorted(df["加热炉"].unique()):
            sub = df[df["加热炉"] == fid].copy()
            sub = sub.sort_values("加热开始(s)").reset_index(drop=True)
            st.markdown(f"**{fid}** （共 {len(sub)} 块）")
            fig_f = go.Figure()
            for _, row in sub.iterrows():
                fig_f.add_trace(go.Bar(
                    x=[row["加热时间(min)"] * 60], y=[row["板坯"]],
                    base=[row["加热开始(s)"]], orientation="h",
                    name="加热", marker_color="gold", showlegend=False,
                ))
                if row["炉内等待(s)"] > 0:
                    fig_f.add_trace(go.Bar(
                        x=[row["炉内等待(s)"]], y=[row["板坯"]],
                        base=[row["加热开始(s)"] + row["加热时间(min)"] * 60],
                        orientation="h", name="炉内等待", marker_color="orange",
                        showlegend=False,
                    ))
                fig_f.add_trace(go.Bar(
                    x=[row["粗轧时间(s)"]], y=[row["板坯"]],
                    base=[row["粗轧开始(s)"]], orientation="h",
                    name="粗轧", marker_color="lightgreen", showlegend=False,
                ))
                fig_f.add_trace(go.Bar(
                    x=[row["待温时间(s)"]], y=[row["板坯"]],
                    base=[row["待温开始(s)"]], orientation="h",
                    name="待温", marker_color="lightblue", showlegend=False,
                ))
                fig_f.add_trace(go.Bar(
                    x=[row["精轧时间(s)"]], y=[row["板坯"]],
                    base=[row["精轧开始(s)"]], orientation="h",
                    name="精轧", marker_color="salmon", showlegend=False,
                ))
            fig_f.update_layout(
                barmode="overlay", xaxis_title="时间 (s)", yaxis_title="板坯",
                height=max(400, 20 * len(sub)),
            )
            fig_f.update_yaxes(categoryorder="array",
                categoryarray=list(sub["板坯"])[::-1])
            st.plotly_chart(fig_f, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "下载排产结果 CSV",
            data=csv,
            file_name="排产结果.csv",
            mime="text/csv",
        )