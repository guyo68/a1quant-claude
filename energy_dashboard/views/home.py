import streamlit as st
from ..db import get_summary, get_recent_completions
from ..models import PROCESS_STAGES


def render():
    st.header("סיכום — ענף האנרגיה")
    summary = get_summary()

    cols = st.columns(3)
    for col, (ptype, data) in zip(cols, summary.items()):
        with col:
            st.metric(data["label"], f"{data['active']} פעיל", f"{data['completed']} הושלמו")
            if data["next_steps"]:
                st.caption("ממתין לשלב: " + " | ".join(data["next_steps"][:2]))

    st.divider()
    st.subheader("עדכונים אחרונים")
    recent = get_recent_completions(8)
    if not recent:
        st.info("אין עדכונים עדיין")
        return
    for r in recent:
        stage_def = next(
            (s for s in PROCESS_STAGES.get(r["process_type"], []) if s["key"] == r["stage_key"]),
            None,
        )
        stage_label = stage_def["label"] if stage_def else r["stage_key"]
        amt = f"  — ₪{r['amount']:,.0f}" if r.get("amount") else ""
        date_str = (r.get("completed_at") or "")[:10]
        st.write(f"✅ **{r['period_label']}** | {stage_label}{amt}  `{date_str}`")
