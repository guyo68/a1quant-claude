import streamlit as st
from ..db import get_instances, get_stages, delete_instance
from ..models import completion_pct, is_instance_complete, PROCESS_LABELS


def render(process_type: str, active_only: bool = True) -> int | None:
    """Renders list of instances. Returns selected instance_id or None."""
    instances = get_instances(process_type)

    if active_only:
        instances = [i for i in instances
                     if not is_instance_complete(get_stages(i["id"]), process_type)]
    else:
        instances = [i for i in instances
                     if is_instance_complete(get_stages(i["id"]), process_type)]

    if not instances:
        st.info("אין מקרים" + (" פעילים" if active_only else " שהושלמו"))
        return None

    selected = None
    for inst in instances:
        stages_comp = get_stages(inst["id"])
        pct = completion_pct(stages_comp, process_type)
        label = inst["period_label"]
        note = f"  — {inst['notes']}" if inst.get("notes") else ""

        with st.expander(f"📋 {label}{note}  ({int(pct*100)}%)", expanded=False):
            st.progress(pct)
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("פתח / עדכן", key=f"open_{inst['id']}"):
                    selected = inst["id"]
            with col2:
                if st.button("🗑 מחק", key=f"del_{inst['id']}"):
                    delete_instance(inst["id"])
                    st.rerun()

    return selected
