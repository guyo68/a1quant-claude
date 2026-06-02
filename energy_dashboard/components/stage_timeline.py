from datetime import date
import streamlit as st
from ..models import PROCESS_STAGES, TRACK_LABELS, stages_by_track, current_stage_in_track
from ..db import get_stages, update_stage


def render(instance_id: int, process_type: str, readonly: bool = False):
    stages_comp = get_stages(instance_id)
    tracks = stages_by_track(process_type)
    track_list = list(tracks.items())

    cols = st.columns(len(track_list))
    for col, (track, tstages) in zip(cols, track_list):
        label = TRACK_LABELS.get(track, track)
        col.markdown(f"**{label}**")
        current_key = current_stage_in_track(stages_comp, tstages) if not readonly else None

        for stage in tstages:
            comp = stages_comp.get(stage["key"], {})
            done = bool(comp.get("completed"))
            is_current = (stage["key"] == current_key)
            deps_met = all(stages_comp.get(d, {}).get("completed") for d in stage["depends_on"])

            if done:
                amt_str = f" — ₪{comp['amount']:,.0f}" if comp.get("amount") else ""
                date_str = comp.get("completed_at", "")[:10] if comp.get("completed_at") else ""
                col.success(f"✅ {stage['label']}{amt_str}  \n{date_str}")
            elif is_current and not readonly:
                with col.container(border=True):
                    st.markdown(f"**{stage['label']}**")
                    picked = st.date_input("תאריך", value=date.today(),
                                           key=f"date_{instance_id}_{stage['key']}")
                    amount = None
                    if stage["has_amount"]:
                        amount = st.number_input("סכום (₪)", min_value=0.0, step=100.0,
                                                  key=f"amt_{instance_id}_{stage['key']}")
                    note = st.text_input("הערה (אופציונלי)", key=f"note_{instance_id}_{stage['key']}")
                    if st.button("סמן כהושלם ✓", key=f"done_{instance_id}_{stage['key']}"):
                        update_stage(instance_id, stage["key"],
                                     completed=True,
                                     completed_at=picked.isoformat(),
                                     amount=amount if stage["has_amount"] else None,
                                     notes=note or None)
                        st.rerun()
            elif not deps_met:
                blocking = [s["label"] for s in PROCESS_STAGES[process_type]
                            if s["key"] in stage["depends_on"]
                            and not stages_comp.get(s["key"], {}).get("completed")]
                col.caption(f"🔒 {stage['label']}  \nממתין ל: {', '.join(blocking)}")
            else:
                col.caption(f"⬜ {stage['label']}")
