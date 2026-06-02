"""Generic view used by all three process pages."""
import streamlit as st
from ..db import create_instance
from ..models import PROCESS_LABELS
from ..components import instance_list, stage_timeline


def render(process_type: str):
    st.header(PROCESS_LABELS[process_type])
    tab_active, tab_history, tab_new = st.tabs(["פעיל", "היסטוריה", "מקרה חדש"])

    with tab_active:
        selected = instance_list.render(process_type, active_only=True)
        if selected or st.session_state.get(f"selected_{process_type}"):
            if selected:
                st.session_state[f"selected_{process_type}"] = selected
            inst_id = st.session_state[f"selected_{process_type}"]
            st.divider()
            stage_timeline.render(inst_id, process_type)

    with tab_history:
        selected_h = instance_list.render(process_type, active_only=False)
        if selected_h or st.session_state.get(f"selected_h_{process_type}"):
            if selected_h:
                st.session_state[f"selected_h_{process_type}"] = selected_h
            inst_id = st.session_state[f"selected_h_{process_type}"]
            st.divider()
            stage_timeline.render(inst_id, process_type, readonly=True)

    with tab_new:
        with st.form(f"new_{process_type}"):
            period = st.text_input("תקופה (למשל: אפריל 2025)")
            notes = st.text_input("הערות (אופציונלי)")
            submitted = st.form_submit_button("צור מקרה חדש")
            if submitted:
                if not period.strip():
                    st.error("יש להזין תקופה")
                else:
                    create_instance(process_type, period.strip(), notes.strip())
                    st.success(f"נוצר מקרה חדש: {period}")
                    st.rerun()
