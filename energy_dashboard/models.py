from dataclasses import dataclass, field
from typing import Optional

PROCESS_LABELS = {
    "pv_inside": "ייצור PV במחלק",
    "pv_outside": "ייצור PV מחוץ למחלק",
    "generator": "השלת גנרטור",
}

# Each stage: key, label, track, depends_on (list of stage keys), has_amount
PROCESS_STAGES = {
    "pv_inside": [
        # Main track
        {"key": "collect_data",       "label": "איסוף נתונים",                                    "track": "main",         "depends_on": [],                  "has_amount": False},
        {"key": "calculate",          "label": "חישוב",                                           "track": "main",         "depends_on": ["collect_data"],     "has_amount": False},
        {"key": "consolidate",        "label": "ריכוז",                                           "track": "main",         "depends_on": ["calculate"],        "has_amount": False},
        {"key": "create_affidavits",  "label": "יצירת תצהירים",                                   "track": "main",         "depends_on": ["consolidate"],      "has_amount": False},
        {"key": "send_email",         "label": "שליחה במייל לחח\"י",                              "track": "main",         "depends_on": ["create_affidavits"],"has_amount": False},
        # Track A — billing recommendation
        {"key": "receive_rec",        "label": "קבלת המלצת חשבון מחח\"י",                        "track": "track_rec",    "depends_on": ["send_email"],       "has_amount": False},
        {"key": "issue_invoice_ec",   "label": "הוצאת חשבונית לחח\"י",                           "track": "track_rec",    "depends_on": ["receive_rec"],      "has_amount": True},
        {"key": "receive_payment_ec", "label": "קבלת תשלום מחח\"י",                              "track": "track_rec",    "depends_on": ["issue_invoice_ec"], "has_amount": True},
        # Track B — consumption bill
        {"key": "receive_bill",       "label": "קבלת חשבון מחח\"י",                              "track": "track_bill",   "depends_on": ["send_email"],       "has_amount": False},
        {"key": "record_books_bill",  "label": "רישום בספרים",                                   "track": "track_bill",   "depends_on": ["receive_bill"],     "has_amount": False},
        {"key": "verify_quantity",    "label": "וידוא תואם לכמות שיוצרה",                        "track": "track_bill",   "depends_on": ["record_books_bill"],"has_amount": False},
        # Track C — owners charges
        {"key": "owners_charges",     "label": "קבלת חיובים מבעלי המערכת (מענית, אוכמנית)",     "track": "track_owners", "depends_on": ["receive_rec"],      "has_amount": True},
    ],
    "pv_outside": [
        # Track A — production recommendation (EC initiates)
        {"key": "receive_prod_rec",   "label": "קבלת המלצת חשבון על ייצור מחח\"י",              "track": "track_prod",   "depends_on": [],                      "has_amount": False},
        {"key": "meinit_invoice",     "label": "מענית מוציאה חשבונית",                          "track": "track_prod",   "depends_on": ["receive_prod_rec"],    "has_amount": True},
        {"key": "ec_pays_meinit",     "label": "חח\"י משלמת למענית",                            "track": "track_prod",   "depends_on": ["meinit_invoice"],      "has_amount": True},
        # Track B — consumption bill (direct debit)
        {"key": "receive_cons_bill",  "label": "קבלת חשבון על צריכה (הוראת קבע)",              "track": "track_cons",   "depends_on": [],                      "has_amount": False},
        {"key": "record_books_cons",  "label": "רישום בספרים",                                  "track": "track_cons",   "depends_on": ["receive_cons_bill"],   "has_amount": False},
        {"key": "charge_tenant",      "label": "חיוב שוכר המבנה",                               "track": "track_cons",   "depends_on": ["receive_cons_bill"],   "has_amount": True},
        {"key": "receive_tenant_pmt", "label": "קבלת תשלום משוכר המבנה",                       "track": "track_cons",   "depends_on": ["charge_tenant"],       "has_amount": True},
    ],
    "generator": [
        {"key": "authority_notifies", "label": "רשות החשמל מודיעה למחלק",                       "track": "main",         "depends_on": [],                         "has_amount": False},
        {"key": "machlek_activates",  "label": "מחלק מפעיל גנרטור",                             "track": "main",         "depends_on": ["authority_notifies"],     "has_amount": False},
        {"key": "ec_issues_rec",      "label": "חח\"י מוציאה המלצת חשבון",                      "track": "main",         "depends_on": ["machlek_activates"],      "has_amount": False},
        {"key": "machlek_invoice",    "label": "מחלק מוציא חשבונית",                            "track": "main",         "depends_on": ["ec_issues_rec"],          "has_amount": True},
        {"key": "ec_pays_machlek",    "label": "חח\"י משלמת למחלק",                             "track": "main",         "depends_on": ["machlek_invoice"],        "has_amount": True},
    ],
}

TRACK_LABELS = {
    "main":         "שלבים ראשיים",
    "track_rec":    "מסלול א׳ — המלצת חשבון",
    "track_bill":   "מסלול ב׳ — חשבון צריכה",
    "track_owners": "מסלול ג׳ — חיוב בעלי מערכת",
    "track_prod":   "מסלול א׳ — ייצור",
    "track_cons":   "מסלול ב׳ — צריכה",
}


def get_tracks(process_type: str) -> list[str]:
    seen = []
    for s in PROCESS_STAGES[process_type]:
        if s["track"] not in seen:
            seen.append(s["track"])
    return seen


def stages_by_track(process_type: str) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for s in PROCESS_STAGES[process_type]:
        result.setdefault(s["track"], []).append(s)
    return result


def current_stage_in_track(stages_completion: dict[str, dict], track_stages: list[dict]) -> Optional[str]:
    """Return key of first incomplete stage in a track whose depends_on are all complete."""
    for s in track_stages:
        comp = stages_completion.get(s["key"], {})
        if comp.get("completed"):
            continue
        deps_met = all(stages_completion.get(d, {}).get("completed") for d in s["depends_on"])
        if deps_met:
            return s["key"]
    return None


def is_instance_complete(stages_completion: dict[str, dict], process_type: str) -> bool:
    return all(
        stages_completion.get(s["key"], {}).get("completed")
        for s in PROCESS_STAGES[process_type]
    )


def completion_pct(stages_completion: dict[str, dict], process_type: str) -> float:
    stages = PROCESS_STAGES[process_type]
    if not stages:
        return 0.0
    done = sum(1 for s in stages if stages_completion.get(s["key"], {}).get("completed"))
    return done / len(stages)
