import sqlite3
from datetime import datetime
from pathlib import Path
from .models import PROCESS_STAGES

_DB_PATH = str(Path(__file__).parent.parent / "energy.db")


def _conn():
    c = sqlite3.connect(_DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db(db_path: str = _DB_PATH):
    global _DB_PATH
    _DB_PATH = db_path
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS process_instances (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                process_type TEXT NOT NULL,
                period_label TEXT NOT NULL,
                created_at   TEXT NOT NULL,
                notes        TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS stage_completions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                instance_id  INTEGER NOT NULL REFERENCES process_instances(id) ON DELETE CASCADE,
                stage_key    TEXT NOT NULL,
                completed    INTEGER NOT NULL DEFAULT 0,
                completed_at TEXT,
                amount       REAL,
                notes        TEXT
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_sc_instance ON stage_completions(instance_id)")


def create_instance(process_type: str, period_label: str, notes: str = "") -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO process_instances (process_type, period_label, created_at, notes) VALUES (?,?,?,?)",
            (process_type, period_label, datetime.now().isoformat(), notes),
        )
        instance_id = cur.lastrowid
        for stage in PROCESS_STAGES[process_type]:
            c.execute(
                "INSERT INTO stage_completions (instance_id, stage_key, completed) VALUES (?,?,0)",
                (instance_id, stage["key"]),
            )
        return instance_id


def get_instances(process_type: str) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM process_instances WHERE process_type=? ORDER BY created_at DESC",
            (process_type,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_instance(instance_id: int) -> dict:
    with _conn() as c:
        row = c.execute("SELECT * FROM process_instances WHERE id=?", (instance_id,)).fetchone()
    return dict(row) if row else {}


def get_stages(instance_id: int) -> dict[str, dict]:
    """Returns {stage_key: row_dict} for all stages of an instance."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM stage_completions WHERE instance_id=?", (instance_id,)
        ).fetchall()
    return {r["stage_key"]: dict(r) for r in rows}


def update_stage(instance_id: int, stage_key: str, completed: bool,
                 completed_at: str = None, amount: float = None, notes: str = None):
    with _conn() as c:
        c.execute(
            """UPDATE stage_completions
               SET completed=?, completed_at=?, amount=?, notes=?
               WHERE instance_id=? AND stage_key=?""",
            (int(completed), completed_at, amount, notes, instance_id, stage_key),
        )


def delete_instance(instance_id: int):
    with _conn() as c:
        c.execute("DELETE FROM process_instances WHERE id=?", (instance_id,))


def get_summary() -> dict:
    """Returns counts per process type: active, completed, awaiting_stage."""
    from .models import PROCESS_STAGES, is_instance_complete, current_stage_in_track, stages_by_track, PROCESS_LABELS
    result = {}
    for ptype, label in PROCESS_LABELS.items():
        instances = get_instances(ptype)
        active = completed = 0
        next_steps = []
        for inst in instances:
            stages_comp = get_stages(inst["id"])
            if is_instance_complete(stages_comp, ptype):
                completed += 1
            else:
                active += 1
                for track, tstages in stages_by_track(ptype).items():
                    key = current_stage_in_track(stages_comp, tstages)
                    if key:
                        stage_def = next(s for s in PROCESS_STAGES[ptype] if s["key"] == key)
                        next_steps.append(stage_def["label"])
        result[ptype] = {
            "label": label,
            "active": active,
            "completed": completed,
            "next_steps": list(dict.fromkeys(next_steps)),
        }
    return result


def get_recent_completions(limit: int = 5) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            """SELECT sc.*, pi.process_type, pi.period_label
               FROM stage_completions sc
               JOIN process_instances pi ON pi.id = sc.instance_id
               WHERE sc.completed=1 AND sc.completed_at IS NOT NULL
               ORDER BY sc.completed_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
