import os
import json
import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ─── Configure Spider dataset path ────────────────────────────────────────────
# Place Spider's tables.json here (download from https://yale-lily.github.io/spider)
SPIDER_TABLES_JSON = os.environ.get("SPIDER_TABLES_JSON", "./spider/tables.json")
SPIDER_DB_DIR      = os.environ.get("SPIDER_DB_DIR",      "./spider/database")

# ─── Module-level state (populated by initialize()) ───────────────────────────
embed_model    = None
schema_records = []
global_index   = None
db_indexes     = {}
all_db_ids     = []
fk_graph       = {}   # {db_id: {table_name: set of FK-connected table names}}
_initialized   = False


# ─── Parsing ──────────────────────────────────────────────────────────────────

def parse_tables_json(path, db_type="spider"):
    with open(path, encoding="utf-8") as f:
        dbs = json.load(f)

    records = []

    for db in dbs:
        db_id             = db["db_id"]
        table_names_orig  = db.get("table_names_original", [])
        col_names_orig    = db.get("column_names_original", [])
        table_names_human = db.get("table_names",  table_names_orig)
        col_names_human   = db.get("column_names", col_names_orig)
        col_types         = db.get("column_types", [])
        pks               = set(db.get("primary_keys", []))
        fks_raw           = db.get("foreign_keys", [])

        # col_idx → (other_table_name, other_col_name)
        fk_map = {}
        for pair in fks_raw:
            if len(pair) == 2:
                src, dst = pair
                if dst < len(col_names_orig):
                    other_tbl_idx = col_names_orig[dst][0]
                    other_col     = col_names_orig[dst][1]
                    other_tbl     = (table_names_orig[other_tbl_idx]
                                     if other_tbl_idx < len(table_names_orig) else "?")
                    fk_map[src]   = (other_tbl, other_col)

        table_cols = {i: [] for i in range(len(table_names_orig))}
        for col_idx, (tbl_idx, col_orig) in enumerate(col_names_orig):
            if tbl_idx == -1:
                continue
            col_human = (col_names_human[col_idx][1]
                         if col_idx < len(col_names_human) else col_orig)
            col_type  = col_types[col_idx] if col_idx < len(col_types) else "text"
            is_pk     = col_idx in pks
            is_fk     = col_idx in fk_map
            fk_ref    = fk_map.get(col_idx)
            table_cols[tbl_idx].append({
                "name"      : col_orig,
                "human_name": col_human,
                "type"      : col_type,
                "is_pk"     : is_pk,
                "is_fk"     : is_fk,
                "fk_ref"    : fk_ref,
            })

        for tbl_idx, tbl_orig in enumerate(table_names_orig):
            tbl_human = (table_names_human[tbl_idx]
                         if tbl_idx < len(table_names_human) else tbl_orig)
            cols      = table_cols[tbl_idx]

            col_desc_parts = []
            for c in cols:
                tag = ""
                if c["is_pk"]:
                    tag = " (PK)"
                elif c["is_fk"] and c["fk_ref"]:
                    tag = f" (FK -> {c['fk_ref'][0]}.{c['fk_ref'][1]})"
                col_desc_parts.append(
                    f"  {c['name']} [{c['type']}]{tag} -> {c['human_name']}"
                )

            description = (
                f"Table {tbl_orig}: {tbl_human}\n"
                + "\n".join(col_desc_parts)
            )

            records.append({
                "db_type"         : db_type,
                "db_id"           : db_id,
                "table_name"      : tbl_orig,
                "table_human_name": tbl_human,
                "table_idx"       : tbl_idx,
                "description"     : description,
                "columns"         : cols,
                "col_names"       : [c["name"] for c in cols],
                "embedding"       : None,
            })

    return records


def parse_sqlite_fallback(db_dir, db_type="spider"):
    records = []
    for db_folder in sorted(os.listdir(db_dir)):
        db_path = os.path.join(db_dir, db_folder)
        if not os.path.isdir(db_path):
            continue
        for fname in os.listdir(db_path):
            if not (fname.endswith(".sqlite") or fname.endswith(".db")):
                continue
            try:
                con = sqlite3.connect(os.path.join(db_path, fname))
                cur = con.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [r[0] for r in cur.fetchall()]
                for tbl_idx, tbl in enumerate(tables):
                    cur.execute(f'PRAGMA table_info("{tbl}")')
                    cols     = []
                    col_desc = []
                    for row in cur.fetchall():
                        cname = row[1]; ctype = row[2]; is_pk = bool(row[5])
                        tag   = " (PK)" if is_pk else ""
                        col_desc.append(f"  {cname} [{ctype}]{tag} -> {cname}")
                        cols.append({
                            "name": cname, "human_name": cname,
                            "type": ctype, "is_pk": is_pk,
                            "is_fk": False, "fk_ref": None,
                        })
                    description = f"Table {tbl}: {tbl}\n" + "\n".join(col_desc)
                    records.append({
                        "db_type"         : db_type,
                        "db_id"           : db_folder,
                        "table_name"      : tbl,
                        "table_human_name": tbl,
                        "table_idx"       : tbl_idx,
                        "description"     : description,
                        "columns"         : cols,
                        "col_names"       : [c["name"] for c in cols],
                        "embedding"       : None,
                    })
                con.close()
            except Exception:
                pass
    return records


# ─── Initialization ───────────────────────────────────────────────────────────

def initialize(tables_json_path=None, db_dir=None):
    global schema_records, global_index, db_indexes, all_db_ids, embed_model, fk_graph, _initialized

    if _initialized:
        return

    tables_json  = tables_json_path or SPIDER_TABLES_JSON
    db_directory = db_dir           or SPIDER_DB_DIR

    print("Parsing Spider schemas...")
    if os.path.exists(tables_json):
        schema_records = parse_tables_json(tables_json)
        print(f"  {len(schema_records)} tables from tables.json")
    elif os.path.exists(db_directory):
        schema_records = parse_sqlite_fallback(db_directory)
        print(f"  {len(schema_records)} tables from .sqlite files")
    else:
        raise RuntimeError(
            f"Spider data not found.\n"
            f"  Expected tables.json at : {tables_json}\n"
            f"  Or database dir at      : {db_directory}\n"
            f"  Download Spider from https://yale-lily.github.io/spider "
            f"and extract to ./spider/"
        )

    all_db_ids = sorted(set(r["db_id"] for r in schema_records))
    print(f"  {len(all_db_ids)} unique databases")

    print("Loading embedding model...")
    embed_model  = SentenceTransformer("all-MiniLM-L6-v2")

    print("Encoding schema descriptions...")
    descriptions = [r["description"] for r in schema_records]
    embeddings   = embed_model.encode(
        descriptions,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    for i, r in enumerate(schema_records):
        r["embedding"] = embeddings[i]

    dim          = embeddings.shape[1]
    global_index = faiss.IndexFlatIP(dim)
    global_index.add(embeddings.astype("float32"))

    for db_id in all_db_ids:
        recs = [r for r in schema_records if r["db_id"] == db_id]
        embs = np.array([r["embedding"] for r in recs], dtype="float32")
        idx  = faiss.IndexFlatIP(dim)
        idx.add(embs)
        db_indexes[db_id] = {"faiss": idx, "records": recs}

    # Build bidirectional FK graph for each db (used for FK expansion in retrieval)
    for db_id in all_db_ids:
        graph = {}
        for rec in db_indexes[db_id]["records"]:
            tbl = rec["table_name"]
            if tbl not in graph:
                graph[tbl] = set()
            for col in rec["columns"]:
                if col.get("is_fk") and col.get("fk_ref"):
                    other = col["fk_ref"][0]
                    graph[tbl].add(other)
                    graph.setdefault(other, set()).add(tbl)
        fk_graph[db_id] = graph

    print(f"  FAISS global index: {global_index.ntotal} vectors")
    print(f"  Per-db indexes    : {len(db_indexes)}")
    _initialized = True


# ─── Retrieval ────────────────────────────────────────────────────────────────

def _fk_expand(results, db_id, max_extra=3):
    """
    Add up to max_extra tables that are FK-connected to the retrieved set.
    Each candidate is scored by how many retrieved tables it bridges via FK.
    This ensures junction tables (e.g. writes, publication_keyword) are included
    even when their sparse descriptions score low in semantic search.
    """
    if db_id not in fk_graph or db_id not in db_indexes:
        return results

    graph           = fk_graph[db_id]
    retrieved_names = {rec["table_name"] for rec, _ in results}
    rec_by_name     = {rec["table_name"]: rec for rec in db_indexes[db_id]["records"]}

    # Score every non-retrieved table by FK overlap with retrieved set
    scores = {}
    for tbl, neighbors in graph.items():
        if tbl in retrieved_names:
            continue
        overlap = len(neighbors & retrieved_names)
        if overlap > 0:
            scores[tbl] = overlap

    # Add the top candidates (ties broken by table_idx for determinism)
    extras = sorted(scores, key=lambda t: (-scores[t], rec_by_name[t]["table_idx"]))
    for tbl in extras[:max_extra]:
        results.append((rec_by_name[tbl], 0.0))

    return results


def retrieve_tables(question, db_id=None, top_k=5):
    if not _initialized:
        initialize()

    q_emb = embed_model.encode(
        [question], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")

    if db_id and db_id not in ("(all databases)", "") and db_id in db_indexes:
        idx  = db_indexes[db_id]["faiss"]
        recs = db_indexes[db_id]["records"]
        k    = min(top_k, len(recs))
        D, I = idx.search(q_emb, k)
        results = [(recs[i], float(D[0][j])) for j, i in enumerate(I[0])]
        return _fk_expand(results, db_id)
    else:
        k    = min(top_k, global_index.ntotal)
        D, I = global_index.search(q_emb, k)
        return [(schema_records[i], float(D[0][j])) for j, i in enumerate(I[0])]


# ─── Schema text builder ──────────────────────────────────────────────────────

def build_schema_text(results):
    """Build [SCHEMA] block in Spider training format, sorted by table_idx."""
    if not results:
        return ""

    # Preserve original tables.json order regardless of retrieval score order
    results  = sorted(results, key=lambda x: x[0].get("table_idx", 0))
    lines    = ["[SCHEMA]"]
    fk_lines = []

    for rec, _ in results:
        tbl_orig  = rec["table_name"]
        tbl_human = rec.get("table_human_name", tbl_orig)

        lines.append(f"table {tbl_orig}: {tbl_human}")

        for c in rec["columns"]:
            cname      = c["name"]
            human_name = c.get("human_name", cname)
            ctype      = c.get("type", "text")
            is_pk      = c.get("is_pk", False)
            is_fk      = c.get("is_fk", False)
            fk_ref     = c.get("fk_ref")

            qualifier  = ("descriptor"
                          if ctype.lower() in ("text", "varchar", "char")
                          else "measure")

            if is_pk:
                col_line = f"  {cname} ({ctype}, {qualifier}) (PK) -> {human_name}"
            elif is_fk and fk_ref:
                col_line = (f"  {cname} ({ctype}, {qualifier})"
                            f" (FK -> {fk_ref[0]}.{fk_ref[1]})"
                            f" -> {human_name}")
                fk_lines.append(f"{tbl_orig}.{cname} = {fk_ref[0]}.{fk_ref[1]}")
            else:
                col_line = f"  {cname} ({ctype}, {qualifier}) -> {human_name}"

            lines.append(col_line)

        lines.append("")

    if fk_lines:
        seen = set()
        lines.append("[JOINS]")
        for fk in fk_lines:
            if fk not in seen:
                lines.append(fk)
                seen.add(fk)

    return "\n".join(lines).strip()
