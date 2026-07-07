import os
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import nl2sqlSchemas as rag


# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="NL2SQL — Spider", layout="wide")


# ─── Cached loaders (run once per session) ────────────────────────────────────
@st.cache_resource(show_spinner="Loading BART-Large + LoRA…")
def load_model():
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    BASE_MODEL = "facebook/bart-large"
    LORA_PATH  = "./bart_large_lora_spider"

    if not os.path.exists(LORA_PATH):
        raise ValueError(f"LoRA folder not found at {LORA_PATH}")

    tokenizer  = AutoTokenizer.from_pretrained(BASE_MODEL)
    model      = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    model      = PeftModel.from_pretrained(model, LORA_PATH)
    model      = model.merge_and_unload()
    model      = model.to(device)
    model.eval()
    return tokenizer, model, device


@st.cache_resource(show_spinner="Initialising Spider RAG index…")
def load_rag():
    rag.initialize()
    return rag.all_db_ids


# ─── Inference ────────────────────────────────────────────────────────────────
def generate_sql(prompt, tokenizer, model, device):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=256, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ─── UI ───────────────────────────────────────────────────────────────────────
st.title("NL2SQL")
st.caption("Natural Language → SQL using BART-Large + LoRA fine-tuned on Spider")

tokenizer, model, device = load_model()
all_db_ids               = load_rag()

st.divider()

# ── Inputs ────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 2])

with col_left:
    db_options = ["(all databases)"] + all_db_ids
    db_id      = st.selectbox(
        "Database (db_id)",
        options=db_options,
        index=0,
        help=f"{len(all_db_ids)} Spider databases available",
    )
    top_k = st.slider("Top-K tables to retrieve", min_value=1, max_value=10, value=5)

with col_right:
    question = st.text_area(
        "Natural Language Question",
        placeholder="e.g.  How many papers has each author published?",
        height=120,
    )
    run = st.button("Generate SQL", type="primary", use_container_width=True)

st.divider()

# ── Example questions ─────────────────────────────────────────────────────────
with st.expander("Example questions"):
    examples = [
        ("How many papers has each author published?",         "academic"),
        ("Which concert had the most singers?",                "concert_singer"),
        ("What is the average salary of employees?",           "employee_hire_evaluation"),
        ("List all students enrolled in more than 3 courses",  "student_transcripts_tracking"),
        ("How many cars does each manufacturer produce?",      "car_1"),
        ("Find all flights departing from London",             "flight_2"),
    ]
    for q, db in examples:
        st.markdown(f"- **{db}**: {q}")

# ── Pipeline + Output ─────────────────────────────────────────────────────────
if run:
    question = question.strip()
    if not question:
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Retrieving relevant tables…"):
        results     = rag.retrieve_tables(question, db_id=db_id, top_k=top_k)
        schema_text = rag.build_schema_text(results)

    with st.spinner("Generating SQL…"):
        prompt = f"[QUESTION]\n{question}\n\n{schema_text}\n"
        sql    = generate_sql(prompt, tokenizer, model, device)

    # ── Retrieved tables ──────────────────────────────────────────────────────
    st.subheader("Retrieved Tables")
    for i, (rec, score) in enumerate(results):
        label = "FK-expanded" if score == 0.0 else f"score {score:.4f}"
        with st.expander(
            f"#{i+1}  [{label}]  `{rec['db_id']}` / `{rec['table_name']}`",
            expanded=(i == 0),
        ):
            st.markdown(f"**Table description:** {rec['table_human_name']}")
            rows = []
            for c in rec["columns"]:
                tags = []
                if c.get("is_pk"):
                    tags.append("PK")
                if c.get("is_fk") and c.get("fk_ref"):
                    tags.append(f"FK → {c['fk_ref'][0]}.{c['fk_ref'][1]}")
                rows.append({
                    "Column"     : c["name"],
                    "Type"       : c["type"],
                    "Description": c["human_name"],
                    "Tags"       : ", ".join(tags),
                })
            st.table(rows)

    # ── Schema block ──────────────────────────────────────────────────────────
    st.subheader("[SCHEMA] Block (fed to model)")
    st.code(schema_text, language="text")

    # ── SQL output ────────────────────────────────────────────────────────────
    st.subheader("Generated SQL")
    st.code(sql, language="sql")
