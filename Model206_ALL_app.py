import streamlit as st
import pandas as pd
import pickle
import re

st.set_page_config(page_title="Model206 ALL", page_icon="🧬")
st.title("🧬 miRNA Upregulation Predictor — Model 206 ALL")
st.caption("LightGBM · Target Encoding · All conservation levels · family_conservation feature")

@st.cache_resource
def load_model():
    with open('Model206_ALL_model.pkl', 'rb') as f:
        return pickle.load(f)

bundle           = load_model()
model            = bundle['model']
mirna_lookup     = bundle['mirna_lookup']
accession_lookup = bundle['accession_lookup']
options          = bundle['options']
metrics          = bundle['metrics']

if 'history' not in st.session_state:
    st.session_state.history = []

# ── Conservation level labels
CONS_LABELS = {
    2:  "2 — Broadly conserved",
    1:  "1 — Mammal conserved",
    0:  "0 — Poorly conserved",
    -1: "-1 — Species-specific",
}

# ── Metrics banner
st.markdown("#### Model Performance")
c1, c2, c3 = st.columns(3)
c1.metric("ROC-AUC", f"{metrics['auc_mean']:.3f} ± {metrics['auc_std']:.3f}")
c2.metric("Accuracy", f"{metrics['acc_mean']:.3f}")
c3.metric("F1",       f"{metrics['f1_mean']:.3f}")
st.divider()

def normalize(name: str) -> str:
    return re.sub(r'-(5p|3p)$', '', name.strip().lower())

def resolve_mirna(user_input: str):
    """Returns (group, family, accession, conservation) or None."""
    user_input = user_input.strip()
    if user_input in accession_lookup:
        e = accession_lookup[user_input]
        return (e['microrna_group_simplified'], e['family_name'],
                user_input, e.get('family_conservation'))
    if user_input in mirna_lookup:
        e = mirna_lookup[user_input]
        return (e['microrna_group_simplified'], e['family_name'],
                e.get('mirbase_accession'), e.get('family_conservation'))
    norm_input = normalize(user_input)
    for key, val in mirna_lookup.items():
        if normalize(key) == norm_input:
            return (val['microrna_group_simplified'], val['family_name'],
                    val.get('mirbase_accession'), val.get('family_conservation'))
    return None

st.subheader("Enter Prediction Inputs")

mirna_input = st.text_input("miRNA name", placeholder="e.g. hsa-miR-21, miR-155-5p")
parasite    = st.selectbox("Parasite", ["L.major", "L.donovani", "L.amazonensis", "L. donovani"])
organism    = st.selectbox("Organism", ["Human", "Mouse"])
cell_type   = st.selectbox("Cell type", ["PBMC", "THP-1", "BMDM (BALB/c females)", "RAW 264.7",
                                          "Blood serum + liver (BALB/c )"])
time        = st.number_input("Time (hours post-infection)", min_value=1, value=24)

resolved = None
if mirna_input:
    resolved = resolve_mirna(mirna_input)
    if resolved:
        group, family, accession, conservation = resolved
        fam_display  = family if (family and family != 'not_found') else 'Not found'
        cons_display = CONS_LABELS.get(int(conservation), "Unknown") if conservation is not None else "N/A"
        st.success(f"**miRNA group:** `{group}`")
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"**Family name**\n\n{fam_display}")
        col2.markdown(f"**Conservation level**\n\n{cons_display}")
        col3.markdown(f"**Accession**\n\n{accession or 'N/A'}")
    else:
        st.warning("miRNA not found in lookup. Prediction will use unknown family.")

if st.button("Predict", type="primary"):
    if not mirna_input.strip():
        st.warning("Please enter a miRNA name.")
    else:
        if resolved:
            group, family, _, conservation = resolved
        else:
            group        = re.sub(r'^[a-z]{3}-', '', mirna_input.strip().lower())
            group        = re.sub(r'-(5p|3p)$', '', group)
            family       = None
            conservation = None

        fam_val          = None if (not family or family == 'not_found') else family
        cons_val         = float(conservation) if conservation is not None else None
        parasite_celltype = f"{parasite.strip()}_{cell_type.strip()}"

        input_df = pd.DataFrame([{
            'parasite':            parasite,
            'organism':            organism,
            'cell type':           cell_type,
            'family_name':         fam_val,
            'parasite_celltype':   parasite_celltype,
            'time':                int(time),
            'is_found':            0 if fam_val is None else 1,
            'family_conservation': cons_val,
        }])

        proba = model.predict_proba(input_df)[0][1]
        pred  = int(proba >= 0.5)
        label = "⬆️ Upregulated" if pred == 1 else "⬇️ Downregulated"
        color = "green" if pred == 1 else "red"

        st.markdown(f"### Prediction: :{color}[{label}]")
        st.metric("Confidence", f"{proba*100:.1f}%")

        fam_display  = fam_val or 'Not found'
        cons_display = CONS_LABELS.get(int(conservation), "N/A") if conservation is not None else "N/A"
        st.session_state.history.append({
            "miRNA":        mirna_input.strip(),
            "Family":       fam_display,
            "Conservation": cons_display,
            "Parasite":     parasite,
            "Organism":     organism,
            "Cell type":    cell_type,
            "Time (h)":     time,
            "Prediction":   label,
            "Confidence":   f"{proba*100:.1f}%",
        })

if st.session_state.history:
    st.subheader("Prediction History")
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
    if st.button("Clear history"):
        st.session_state.history = []
