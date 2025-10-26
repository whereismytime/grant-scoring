import streamlit as st
from app.domain.schemas import Applicant
from app.services.scorer import score

st.set_page_config(page_title="Grant Scoring UI", layout="centered")
st.title("Grant Scoring UI")

with st.sidebar:
    st.header("Parameters")
    with st.form("params"):
        distance = st.slider("Distance (km)", 0.0, 80.0, 20.0, 1.0, key="distance")
        residency = st.slider("Residency in Ireland (years)", 0.0, 6.0, 2.0, 0.1, key="residency")
        is_imm = st.checkbox("Immigrant", value=False, key="is_imm")
        parents = st.selectbox("Parents' income status", ["low", "middle", "high"], key="parents")
        use_ml = st.checkbox("Use ML model", value=True, key="use_ml")
        thr = st.slider("Approval threshold", 0.0, 1.0, 0.5, 0.01, key="thr")
        submitted = st.form_submit_button("Evaluate", use_container_width=True)

if submitted:
    params = dict(
        distance=float(st.session_state.distance),
        residency=float(st.session_state.residency),
        is_imm=bool(st.session_state.is_imm),
        parents=st.session_state.parents,
        use_ml=bool(st.session_state.use_ml),
        thr=float(st.session_state.thr),
    )
    with st.spinner("Evaluating..."):
        try:
            a = Applicant(
                distance_km=params["distance"],
                residency_years_ie=params["residency"],
                is_immigrant=params["is_imm"],
                parents_status=params["parents"],
            )
            res = score(a, use_ml=params["use_ml"], threshold=params["thr"])
        except Exception as e:
            st.error(f"Scoring error: {e}")
            st.code({"params": params})
            st.stop()

    st.caption(f"Mode: **{'ML' if params['use_ml'] else 'Rules'}** | Threshold = {params['thr']:.2f}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Decision", res.decision.upper())
    c2.metric("Approved", "Yes" if res.approved else "No")
    c3.metric("Grant amount", f"{res.amount}")

    if res.prob is not None:
        st.progress(min(max(res.prob, 0.0), 1.0))
        if "rule override" in res.reasons:
            st.caption("ML probability is shown, but the decision was overridden by rules.")

    st.caption("Reasons: " + ", ".join(res.reasons))
    with st.expander("Debug info"):
        st.code({"params": params, "result": res.model_dump()})
