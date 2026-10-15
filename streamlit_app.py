# ── STREAMLIT POWER FUSION PARAPHRASER ───────────────────────────────────
import re
import streamlit as st
import torch
from transformers import (
    BartForConditionalGeneration, BartTokenizer,
    AutoTokenizer, AutoModelForSeq2SeqLM
)

# ── PAGE CONFIG ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Power Fusion Paraphraser",
    page_icon="🤗",
    layout="wide"
)

# ── CUSTOM CSS (mirrors your notebook gradient header) ───────────────────
st.markdown("""
<style>
.header-box {
    background: linear-gradient(135deg,#1a237e,#1565c0);
    padding:16px 20px; border-radius:10px; margin-bottom:14px;
}
.header-box h3 { color:#fff; margin:0; font-size:20px; }
.header-box p  { color:#bbdefb; margin:6px 0 0; font-size:13px; }
.orig-box {
    background:#fff3e0; border-left:5px solid #fb8c00;
    padding:12px; border-radius:6px; margin-bottom:14px;
    font-size:14px; line-height:1.5;
}
.final-box {
    background:#fff; border:1px solid #dcdcdc;
    border-left:5px solid #6a1b9a; border-radius:6px;
    padding:12px; margin:8px 0;
}
.final-box .label { color:#6a1b9a; font-size:12px; font-weight:600; margin-bottom:6px; }
</style>
""", unsafe_allow_html=True)


# ── MODEL LOADING (cached so it loads once) ──────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bart_tokenizer = BartTokenizer.from_pretrained("eugenesiow/bart-paraphrase")
    bart_model = BartForConditionalGeneration.from_pretrained(
        "eugenesiow/bart-paraphrase"
    ).to(device)

    t5_tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    t5_model = AutoModelForSeq2SeqLM.from_pretrained(
        "Vamsi/T5_Paraphrase_Paws"
    ).to(device)

    return device, bart_tokenizer, bart_model, t5_tokenizer, t5_model


# ── SENTENCE SPLITTER ────────────────────────────────────────────────────
def split_sentences(text):
    text = re.sub(r'\s+', ' ', text).strip()
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if len(s.strip()) > 5]


def unique_clean(lst):
    seen, out = set(), []
    for x in lst:
        x = x.strip()
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


# ── BART GENERATOR ───────────────────────────────────────────────────────
def bart_generate(sentence, strength, device, tok, model):
    batch = tok(
        sentence,
        return_tensors='pt',
        truncation=True,
        max_length=128,
        padding=True
    ).to(device)

    gen = model.generate(
        batch['input_ids'],
        num_beams=2 + strength,
        num_return_sequences=1,
        max_length=128,
        no_repeat_ngram_size=2
    )

    return tok.decode(gen[0], skip_special_tokens=True,
                      clean_up_tokenization_spaces=True)


# ── T5 GENERATOR ─────────────────────────────────────────────────────────
def t5_generate(sentence, strength, device, tok, model):
    prompt = "paraphrase: " + sentence + " </s>"

    enc = tok(prompt, return_tensors="pt",
              truncation=True, max_length=128, padding=True)
    ids = enc["input_ids"].to(device)
    masks = enc["attention_mask"].to(device)

    gen = model.generate(
        input_ids=ids,
        attention_mask=masks,
        max_length=128,
        do_sample=True,
        top_k=40 + strength * 10,
        top_p=0.88 + (strength * 0.015),
        temperature=1.05 + (strength * 0.05),
        num_return_sequences=1
    )

    return tok.decode(gen[0], skip_special_tokens=True,
                      clean_up_tokenization_spaces=True)


# ── BEST CANDIDATE PICKER ────────────────────────────────────────────────
def choose_best(original, candidates):
    candidates = unique_clean(candidates)
    scored = []
    for c in candidates:
        score = abs(len(c) - len(original))
        if c.lower() != original.lower():
            scored.append((score, c))
    if not scored:
        return original
    scored.sort(key=lambda x: x[0])
    return scored[0][1]


# ── HYBRID PARAPHRASE ────────────────────────────────────────────────────
def hybrid_paraphrase(text, strength, device, bt, bm, tt, tm):
    sentences = split_sentences(text)
    final_para = []

    for sent in sentences:
        candidates = []
        for _ in range(strength):
            candidates.append(bart_generate(sent, strength, device, bt, bm))
            candidates.append(t5_generate(sent, strength, device, tt, tm))
        best = choose_best(sent, candidates)
        final_para.append(best)

    return " ".join(final_para)


# ── UI ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
  <h3>🤗 Power Fusion Academic Paraphraser</h3>
  <p>Hidden BART + T5 candidate generation • One final polished rewrite</p>
</div>
""", unsafe_allow_html=True)

# Load models with spinner
with st.spinner("📥 Loading BART + T5 neural models (first run may take 1–2 min)..."):
    device, bart_tok, bart_model, t5_tok, t5_model = load_models()

st.success(f"✅ Models loaded on **{device}**")

text = st.text_area(
    "📝 Paste your text:",
    height=200,
    placeholder="Paste journal abstract, methodology, literature review, or any paragraph here..."
)

col1, col2 = st.columns([3, 1])
with col1:
    strength = st.slider("⚡ Power (rewrite strength):", 1, 5, 3)
with col2:
    st.write("")
    st.write("")
    rephrase = st.button("🔄 Rephrase", type="primary", use_container_width=True)

if rephrase:
    if not text.strip():
        st.warning("⚠️ Please paste some text first.")
    else:
        st.markdown(
            f'<div class="orig-box"><b>📝 Original Text</b><br><br>{text}</div>',
            unsafe_allow_html=True
        )
        with st.spinner("⏳ Building powerful combined paraphrase..."):
            try:
                result = hybrid_paraphrase(
                    text, strength, device,
                    bart_tok, bart_model, t5_tok, t5_model
                )
                st.markdown(
                    '<div class="final-box">'
                    '<div class="label">✨ Final Overall Combined Paraphrase</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
                st.text_area("Result (copy below):", value=result, height=220, key="out")
                st.code(result, language=None)
                st.success("✅ Final combined paraphrase ready.")
            except Exception as e:
                st.error(f"❌ Error: {e}")

st.caption("Built with BART (eugenesiow/bart-paraphrase) + T5 (Vamsi/T5_Paraphrase_Paws)")
