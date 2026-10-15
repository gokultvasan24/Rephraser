# 🤗 Power Fusion Academic Paraphraser

Streamlit port of a hybrid **BART + T5** neural paraphrasing engine.

## Live App
Deploy at: https://share.streamlit.io

## Features
- Dual-model candidate generation (BART + T5)
- Power slider (1–5) controls rewrite strength
- Sentence-level hybrid rewrite with best-candidate selection
- Academic-friendly output

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
