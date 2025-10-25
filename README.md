# Food Security

**What**: Ethiopia Food & Market Monitor (FEWS NET prices + simple alerts).

**Why**: Track staple food price dynamics by market to inform ag & livestock programming.

**Quickstart**
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py

**Config**: see .env.example and app/.streamlit/config.toml.

**Data sources**: FEWS NET FDW API (public).

**Testing**: make test

**Deploy**: Streamlit Community Cloud.

**Limitations**: API coverage varies by commodity/market; occasional schema drift.