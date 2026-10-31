# Python version follows .python-version / requirements.txt (3.14.x).
FROM python:3.14-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=src PYTHON=python
EXPOSE 8501 8000
# On first start, build data and model if missing (audit in --fast mode), then serve the app.
# For the API instead, override the command:
#   docker run -p 8000:8000 secondlook bash -c "[ -f artefacts/metrics.json ] || AUDIT_ARGS=--fast scripts/run_all.sh; exec uvicorn secondlook.api.main:app --host 0.0.0.0 --port 8000"
CMD ["bash", "-c", "[ -f artefacts/metrics.json ] || AUDIT_ARGS=--fast scripts/run_all.sh; exec streamlit run app/Home.py --server.address 0.0.0.0 --server.port 8501"]
