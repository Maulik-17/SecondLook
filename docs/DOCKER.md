# Running SecondLook in a container

Status: the image has NOT been built, so the Dockerfile is untested. Docker is installed on the development machine, but a build needs to pull the `python:3.14-slim` base image from a registry and install about 20 pinned packages, so it was not attempted. `scripts/run_all.sh` has also not been run end to end by this change; it chains the four existing scripts from the README.

## Without a container

    PYTHON=python scripts/run_all.sh          # AUDIT_ARGS=--fast skips the audit bootstrap

## With Docker

    docker build -t secondlook .
    docker run --rm -p 8501:8501 secondlook   # app at http://localhost:8501

On first start the container builds the data and model (`scripts/run_all.sh` with the audit in `--fast` mode) if `artefacts/metrics.json` is missing, then serves Streamlit. `data/` and `artefacts/` are excluded from the image, so this happens on every fresh container. To keep results, mount a volume:

    docker run --rm -p 8501:8501 -v "$PWD/artefacts:/app/artefacts" -v "$PWD/data:/app/data" secondlook

To serve the API instead (docs at http://localhost:8000/docs):

    docker run --rm -p 8000:8000 secondlook bash -c "[ -f artefacts/metrics.json ] || AUDIT_ARGS=--fast scripts/run_all.sh; exec uvicorn secondlook.api.main:app --host 0.0.0.0 --port 8000"

`.dockerignore` excludes `data/`, `artefacts/`, `.git`, `.claude`, `__pycache__`, `kaggle.json` and common key files, so no credentials are copied into the image.
