
FROM python:3.11-slim

RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Backend
COPY --chown=user app_backend.py .

# Trained XGBoost models
COPY --chown=user xgb_teacher_model.json .
COPY --chown=user xgb_student_model.json .

# Required datasets
COPY --chown=user scored_production_drift_stream.csv .
COPY --chown=user clean_labeled_wafers.csv .
COPY --chown=user clean_unlabeled_stream.csv .

# Metadata
COPY --chown=user golden_baseline_metadata.json .

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app_backend:app --host 0.0.0.0 --port ${PORT:-8080}"]
