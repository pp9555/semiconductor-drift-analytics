FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code, data, AND the real trained XGBoost model binaries
COPY --chown=user app_backend.py .
COPY --chown=user xgb_teacher_model.json .
COPY --chown=user xgb_student_model.json .
COPY --chown=user scored_production_drift_stream.csv .
COPY --chown=user clean_labeled_wafers.csv .
COPY --chown=user golden_baseline_metadata.json .

EXPOSE 8080

# Force unbuffered output so Python prints errors immediately
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app_backend:app", "--host", "0.0.0.0", "--port", "8080"]