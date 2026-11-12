FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Install dependencies first
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ultimate Fix: Copy EVERY file from GitHub into the container at once
COPY --chown=user . .

EXPOSE 8080

CMD ["uvicorn", "app_backend:app", "--host", "0.0.0.0", "--port", "8080"]