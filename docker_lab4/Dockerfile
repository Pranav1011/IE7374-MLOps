# Stage 1: Build and Train the Model
FROM python:3.10-slim AS model_training

WORKDIR /app

COPY src/model_training.py /app/
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt
RUN python model_training.py


# Stage 2: Serve Predictions
FROM python:3.10-slim AS serving

WORKDIR /app

COPY --from=model_training /app/wine_gb_model.joblib /app/
COPY src/main.py /app/
COPY src/templates /app/templates
COPY src/statics /app/statics
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 4000

CMD ["python", "main.py"]
