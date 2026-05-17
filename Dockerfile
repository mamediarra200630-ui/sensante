FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir huggingface_hub hf_xet

COPY . .

RUN mkdir -p models && python -c "from huggingface_hub import hf_hub_download; [hf_hub_download(repo_id='Didi66/sensante', filename='models/'+f, repo_type='space', local_dir='.') for f in ['model.pkl','encoder_sexe.pkl','encoder_region.pkl','feature_cols.pkl']]"

EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]