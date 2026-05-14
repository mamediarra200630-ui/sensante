FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Telecharger les modeles depuis Hugging Face
RUN python -c "
from huggingface_hub import hf_hub_download, HfApi
import os
os.makedirs('models', exist_ok=True)
api = HfApi()
files = ['model.pkl', 'encoder_sexe.pkl', 'encoder_region.pkl', 'feature_cols.pkl']
for f in files:
    api.hf_hub_download = hf_hub_download
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id='Didi66/sensante', filename=f'models/{f}', repo_type='space', local_dir='.')
    print(f'Downloaded: {f}')
"

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]