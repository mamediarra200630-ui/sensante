# notebooks/test_groq.py
# Exercice 2 - Tester differentes temperatures
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

prompt_user = """Patient : Femme, 28 ans, region Dakar
Symptomes : temperature 39.5, toux, fatigue, maux de tete
Diagnostic du modele : paludisme (probabilite 72%)
Explique ce resultat au patient."""

prompt_system = """Tu es un assistant medical senegalais.
Explique le resultat en francais simple.
Maximum 3 phrases. Ne fais jamais de diagnostic toi-meme."""

for temp in [0.0, 0.5, 1.0]:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user}
        ],
        max_tokens=200,
        temperature=temp
    )
    print(f"=== Temperature = {temp} ===")
    print(response.choices[0].message.content)
    print()