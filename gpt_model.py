from openai import OpenAI
from config import OPENAI_KEY

client = OpenAI(api_key=OPENAI_KEY)


def detectar_genero(texto):

    prompt = f"""
Detecta el genero de pelicula:
terror, accion, comedia, romance, ciencia, fantasia, aventura, drama, animacion

Texto: {texto}

Responde solo el genero
"""

    r = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": prompt}]
    )

    return r.choices[0].message.content.lower().strip()