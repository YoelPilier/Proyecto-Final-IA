import requests
import random
from config import TMDB_KEY


mapa = {
    "terror": 27,
    "accion": 28,
    "comedia": 35,
    "romance": 10749,
    "ciencia": 878,
    "fantasia": 14,
    "aventura": 12,
    "drama": 18,
    "animacion": 16
    
}


def buscar(genero):

    idg = mapa[genero]

    url = f"https://api.themoviedb.org/3/discover/movie?api_key={TMDB_KEY}&with_genres={idg}"

    r = requests.get(url).json()

    peli = random.choice(r["results"])

    titulo = peli["title"]
    sinopsis = peli["overview"]
    nota = peli["vote_average"]
    img = "https://image.tmdb.org/t/p/w500" + peli["poster_path"]

    return titulo, sinopsis, nota, img