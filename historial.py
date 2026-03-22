import pandas as pd

archivo = "historial.csv"


def guardar(texto, genero):

    try:
        df = pd.read_csv(archivo)
    except:
        df = pd.DataFrame(columns=["texto", "genero"])

    df.loc[len(df)] = [texto, genero]

    df.to_csv(archivo, index=False)