Proyecto1 = 'Proyecto1.csv'
df = pd.read_csv('Proyecto1.csv')

# Convertir la columna 'cast' en una cadena separada por un delimitador específico 
df['cast'] = df['cast'].apply(lambda x: ', '.join(x) if isinstance(x, list) else '')
df['crew'] = df['crew'].apply(lambda x: ', '.join(x) if isinstance(x, list) else '')


# Utilizar un formato de archivo diferente que admita la representación de estructuras de datos complejas, como JSON.
df['cast'] = df['cast'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])
df['crew'] = df['crew'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])

from fastapi import FastAPI

# Crear la aplicación FastAPI
app = FastAPI()

#http://127.0.0.1:8000

@app.get('/cantidad_filmaciones_mes')
def cantidad_filmaciones_mes(mes):
        meses = {
            'enero': '01',
            'febrero': '02',
            'marzo': '03',
            'abril': '04',
            'mayo': '05',
            'junio': '06',
            'julio': '07',
            'agosto': '08',
            'septiembre': '09',
            'octubre': '10',
            'noviembre': '11',
            'diciembre': '12'}
    
        mes_numero = meses.get(mes.lower())
    
        if mes_numero is None:
            return "El mes ingresado no es válido."
    
        cantidad = 0
        for fecha in df['release_date']:
            fecha_str = str(fecha)  # Convertir el objeto 'Timestamp' a cadena de texto
            if fecha_str[5:7] == mes_numero:
                cantidad += 1
    

        return f"{cantidad} cantidad de películas fueron estrenadas en el mes de {mes.capitalize()}."

cantidad_filmaciones_mes('')

from datetime import datetime

@app.get('/cantidad_filmaciones_dia')
def cantidad_filmaciones_dia(dia):
    def obtener_nombre_dia(fecha):
        nombre_dia = datetime.strptime(fecha, '%Y-%m-%d').strftime('%A')
        return nombre_dia

    dias = {
        'lunes': 'Monday',
        'martes': 'Tuesday',
        'miercoles': 'Wednesday',
        'jueves': 'Thursday',
        'viernes': 'Friday',
        'sábado': 'Saturday',
        'domingo': 'Sunday'
    }
    
    dia_ingles = dias.get(dia.lower())
    
    if dia_ingles is None:
        return "El día ingresado no es válido."
    
    cantidad = 0
    for fecha in df['release_date']:
        fecha_str = str(fecha)  # Convertir el objeto 'Timestamp' a cadena de texto
        nombre_dia = obtener_nombre_dia(fecha_str.split()[0])  # Obtener solo la fecha sin la hora
        if nombre_dia == dia_ingles:
            cantidad += 1
    
    return f"{cantidad} cantidad de películas fueron estrenadas en los días {dia}."

cantidad_filmaciones_dia('')

@app.get('/score_titulo')
def score_titulo(titulo_de_la_filmacion: str):
    for index, row in df.iterrows():
        if row['title'].lower() == titulo_de_la_filmacion.lower():
            titulo = row['title']
            fecha_estreno = pd.to_datetime(row['release_date'])
            año_estreno = fecha_estreno.year
            score = row['popularity']
            return f"La película {titulo} fue estrenada en el año {año_estreno} con un score/popularidad de {score}."
    
    return "No se encontró información para la película especificada."

score_titulo('')

@app.get('/votos_titulo')
def votos_titulo(titulo_de_la_filmacion):
    for index, row in df.iterrows():
        if row['title'].lower() == titulo_de_la_filmacion.lower():
            titulo = row['title']
            votos = row['vote_count']
            promedio = row['vote_average']
            año_estreno = row['release_year']
            
            if votos >= 2000:
                return f"La película {titulo} fue estrenada en el año {año_estreno}. La misma cuenta con un total de {votos} valoraciones, con un promedio de {promedio}."
            else:
                return f"La película {titulo} fue estrenada en el año {año_estreno}, pero no cumple con la condición mínima de 2000 valoraciones."
    
    return "No se encontró información para la película especificada."

votos_titulo('')

@app.get('/get_actor')
def get_actor(nombre_actor):
    actor_films = df[df['cast'].apply(lambda x: nombre_actor.lower() in [actor.lower() for actor in x] if isinstance(x, list) else False)]
    total_films = len(actor_films)
    total_return = actor_films['return'].sum()
    average_return = total_return / total_films if total_films > 0 else 0

    return f"El actor {nombre_actor} ha participado en {total_films} filmaciones. Ha conseguido un retorno de {total_return} con un promedio de {average_return} por filmación."

get_actor('')


@app.get('/get_director')
def get_director(nombre_director):
    director_films = df[df['crew'].apply(lambda x: nombre_director.lower() in [director.lower() for director in x] if isinstance(x, list) else False)]
    total_films = len(director_films)

    if total_films == 0:
        return f"No se encontró al director {nombre_director} en ninguna filmación"

    films_info = []
    for index, row in director_films.iterrows():
        title = row['title']
        release_date = row['release_date']
        retorno = row['return']
        costo = row['budget']
        ganancia = row['revenue']
        films_info.append({"title": title, "release_date": release_date, "return": retorno, "budget": costo, "revenue": ganancia})

    return {
        "director": nombre_director,
        "peliculas": films_info}

get_director('')

#SISTEMA DE RECOMENDACION:

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Ejemplo de uso
df = pd.read_csv('Proyecto1.csv')  # Cargar los datos de las películas desde un archivo CSV

@app.get('/recomendacion')
def recomendacion(titulo):
    # Convertir el título buscado y los títulos en el DataFrame a minúsculas
    titulo = titulo.lower()
    df['title_lower'] = df['title'].str.lower()
    
    # Obtener el índice de la película buscada
    indice_pelicula = df[df['title_lower'] == titulo].index
    if len(indice_pelicula) == 0:
        return "No se encontró información para la película especificada."
    
    # Obtener los vectores TF-IDF de las sinopsis de las películas
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['overview'].fillna(''))
    
    # Calcular la similitud de coseno entre la película buscada y las demás películas
    similarities = cosine_similarity(tfidf_matrix[indice_pelicula], tfidf_matrix).flatten()
    
    # Ordenar las películas según el score de similaridad en orden descendente
    indices_similares = similarities.argsort()[::-1]
    
    # Obtener los títulos de las películas recomendadas
    peliculas_recomendadas = df.iloc[indices_similares[1:6]]['title'].values.tolist()
    
    return peliculas_recomendadas

titulo_busqueda = ''
recomendaciones = recomendacion(titulo_busqueda)
print(f"Películas recomendadas para '{titulo_busqueda}':")
for pelicula in recomendaciones:
    print(pelicula)
