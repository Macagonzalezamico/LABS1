import pandas as pd
import numpy as np
from fastapi import FastAPI

import pandas as pd

"""
# Cargar los archivos CSV
df1 = pd.read_csv('Dataset\movies_dataset.csv')
df2 = pd.read_csv('Dataset\credits.csv')

# Juntar los DataFrames por columna
df = pd.concat([df1, df2], axis=1)

# Guardar el DataFrame combinado en un nuevo archivo CSV
df.to_csv('Concatenated.csv', index=False)

revenue_null_count = df['revenue'].isnull().sum()
budget_null_count = df['budget'].isnull().sum()

print("Nulos en la columna 'revenue':", revenue_null_count)
print("Nulos en la columna 'budget':", budget_null_count)

# Transformaciones de los datos.
#Valores nulos rellenados por 0.
df['revenue'] = df['revenue'].fillna(0) 
df['budget'] = df['budget'].fillna(0) 

revenue_null_count = df['revenue'].isnull().sum()
budget_null_count = df['budget'].isnull().sum()

#Los valores nulos del campo release date.
release_date_null_count = df['release_date'].isnull().sum()

print("Nulos en la columna 'release_date':",release_date_null_count)

#Elimino Valores Nulos.
df.dropna(subset=['release_date'], inplace=True)

release_date_null_count = df['release_date'].isnull().sum()
print("Nulos en la columna 'release_date':",release_date_null_count)

#convierte la columna "release_date" en objetos de fecha y hora
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

#Formato de fecha y creo columna release_year.
df['release_year'] = df['release_date'].dt.year

#Pasando a formato fecha
df['release_year'] = df['release_year'].fillna(0).astype(int)

#Reemplazar los valores no numéricos por NaN
df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
df['budget'] = pd.to_numeric(df['budget'], errors='coerce')

#Eliminar las filas con valores no numéricos
df = df.dropna(subset=['revenue', 'budget'])

#Creo columna return con los campos revenue y budget y si no tiene datos P/ calcular toma el valor de 0.
df['revenue'] = df['revenue'].astype(float)
df['budget'] = df['budget'].astype(float)

df['return'] = np.where(df['budget'] != 0, df['revenue'] / df['budget'], 0)

#Eliminar las columnas que no serán utilizadas, video,imdb_id,adult,original_title,vote_count,poster_path y homepage.
df.drop(['video', 'imdb_id', 'adult', 'original_title', 'poster_path', 'homepage'], axis=1, inplace=True)

import ast

# Desanidar la columna 'belongs_to_collection'
df['belongs_to_collection'] = df['belongs_to_collection'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else {})

# Obtener los valores individuales del diccionario
df['belongs_to_collection_id'] = df['belongs_to_collection'].apply(lambda x: x.get('id') if isinstance(x, dict) else None)
df['belongs_to_collection_name'] = df['belongs_to_collection'].apply(lambda x: x.get('name') if isinstance(x, dict) else None)
df['belongs_to_collection_poster_path'] = df['belongs_to_collection'].apply(lambda x: x.get('poster_path') if isinstance(x, dict) else None)
df['belongs_to_collection_backdrop_path'] = df['belongs_to_collection'].apply(lambda x: x.get('backdrop_path') if isinstance(x, dict) else None)

# Eliminar la columna anidada 'genres'
df = df.drop('belongs_to_collection', axis=1)

import ast

# Convertir las cadenas de texto en listas de diccionarios
df['genres'] = df['genres'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])

# Crear una lista para almacenar los nombres de los géneros
genre_names = []

# Iterar sobre cada fila y obtener los nombres de los géneros
for index, row in df.iterrows():
    genres_list = row['genres']
    genre_names_list = [genre['name'] for genre in genres_list]
    genre_names.append(genre_names_list)

# Crear una nueva columna con los nombres de los géneros
df['genre_names'] = genre_names

# Eliminar la columna anidada 'genres'
df = df.drop('genres', axis=1)

import ast

# Convertir las cadenas de texto en listas de diccionarios
df['production_companies'] = df['production_companies'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])

# Crear una lista para almacenar los nombres de las compañías de producción
company_names = []

# Iterar sobre cada fila y obtener los nombres de las compañías de producción
for index, row in df.iterrows():
    companies_list = row['production_companies']
    company_names_list = [company['name'] for company in companies_list]
    company_names.append(company_names_list)

# Crear una nueva columna con los nombres de las compañías de producción
df['production_companies'] = pd.Series(company_names)

import pandas as pd

# Desanidar la columna 'production_countries'
df['production_countries'] = df['production_countries'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])

Countries_name = []

# Iterar sobre cada fila y obtener los nombres de las compañías de producción
for index, row in df.iterrows():
    Countries_list = row['production_countries']
    Countries_name_list = [company['name'] for company in Countries_list]
    Countries_name.append(Countries_name_list )

# Crear una nueva columna con los nombres de las compañías de producción
df['production_countries'] = pd.Series(Countries_name)

import pandas as pd

# Desanidar la columna 'cast'
df['cast'] = df['cast'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])

# Iterar sobre cada fila y obtener los nombres del elenco
cast_names = []

for index, row in df.iterrows():
    cast_list = row['cast']
    cast_names_list = [member['name'] for member in cast_list]
    cast_names.append(cast_names_list)

# Crear una nueva columna con los nombres del elenco
df['cast'] = pd.Series(cast_names)

import pandas as pd

# Desanidar la columna 'cast'
df['crew'] = df['crew'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])

# Iterar sobre cada fila y obtener los nombres del elenco
crew_names = []

for index, row in df.iterrows():
    crew_list = row['crew']
    crew_names_list = [member['name'] for member in crew_list]
    crew_names.append(crew_names_list)

# Crear una nueva columna con los nombres del elenco
df['crew'] = pd.Series(crew_names)

# Convertir la columna 'cast' en una cadena separada por un delimitador específico 
df['cast'] = df['cast'].apply(lambda x: ', '.join(x) if isinstance(x, list) else '')
df['crew'] = df['crew'].apply(lambda x: ', '.join(x) if isinstance(x, list) else '')
"""

Proyecto1 = 'Proyecto1.csv'
df = pd.read_csv('Proyecto1.csv')

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

#ESTADISTICAS DESCRIPTIVAS:

import pandas as pd

# Calcular medidas estadísticas descriptivas
descripcion = df.describe()

# Imprimir el resumen estadístico
df.describe()

#Información del dataframa
df.info()

df.head(2)

#VALORES FALTANTES:
# Verificar si hay valores faltantes en el DataFrame
df.isnull()

# Contar la cantidad de valores faltantes por columna
df.isnull().sum()

#VISUALIZACION DE DATOS:

#GRAFICO DE BARRAS: CANTIDAD DE FILMACIONES POR MES:
import matplotlib.pyplot as plt
import pandas as pd

# Supongamos que tienes un DataFrame llamado 'df' que contiene las columnas 'title' y 'release_date'

# Convertir la columna 'release_date' al formato de fecha
df['release_date'] = pd.to_datetime(df['release_date'], format='%Y-%m-%d')

# Crear una nueva columna 'month' que contenga solo el mes de la fecha
df['month'] = df['release_date'].dt.month

# Contar la cantidad de filmaciones por mes
filmaciones_por_mes = df['month'].value_counts().sort_index()

# Obtener los nombres de los meses en lugar de los números de mes
nombres_meses = pd.to_datetime(filmaciones_por_mes.index, format='%m').strftime('%B')

# Crear el gráfico de barras
plt.bar(nombres_meses, filmaciones_por_mes, color='red')

# Personalizar el gráfico
plt.xlabel('Meses')
plt.ylabel('Cantidad de Filmaciones')
plt.title('Cantidad de Filmaciones de Películas por Mes')

# Rotar los nombres de los meses en el eje x si es necesario
plt.xticks(rotation=45)

# Mostrar el gráfico
plt.show()

#GRAFICO ANALISIS TEMPORAL CANTIDAD DE PELICULAS VISTAS POR MES:
import random
import matplotlib.pyplot as plt

x = ['Enero', 'Feb', 'Mar', 'Abril', 'May', 'Jun', 'Jul', 'Agost', 'Sept', ' Oct', ' Nov', ' Dic']
y = [random.randint(0, 20) for _ in range(12)]

plt.xticks(rotation=45)
plt.plot(x, y, color='red')
plt.xlabel('Fechas')
plt.ylabel('Cantidad de películas vistas')
plt.title('Análisis Temporal de la cantidad de películas vistas por mes')
plt.show()

#GRAFICO DE DISPERSIÓN (Scatter plot):
import matplotlib.pyplot as plt

# Crear un gráfico de dispersión para dos columnas
plt.scatter(df['budget'], df['popularity'], color= 'Blue')
plt.xlabel('budget')
plt.ylabel('popularuty')
plt.title('Gráfico de Dispersión')
plt.show()


#NUBE DE PALABRAS:
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Obtener una lista de todos los títulos de las películas
titulos = df['title'].tolist()  # Reemplaza 'df' con el nombre de tu DataFrame y 'titulo' con el nombre de la columna de títulos

# Unir todos los títulos en un solo string
text = ' '.join(titulos)

# Crear el objeto de WordCloud
wordcloud = WordCloud(width=800, height=400, max_words=100, background_color='white').generate(text)

# Mostrar la nube de palabras
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

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
