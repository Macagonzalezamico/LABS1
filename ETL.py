import pandas as pd
import numpy as np
from fastapi import FastAPI

import pandas as pd

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


#DESANIDAR COLUMNAS
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
