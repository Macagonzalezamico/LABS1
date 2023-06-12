<p align="center">
  <img src="image/README/1686531329752.png">
</p>

<h1 align="center">PROYECTO INDIVIDUAL Nº1</h1>

<h2 align="center">MACHINE LEARNING OPERATIONS (MLOps)</h2>

## **Descripción del problema (Contexto y rol a desarrollar)**

En el transcurso de este proyecto haremos un recorrido por cada una de las temáticas esenciales. Vamos a crear mi primer modelo de ML que soluciona un problema de negocio: un sistema de recomendación que aún no ha sido puesto en marcha!

Vamos a empezar desde 0, haciendo un trabajo rápido de **`Data Engineer`** y tener un **`MVP`** ( *Minimum Viable Product* ). Vamos a utiliza los archivos credits.csv y movies_dataset.csv .

Links del proyecto:

GITHUB: https://github.com/Macagonzalezamico/LABS1

RENDER:

# Propuesta de Trabajo

Este proyecto tiene como objetivo realizar transformaciones en los datos, desarrollar una API utilizando el framework FastAPI, realizar análisis exploratorio de los datos y construir un sistema de recomendación de películas. A continuación se detallan los pasos y requerimientos necesarios.

## Transformaciones

* Algunos campos están anidados en el dataset y deben ser desanidados para realizar consultas en la API.
* Los valores nulos de los campos 'revenue' y 'budget' deben ser reemplazados por 0.
* Los valores nulos del campo 'release date' deben ser eliminados.
* Las fechas deben tener el formato AAAA-mm-dd y se debe crear la columna 'release_year' para extraer el año de estreno.
* Se debe crear la columna 'return' que representa el retorno de inversión calculado como revenue / budget. Si no hay datos disponibles, el valor debe ser 0.
* Se deben eliminar las columnas que no serán utilizadas: 'video', 'imdb_id', 'adult', 'original_title', 'poster_path' y 'homepage'.

## Desarrollo de la API

Se utilizará el framework FastAPI para desarrollar una API con los siguientes endpoints:

1. `cantidad_filmaciones_mes(Mes)`: Devuelve la cantidad de películas estrenadas en el mes consultado en todo el dataset.
   > Ejemplo de retorno: "X cantidad de películas fueron estrenadas en el mes de X".
   >
2. `cantidad_filmaciones_dia(Dia)`: Devuelve la cantidad de películas estrenadas en el día consultado en todo el dataset.
   > Ejemplo de retorno: "X cantidad de películas fueron estrenadas en los días X".
   >
3. `score_titulo(titulo_de_la_filmación)`: Ingresando el Titulo de una filmacion devuelve el título, año de estreno y score/popularidad de una película.
   > Ejemplo de retorno: "La película X fue estrenada en el año X con un score/popularidad de X".
   >
4. `votos_titulo(titulo_de_la_filmación)`: Ingresando el Titulo de una filmacion devuelve el título, cantidad de votos y valor promedio de votaciones de una película. Se requiere que la película tenga al menos 2000 valoraciones; de lo contrario, se mostrará un mensaje indicando que no cumple con esta condición.
   > Ejemplo de retorno: "La película X fue estrenada en el año X. La misma cuenta con un total de X valoraciones, con un promedio de X".
   >
5. `get_actor(nombre_actor)`: Ingresando el nombre del Actor devuelve el éxito del mismo medido a través del retorno, la cantidad de películas en las que ha participado y el promedio de retorno. No se considerarán los directores.
   > Ejemplo de retorno: "El actor X ha participado de X cantidad de filmaciones. Ha conseguido un retorno de X con un promedio de X por filmación".
   >
6. `get_director(nombre_director)`: Ingresando el nombre del Director devuelve el éxito del mismo medido a través del retorno, así como el nombre, fecha de lanzamiento, retorno individual, costo y ganancia de cada película dirigida por el mismo.

## Deployment

El proyecto es desplegado utilizando servicios como Render, que permiten que la API sea consumida desde la web.

## Análisis Exploratorio de los Datos (EDA)

Se realiza un análisis exploratorio de los datos. El objetivo es investigar las relaciones entre las variables del dataset, identificar outliers o anomalías y descubrir patrones interesantes. También se genero una nube de palabras para conocer las palabras más frecuentes en los títulos de las películas.

## Sistema de Recomendación

Los datos están disponibles a través de la API y se realizado el análisis exploratorio, se comomenzo a construir un sistema de recomendación de películas. El sistema recomendará películas similares basándose en la similitud de puntuación entre una película dada y el resto de películas. Se ordenarán según el score de similaridad y se devolverá una lista con los 5 títulos de películas con mayor puntaje, en orden descendente. Esta funcionalidad se agregará como una función adicional en la API y se llamará `recomendacion(titulo)`.
