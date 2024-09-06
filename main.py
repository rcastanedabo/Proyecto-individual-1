from fastapi import FastAPI
import pandas as pd
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Crear una instancia de la aplicación
app = FastAPI()

# Cargamos el dataframe
data = pd.read_csv('PI_RuthCastañeda/data_preparada_parte1.csv')

# Necesario para funcion recomendacion:
# Preprocesamiento de datos
# Convertir los valores de la columna 'genre' en cadenas de texto
data['genre'] = data['genre'].apply(lambda x: ' '.join(set(str(x).split(','))))
# data['genre'] = data['genre'].apply(lambda x: ' '.join(set(str(x))))

# Crear una matriz TF-IDF para el texto del título de las películas
stopwords_custom = ["the", "and", "in", "of"]  # Agrega aquí tus stopwords personalizados
tfidf = TfidfVectorizer(stop_words=stopwords_custom)
tfidf_matrix = tfidf.fit_transform(data['title'])

# Calcular la similitud del coseno entre los títulos de las películas
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

# Definir la función con el decorador
@app.get("/cantidad_filmaciones_mes/{mes}")
def cantidad_filmaciones_mes(mes: str):
    # Convertir el mes a minúsculas
    mes = mes.lower()
    
    # Mapear los nombres de los meses en español a los números de los meses
    meses_map = {'enero': 1,'febrero': 2,'marzo': 3,'abril': 4,
                'mayo': 5,'junio': 6,'julio': 7,'agosto': 8,'septiembre': 9,
                'octubre': 10,'noviembre': 11,'diciembre': 12}
    
    # Verificar si el mes ingresado es válido
    if mes not in meses_map:
        return f"Mes inválido: {mes}"
    
    # Obtener el número de mes correspondiente
    mes_numero = meses_map[mes]
    
    # Filtrar las filas que corresponden al mes consultado
    peliculas_mes = data[data['release_date'].str.contains(f'-{mes_numero:02d}-')] # La expresión f'-{mes_numero:02d}-' genera una cadena de texto en formato '-MM-', donde 'MM' representa el número del mes formateado con 2 dígitos.

    # Obtener la cantidad de películas en el mes consultado
    cantidad = len(peliculas_mes)
    
    # Devolver el resultado como un string formateado
    #return f"{cantidad} películas fueron estrenadas en el mes de {mes.capitalize()}" # capitalize() convierte el primer carácter de una cadena en mayúscula y el resto de los caracteres en minúscula.
    return {'mes':mes.capitalize(), 'cantidad':cantidad}

# Definir la función
@app.get("/cantidad_filmaciones_dia/{dia}")
def cantidad_filmaciones_dia(dia: str): # Devuelve la cantidad de filmaciones 
    contador = 0

    # Mapear los nombres de los dias en español a los números de los dias
    dias_semana = {"lunes": 0,"martes": 1,"miércoles": 2,"jueves": 3,
                   "viernes": 4,"sábado": 5,"domingo": 6}

    # Convertir el dia a minúsculas
    dia = dia.lower()

    # Verificar si el dia ingresado es válido
    if dia not in dias_semana:
        return f"Dia inválido: {dia}"
    
    # Aquí deberías tener la lógica para obtener los datos del dataset y contar las películas estrenadas en el día consultado
    # Asegúrate de tener la estructura adecuada del dataset y ajusta el código según tus necesidades
    for fecha_estreno in data["release_date"]:
        fecha_estreno_obj = datetime.datetime.strptime(fecha_estreno, "%Y-%m-%d") # strptime se utiliza para convertir una cadena de texto en un objeto de fecha y hora (datetime).

        if fecha_estreno_obj.weekday() == dias_semana[dia]: # La función weekday() se utiliza para obtener el día de la semana correspondiente a un objeto datetime. Retorna un número entero que representa el día de la semana, donde el lunes es el día 0 y el domingo es el día 6.
            contador += 1

    # return f"{contador} películas fueron estrenadas en los días {dia.capitalize()}" # capitalize() convierte el primer carácter de una cadena en mayúscula y el resto de los caracteres en minúscula.
    return {'dia':dia.capitalize(), 'cantidad':contador}

# Definir la función
@app.get("/score_titulo/{titulo_de_la_filmacion}")
def score_titulo(titulo_de_la_filmacion: str):
     # Convertir el título a minúsculas para la búsqueda
    titulo_de_la_filmacion = titulo_de_la_filmacion.lower()
    
    # Buscar la película por título en el dataframe
    pelicula = data[data['title'].str.lower() == titulo_de_la_filmacion]
    
    # Verificar si se encontró la película
    if pelicula.empty: # El método empty devuelve True si el DataFrame está vacío.
        return "Película no encontrada"
    
    # Obtener los valores de título, año de estreno y score
    titulo = pelicula['title'].iloc[0] # iloc[0] sirve para acceder al primer registro
    año_estreno = str(pelicula['release_year'].iloc[0])
    score = str(pelicula['popularity'].iloc[0])
    
    # return f"La pelicula {titulo} fue estrenada en el año {año_estreno} con un score de {score}."
    return {'titulo':titulo, 'anio':año_estreno, 'popularidad':score}

# Definir la función con el decorador
@app.get("/votos_titulo/{titulo_de_la_pelicula}")
def votos_titulo(titulo_de_la_pelicula: str):
    
    # Convertir el título a minúsculas para la búsqueda
    titulo_de_la_pelicula = titulo_de_la_pelicula.lower()
    
    # Buscar la película por título en el dataframe
    pelicula = data[data['title'].str.lower() == titulo_de_la_pelicula]
    
    # Verificar si la película existe en el dataframe
    if pelicula.empty:
        return "La película no existe en el dataset."
    
    # Obtener los valores de título, cantidad de votos y valor promedio de las votaciones
    titulo = pelicula['title'].iloc[0]
    votos = pelicula['vote_count'].iloc[0]
    promedio_votos = pelicula['vote_average'].iloc[0]
    año_estreno = str(pelicula['release_year'].iloc[0])

    # Verificar si la película cumple con la condición de tener más de 2000 votos
    if votos < 2000:
        return f"La película {titulo} no cumple con la condición de tener más de 2000 votos. La misma cuenta con {int(votos)} votos"
    else:
        # return f"La película {titulo} fue estrenada en el año {año_estreno}. La misma cuenta con un total de {int(votos)} valoraciones, con un promedio de {promedio_votos}"
        return {'titulo':titulo, 'anio':año_estreno, 'voto_total':votos, 'voto_promedio':promedio_votos}

# Definir la función con el decorador
@app.get("/get_actor/{nombre_actor}")
def get_actor(nombre_actor: str):
    # Convertir el nombre del actor a minúsculas para la búsqueda
    nombre_actor = nombre_actor.lower()
    # Filtrar las filas que contienen al actor consultado
    peliculas_actor = data[data['actor'].apply(lambda x: nombre_actor in str(x).lower())]
    
    # Verificar si el actor existe en el dataset
    if peliculas_actor.empty:
        return {"message": f"No se encontraron películas para el actor: {nombre_actor}"}
    
    # Obtener la cantidad de películas y el promedio de retorno del actor
    cantidad_peliculas = len(peliculas_actor)
    promedio_retorno = peliculas_actor['return'].mean()
    retorno = sum(peliculas_actor['return'])
    return {'actor':nombre_actor, 'cantidad_filmaciones':cantidad_peliculas, 'retorno_total':retorno, 'retorno_promedio':promedio_retorno}

@app.get('/get_director/{nombre_director}')
def get_director(nombre_director: str):
     # Convertir el nombre del director a minúsculas para la búsqueda
    nombre_director = nombre_director.lower()
    # Filtrar las filas que contienen al director consultado
    peliculas_director = data[data['director'].str.lower() == nombre_director]
    
    # Verificar si el director existe en el dataset
    if peliculas_director.empty:
        return {"message": f"No se encontraron películas para el director: {nombre_director}"}
    
    # Calcular la suma del retorno de inversión total
    retorno_total = peliculas_director['return'].sum()
    
    # Crear una lista para almacenar la información de cada película
    peliculas_info = []
    
    # Recorrer cada película del director
    for _, pelicula in peliculas_director.iterrows(): # iterrows() se utiliza para iterar sobre un DataFrame de Pandas fila por fila, cada iteración devuelve una tupla que contiene el índice de la fila y la serie de datos correspondiente a esa fila.
        titulo = pelicula['title']
        año_lanzamiento = pelicula['release_year']
        retorno_individual = pelicula['return']
        costo = pelicula['budget']
        ganancia = pelicula['revenue']
        
        # Crear un diccionario con la información de la película
        pelicula_info = {
            'titulo': titulo,
            'año_lanzamiento': año_lanzamiento,
            'retorno_pelicula': retorno_individual,
            'budget_pelicula': costo,
            'revenue_pelicula': ganancia
        }
        
        # Agregar el diccionario a la lista de películas
        peliculas_info.append(pelicula_info)
    
    # Crear el diccionario de respuesta con la suma del retorno total y la lista de películas
    respuesta = {
        'director': nombre_director,
        'retorno_total': retorno_total,
        'peliculas': peliculas_info
    }
    
    return respuesta

@app.get('/recomendacion/{titulo}')
def recomendacion(titulo):
    # Convertir el título a minúsculas para la búsqueda
    titulo = titulo.lower()
    # Verificar si el título está en el DataFrame
    if titulo not in data['title'].str.lower().values:
        return f"No se encontró ninguna película con el título '{titulo}'."

    # Encontrar el índice de la película con el título dado
    indices = pd.Series(data.index, index=data['title'].str.lower()).drop_duplicates()
    idx = indices[titulo]
    
           # Verificar si idx es una serie 
    if isinstance(idx, pd.Series):
        #Si es una serie, obtener el primer valor, que corresponde al índice de la primera película con ese título
        idx = idx.iloc[0]

    # Calcular las puntuaciones de similitud de todas las películas con la película dada
    sim_scores = list(enumerate(cosine_similarities[idx]))

    # Ordenar las películas por puntaje de similitud en orden descendente
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Obtener los índices de las películas más similares (excluyendo la película dada)
    sim_scores = sim_scores[1:6]  # Obtener las 5 películas más similares
    
    movie_indices = [x[0] for x in sim_scores]
    respuesta_recomendacion = data['title'].iloc[movie_indices].tolist()
    
     # Eliminar el título de la película original de la lista de recomendaciones
    movie_indices = [i for i in movie_indices if data['title'].iloc[i].lower() != titulo]

    # Devolver los títulos de las películas más similares
    respuesta_recomendacion = data['title'].iloc[movie_indices].tolist()
    
    
    
    return {'lista recomendada': respuesta_recomendacion}


# Ejecutar la aplicación con Uvicorn
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)

