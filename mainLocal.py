import pandas as pd
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Cargar el DataFrame que se ha dividido en dos partes para poderlo subir a GitHub
parte1 = pd.read_csv('PI_RuthCastañeda/data_preparada_parte1.csv')
parte2 = pd.read_csv('PI_RuthCastañeda/data_preparada_parte2.csv')

# Concatenar el dataset en un DataFrame
data =pd.concat([parte1,parte2], ignore_index = True)

# Preprocesamiento de datos
data['genre'] = data['genre'].apply(lambda x: ' '.join(set(str(x).split(','))))

# Crear una matriz TF-IDF para el texto del título de las películas
stopwords_custom = ["the", "and", "in", "of"]
tfidf = TfidfVectorizer(stop_words=stopwords_custom)
tfidf_matrix = tfidf.fit_transform(data['title'])

# Calcular la similitud del coseno entre los títulos de las películas
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

# Función: Cantidad de filmaciones por mes
def cantidad_filmaciones_mes(mes: str):
    mes = mes.lower()
    meses_map = {
        'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
        'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
        'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
    }
    
    if mes not in meses_map:
        return f"Mes inválido: {mes}"
    
    mes_numero = meses_map[mes]
    peliculas_mes = data[data['release_date'].str.contains(f'-{mes_numero:02d}-')]
    cantidad = len(peliculas_mes)
    
    return {'mes': mes.capitalize(), 'cantidad': cantidad}

# Función: Cantidad de filmaciones por día
def cantidad_filmaciones_dia(dia: str):
    contador = 0
    dias_semana = {"lunes": 0, "martes": 1, "miércoles": 2, "jueves": 3,
                   "viernes": 4, "sábado": 5, "domingo": 6}

    dia = dia.lower()
    
    if dia not in dias_semana:
        return f"Dia inválido: {dia}"
    
    for fecha_estreno in data["release_date"]:
        fecha_estreno_obj = datetime.datetime.strptime(fecha_estreno, "%Y-%m-%d")
        if fecha_estreno_obj.weekday() == dias_semana[dia]:
            contador += 1
    
    return {'dia': dia.capitalize(), 'cantidad': contador}

# Función: Obtener score de una película por título
def score_titulo(titulo_de_la_filmacion: str):
    titulo_de_la_filmacion = titulo_de_la_filmacion.lower()
    pelicula = data[data['title'].str.lower() == titulo_de_la_filmacion]
    
    if pelicula.empty:
        return "Película no encontrada"
    
    titulo = pelicula['title'].iloc[0]
    año_estreno = str(pelicula['release_year'].iloc[0])
    score = str(pelicula['popularity'].iloc[0])
    
    return {'titulo': titulo, 'anio': año_estreno, 'popularidad': score}

# Función: Obtener votos de una película por título
def votos_titulo(titulo_de_la_pelicula: str):
    
    # Convertir el título a minúsculas para la búsqueda
    titulo_de_la_pelicula = titulo_de_la_pelicula.lower()
    pelicula = data[data['title'].str.lower() == titulo_de_la_pelicula]
    
    if pelicula.empty:
        return "La película no existe en el dataset."
    
    titulo = pelicula['title'].iloc[0]
    votos = pelicula['vote_count'].iloc[0]
    promedio_votos = pelicula['vote_average'].iloc[0]
    año_estreno = str(pelicula['release_year'].iloc[0])

    if votos < 2000:
        return f"La película {titulo} no cumple con la condición de tener más de 2000 votos. La misma cuenta con {int(votos)} votos"
    else:
        return {'titulo': titulo, 'anio': año_estreno, 'voto_total': votos, 'voto_promedio': promedio_votos}

# Función: Obtener información de un actor
def get_actor(nombre_actor: str):
    
    # Convertir el nombre del actor a minúsculas para la búsqueda
    nombre_actor = nombre_actor.lower()
    peliculas_actor = data[data['actor'].apply(lambda x: nombre_actor in str(x).lower())]
    
    if peliculas_actor.empty:
        return {"message": f"No se encontraron películas para el actor: {nombre_actor}"}
    
    cantidad_peliculas = len(peliculas_actor)
    promedio_retorno = peliculas_actor['return'].mean()
    retorno = sum(peliculas_actor['return'])
    return {'actor': nombre_actor, 'cantidad_filmaciones': cantidad_peliculas, 'retorno_total': retorno, 'retorno_promedio': promedio_retorno}

# Función: Obtener información de un director
def get_director(nombre_director: str):
    peliculas_director = data[data['director'] == nombre_director]
    
    # Convertir el nombre del director a minúsculas para la búsqueda
    nombre_director = nombre_director.lower()
    peliculas_director = data[data['director'].str.lower() == nombre_director]
    
    retorno_total = peliculas_director['return'].sum()
    peliculas_info = []
    
    for _, pelicula in peliculas_director.iterrows():
        titulo = pelicula['title']
        año_lanzamiento = pelicula['release_year']
        retorno_individual = pelicula['return']
        costo = pelicula['budget']
        ganancia = pelicula['revenue']
        
        pelicula_info = {
            'titulo': titulo,
            'año_lanzamiento': año_lanzamiento,
            'retorno_pelicula': retorno_individual,
            'budget_pelicula': costo,
            'revenue_pelicula': ganancia
        }
        
        peliculas_info.append(pelicula_info)
    
    respuesta = {
        'director': nombre_director,
        'retorno_total': retorno_total,
        'peliculas': peliculas_info
    }
    
    return respuesta

# Función: Recomendación de películas
def recomendacion(titulo):
    # Convertir el título a minúsculas para la búsqueda
    titulo = titulo.lower()
    if titulo not in data['title'].str.lower().values:
        return f"No se encontró ninguna película con el título '{titulo}'."

    indices = pd.Series(data.index, index=data['title'].str.lower()).drop_duplicates()
    idx = indices[titulo]
# CODIGO PRUEBA
       # Verificar si idx es una serie 
    if isinstance(idx, pd.Series):
        #Si es una serie, obtener el primer valor, que corresponde al índice de la primera película con ese título
        idx = idx.iloc[0]


    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [x[0] for x in sim_scores]
    respuesta_recomendacion = data['title'].iloc[movie_indices].tolist()
 
# CODIGO PRUEBA
 # Eliminar el título de la película original de la lista de recomendaciones
    movie_indices = [i for i in movie_indices if data['title'].iloc[i].lower() != titulo]
    respuesta_recomendacion = data['title'].iloc[movie_indices].tolist()

    return {'lista recomendada': respuesta_recomendacion}

# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplos de llamadas a las funciones
    print(cantidad_filmaciones_mes("enero"))
    print(cantidad_filmaciones_dia("lunes"))
    print(score_titulo("toy Story"))
    print(votos_titulo("Toy Story"))
    print(get_actor("tom Hanks"))
    print(get_director("Steven Spielberg"))
    print(recomendacion("titanic"))