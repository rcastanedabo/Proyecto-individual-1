

# <h1 align=center> **PROYECTO INDIVIDUAL Nº1** </h1>
# <h1 align=center> **Ruth Castañeda Bojorques** </h1>

# <h1 align=center>**`Machine Learning Operations (MLOps)`**</h1>

<p align="center">
  <img src="src\MLOpslogo.jpg" alt="MLOps">
</p>

## ```Introducción```

  Desarrollar un sistema avanzado de recomendación de películas que facilite a los usuarios el descubrimiento
  de producciones cinematográficas alineadas con sus gustos e intereses. Para ello, se implementa un modelo de 
  filtrado basado en contenido que analiza las características descriptivas de cada película y mide la similitud 
  entre ellas. Este proceso se optimiza mediante técnicas de procesamiento de lenguaje natural, como la vectorización
  TF-IDF, lo que permite transformar los textos en representaciones numéricas aptas para algoritmos de machine learning, 
  asegurando así recomendaciones precisas y personalizadas.

## :white_check_mark: ```Objetivo General```

- :pushpin: Implementar una API para acceso a datos y recomendaciones.

## :white_check_mark: ```Objetivos Específicos ```

- :pushpin: Realizar un preprocesamiento de datos (ETL)
- :pushpin: Realizar un análisis exploratorio de datos (EDA) 
- :pushpin: Crear endpoints de API que permitan consultas específicas y recomendaciones de películas


## :white_check_mark: ```Metodología de trabajo```

Para llevar a cabo los objetivos, se ejecutaron los siguientes procedimientos utilizando diversas herramientas:

-  :one: ${\color{blue} \textbf{Google Drive}}$: se utilizó  para almacenar el conjunto de datos de los datasets, ambos conjuntos se descargaron en el notebook PI_Ruth_1_ETL.ipynb

- :two: ${\color{blue} \textbf{Visual Studio Code}}$: se utilizó este editor de código para crear un directorio con el nombre del proyecto y se implementó un entorno virtual de forma local. En este entorno se procedió a crear la API junto a los endpoints. Para la construcción de la Api se utilizó el framework de Python FastAPI.

Los endpoints desarrollados fueron: 

- ```def cantidad_peliculas_mes(mes)```: Se ingresa el mes en minúscula, por ejemplo abril, y la función retorna la cantidad de películas que se estrenaron en ese mes
    
- ```def cantidad_peliculas_dia(dia)```: Se ingresa el día en minúscula, por ejemplo sábado, y la función retorna la cantidad de películas que se estrenaron ese día
    
- ```def score_titulo(titulo)```: Se ingresa el título de una película, por ejemplo "Pocahontas", y se retorna el título, el año de estreno y la popularidad.
    

- ```def votos_titulo(titulo)```: Se ingresa el título de una película, por ejemplo "Powder", y se retorna el título, el año de estreno y el score.

- ```def get_actor(nombre_actor)```: Se ingresa el nombre de un actor, por ejemplo "Tom Hanks" y se retorna su éxito medido a través del retorno, cantidad de películas y promedio de retorno.

- ```def get_director(nombre_director)```: Se ingresa el nombre de un director y se retorna su éxito medido a través del retorno, nombre de cada película, fecha de lanzamiento, retorno individual, costo y ganancia.
    
       

    - **ETL:** se realizó limpieza y transformación de los datos para garantizar la calidad y consistencia de la información utilizada en el sistema de recomendación. El resultado del jupyter notebook (PI_Ruth_2_EDA.ipynb) desarrolado para esta etapa corresponde al conjunto de datos que se utilizó para alimentar a la Api, se lo descargó en formato parquet con el nombre  api_consult.parquet.
    - **EDA:** este análisis se realizó con la finalidad de identificar patrones, tendencias y relaciones en los datos, así como detectar posibles outliers y anomalías. Dicho análisis posibilitó decidir cuáles atributos eran los adecuados para aplicar el Modelo de Machine Learning. El resultado del jupyter notebook (EDPI_Ruth_2_EDA.ipynb) desarrolado para esta etapa corresponde al conjunto de datos que se utilizó para aplicar el modelo seleccionado.
    - **Modelo de Machine Learning (ML):** Para el modelado, se eligieron las técnicas TF-IDF (Frecuencia de Término-Inversa Frecuencia de Documento) y la similitud del coseno. Estas herramientas son fundamentales en el procesamiento de lenguaje natural (NLP), ya que permiten evaluar la importancia de los términos dentro de los documentos y calcular la similitud entre ellos. Los modelos implementados se encuentran detallados en el notebook 
    -  PI_Ruth_ML.ipynb. Se eligió el modelo que responda al siguiente endpoint: 
       - def recomendacion(titulo)```: Se ingresa el título de una película, por ejemplo "Balto", y devuelve 5 recomendaciones.
    

Todas las tareas realizadas se encuentran en gGithub. Para ejecutar cada notebook se sugiere descargarlo y ejecutar en Visual Studio Code.
- :three: ${\color{red} \textbf{Github}}$: se usó esta plataforma para almacenar el proyecto. Se creó un repositorió con el nombre **Proyecto-individual-1**. Este paso es imprescindible para deployar la Api en render, dado que se utiliza la dirección del repositorio para realizar el deploy. Cada cambio realizado a nivel local se iba actualizando en el repositorio

- :four: ${\color{red} \textbf{Render}}$: se utilizó este sitio para desplegar el proyecto. Primeramente se creó una cuenta en el sitio y luego se conectó con el repositorio de Github donde se encuentra alojado el proyecto. Se tuvo que tener mucho cuidado al elegir el ML ya que render tiene un límite de memoria de 512 Mb.
  
- :five: ${\color{red} \textbf{Google Colaboratory}}$:Esta plataforma fue empleada para llevar a cabo los procesos de ETL, análisis exploratorio de datos (EDA) y la construcción del modelo de Machine Learning.

    - **ETL:**: Se realizaron tareas de limpieza y transformación de los datos para asegurar la calidad y consistencia de la información utilizada en el sistema de recomendación. El resultado de esta etapa, documentado en el notebook PI_Ruth_1_ETL.ipynb, generó un conjunto de datos que alimenta la API, el cual fue exportado en formato parquet bajo el nombre api_consult.parquet.

    - **EDA:** Durante el análisis exploratorio, se identificaron patrones, tendencias y relaciones en los datos, además de detectar posibles valores atípicos y anomalías. Este análisis permitió seleccionar los atributos más relevantes para el modelo de Machine Learning. Los resultados de esta etapa están reflejados en el notebook PI_Ruth_1_EDA.ipynb, que sirvió como base para los datos del modelo.

    - **Modelo de Machine Learning:** Para el modelado se optó por utilizar TF-IDF (Term Frequency-Inverse Document Frequency) y la similitud del coseno, herramientas clave en el procesamiento de lenguaje natural (NLP) que permiten evaluar la relevancia de términos en documentos y calcular la similitud entre ellos. Los distintos modelos aplicados están detallados en el notebook PI_Ruth_1_ML.ipynb.




## :white_check_mark: :sparkles: ```Deployment de la Api``` :sparkles:
 
 Para realizar consultas y recomendaciones de películas dirigirse a la siguiente dirección: [Proyecto-individual-1](https://proyecto-individual-1-3azf.onrender.com/docs)




## :white_check_mark: ```Video```

El siguiente video muestra el funcionamiento de la Api, es útil como guía para que realices tus consultas: :clapper: [Video](https://drive.google.com/drive/folders/1E1wKhEQF6lTmvzC5doFMEG4sBqA2TAm1?usp=sharing)

