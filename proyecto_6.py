'''
Descripción del proyecto

Trabajas para la tienda online Ice que vende videojuegos por todo el mundo. Las reseñas de usuarios y 
expertos, los géneros, las plataformas (por ejemplo, Xbox o PlayStation) y los datos históricos sobre 
las ventas de juegos están disponibles en fuentes abiertas. Tienes que identificar patrones que 
determinen si un juego tiene éxito o no. Esto te permitirá detectar proyectos prometedores y 
planificar campañas publicitarias.

Delante de ti hay datos que se remontan a 2016. Imaginemos que es diciembre de 2016 y estás planeando 
una campaña para 2017.
Lo importante es adquirir experiencia de trabajo con datos. Realmente no importa si estás pronosticando 
las ventas de 2017 en función de los datos de 2016 o las ventas de 2027 en función de los datos de 2026.

El dataset contiene una columna "rating" que almacena la clasificación ESRB de cada juego. El 
Entertainment Software Rating Board (la Junta de clasificación de software de entretenimiento) evalúa el 
contenido de un juego y asigna una clasificación de edad como Adolescente o Adulto.

Instrucciones para completar el proyecto

Paso 1. Abre el archivo de datos y estudia la información general 

Ruta de archivo:

/datasets/games.csv . Descarga el dataset

Paso 2. Prepara los datos

Reemplaza los nombres de las columnas (ponlos en minúsculas).
Convierte los datos en los tipos necesarios.
Describe las columnas en las que los tipos de datos han sido cambiados y explica por qué.
Si es necesario, elige la manera de tratar los valores ausentes:
Explica por qué rellenaste los valores ausentes como lo hiciste o por qué decidiste dejarlos en blanco.
¿Por qué crees que los valores están ausentes? Brinda explicaciones posibles.
Presta atención a la abreviatura TBD: significa "to be determined" (a determinar). Especifica cómo 
piensas manejar estos casos.
Calcula las ventas totales (la suma de las ventas en todas las regiones) para cada juego y coloca estos 
valores en una columna separada.

Paso 3. Analiza los datos

Mira cuántos juegos fueron lanzados en diferentes años. ¿Son significativos los datos de cada período?
Observa cómo varían las ventas de una plataforma a otra. Elige las plataformas con las mayores ventas 
totales y construye una distribución basada en los datos de cada año. Busca las plataformas que solían ser 
populares pero que ahora no tienen ventas. ¿Cuánto tardan generalmente las nuevas plataformas en aparecer 
y las antiguas en desaparecer?
Determina para qué período debes tomar datos. Para hacerlo mira tus respuestas a las preguntas anteriores. Los datos deberían permitirte construir un modelo para 2017.
Trabaja solo con los datos que consideras relevantes. Ignora los datos de años anteriores.
¿Qué plataformas son líderes en ventas? ¿Cuáles crecen y cuáles se reducen? Elige varias plataformas 
potencialmente rentables.
Crea un diagrama de caja para las ventas globales de todos los juegos, desglosados por plataforma. ¿Son 
significativas las diferencias en las ventas? ¿Qué sucede con las ventas promedio en varias plataformas? 
Describe tus hallazgos.
Mira cómo las reseñas de usuarios y profesionales afectan las ventas de una plataforma popular (tu 
elección). Crea un gráfico de dispersión y calcula la correlación entre las reseñas y las ventas. Saca 
conclusiones.
Teniendo en cuenta tus conclusiones compara las ventas de los mismos juegos en otras plataformas.
Echa un vistazo a la distribución general de los juegos por género. ¿Qué se puede decir de los géneros 
más rentables? ¿Puedes generalizar acerca de los géneros con ventas altas y bajas?

Paso 4. Crea un perfil de usuario para cada región

Para cada región (NA, UE, JP) determina:

Las cinco plataformas principales. Describe las variaciones en sus cuotas de mercado de una región a otra.
Los cinco géneros principales. Explica la diferencia.
Si las clasificaciones de ESRB afectan a las ventas en regiones individuales.

Paso 5. Prueba las siguientes hipótesis:

— Las calificaciones promedio de los usuarios para las plataformas Xbox One y PC son las mismas.

— Las calificaciones promedio de los usuarios para los géneros de Acción y Deportes son diferentes.

Establece tu mismo el valor de umbral alfa.

Explica:

— Cómo formulaste las hipótesis nula y alternativa.

— Qué criterio utilizaste para probar las hipótesis y por qué.

Paso 6. Escribe una conclusión general

Formato: Completa la tarea en Jupyter Notebook. Inserta el código de programación en las celdas code y las 
explicaciones de texto en las celdas markdown. Aplica formato y agrega encabezados.

Descripción de datos
— Name (Nombre)

— Platform (Plataforma)

— Year_of_Release (Año de lanzamiento)

— Genre (Género) 

— NA_sales (ventas en Norteamérica en millones de dólares estadounidenses) 

— EU_sales (ventas en Europa en millones de dólares estadounidenses) 

— JP_sales (ventas en Japón en millones de dólares estadounidenses) 

— Other_sales (ventas en otros países en millones de dólares estadounidenses) 

— Critic_Score (máximo de 100) 

— User_Score (máximo de 10) 

— Rating (ESRB)

Es posible que los datos de 2016 estén incompletos.

¿Cómo será evaluado mi proyecto?
Lee atentamente estos criterios de evaluación de proyectos antes de empezar a trabajar.

Esto es lo que buscan los revisores de proyecto cuando evalúan tu proyecto:

¿Cómo describirías los problemas identificados en los datos?
¿Cómo se prepara un dataset para el análisis?
¿Cómo creas gráficos de distribución y cómo los explicas?
¿Cómo calculas la desviación estándar y varianza?
¿Formulas las hipótesis alternativas y nulas?
¿Qué métodos aplicas a la hora de probarlos?
¿Explicas los resultados de tus pruebas de hipótesis?
¿Sigues la estructura del proyecto y mantienes tu código ordenado y comprensible?
¿A qué conclusiones llegas?
¿Has dejado comentarios claros y relevantes en cada paso?



'''
# Paso 1. Abre el archivo de datos y estudia la información general 

#Importando librerias del proyecto
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import ttest_ind


# Cargando el dataset en un dataframe
game_data = pd.read_csv('games.csv')


# Mostrando las primeras filas del dataframe para estudiar la información general
print("Primeras filas del DataFrame:")
print(game_data.head(10))


# Mostrando la información general del dataframe
print("\nInformación general del DataFrame:")
print(game_data.info())


# Paso 2: Prepara los datos
# Reemplazando los nombres de las columnas por minúsculas
game_data.columns = game_data.columns.str.lower()


# Muestrando las primeras filas del dataframe con las columnas renombradas
print("\nPrimeras filas del DataFrame con columnas renombradas:")
print(game_data.head())


# Conversión de tipos de datos
# year_of_release de float64 a Int64 (uso Int64 para manejar NaN)
game_data['year_of_release'] = game_data['year_of_release'].astype('Int64')


# user_score de object a float
# Primero convertimos los valores no numéricos a NaN para hacer la conversión a float
game_data['user_score'] = pd.to_numeric(game_data['user_score'].replace('tbd', pd.NA), errors='coerce')


# Muestra las primeras filas del DataFrame con los tipos de datos convertidos
print("\nPrimeras filas del DataFrame con tipos de datos convertidos:")
print(game_data.head())


# Muestra la información general del DataFrame después de la conversión
print("\nInformación general del DataFrame después de la conversión:")
print(game_data.info())


# Tratamiento de valores ausentes
# 'name', 'genre', 'rating' - Llenar los valores ausentes con una cadena vacía
game_data['name'].fillna('', inplace=True)
game_data['genre'].fillna('', inplace=True)
game_data['rating'].fillna('', inplace=True)


# 'critic_score', 'user_score' - Rellenar con la media de la columna
game_data['critic_score'].fillna(game_data['critic_score'].mean(), inplace=True)
game_data['user_score'].fillna(game_data['user_score'].mean(), inplace=True)
# 'year_of_release' - Dejar valores ausentes como NaN
# 'year_of_release' contiene información crítica que no debe ser adivinada o interpolada


# Muestra las primeras filas del DataFrame después de tratar los valores ausentes
print("\nPrimeras filas del DataFrame después de tratar los valores ausentes:")
print(game_data.head())


# Muestra la información general del DataFrame después de tratar los valores ausentes
print("\nInformación general del DataFrame después de tratar los valores ausentes:")
print(game_data.info())


# Cálculo de ventas totales
game_data['total_sales'] = game_data[['na_sales', 'eu_sales', 'jp_sales', 'other_sales']].sum(axis=1)


# Muestra las primeras filas del DataFrame después de tratar los valores ausentes y añadir la columna de ventas totales
print("\nPrimeras filas del DataFrame después de tratar los valores ausentes y añadir la columna de ventas totales:")
print(game_data.head())


# Muestra la información general del DataFrame después de tratar los valores ausentes y añadir la columna de ventas totales
print("\nInformación general del DataFrame después de tratar los valores ausentes y añadir la columna de ventas totales:")
print(game_data.info())


# Paso 3. Analiza los datos
# Análisis del número de juegos lanzados por año

# Contar el número de juegos lanzados cada año
games_per_year = game_data['year_of_release'].value_counts().sort_index()
# Mostrar el número de juegos lanzados por año
print("Número de juegos lanzados por año:")
print(games_per_year)


# Visualizar el número de juegos lanzados por año como un gráfico de barras
plt.figure(figsize=(12, 6))
plt.bar(games_per_year.index.astype(str), games_per_year.values)
plt.title('Número de juegos lanzados por año')
plt.xlabel('Año de lanzamiento')
plt.ylabel('Número de juegos')
plt.xticks(rotation=90)  # Rotar etiquetas del eje x para mejor legibilidad
plt.grid(axis='y')
plt.show()


# Observación de las ventas por plataforma

# Sumamos las ventas totales por plataforma
platform_sales = game_data.groupby('platform')['total_sales'].sum().sort_values(ascending=False)


# Mostramos las ventas totales por plataforma
print("Ventas totales por plataforma:")
print(platform_sales)


# Identificación de plataformas con mayores ventas totales y construcción de distribución
# Seleccionamos las plataformas con mayores ventas totales
top_platforms = platform_sales.head(5).index


# Filtramos el DataFrame para solo incluir las plataformas seleccionadas
filtered_data = game_data[game_data['platform'].isin(top_platforms)]


# Agrupamos los datos por año y plataforma y sumamos las ventas
yearly_sales = filtered_data.groupby(['year_of_release', 'platform'])['total_sales'].sum().unstack().fillna(0)


# Graficamos la distribución de ventas por año para las plataformas populares
plt.figure(figsize=(12, 6))
yearly_sales.plot(kind='bar', stacked=True)
plt.title('Distribución de ventas por año para las plataformas populares')
plt.xlabel('Año de lanzamiento')
plt.ylabel('Ventas totales')
plt.legend(title='Plataforma')
plt.grid(axis='y')
plt.show()


# Análisis de plataformas populares que ya no tienen ventas
# Identificamos las plataformas que solían ser populares
popular_platforms = platform_sales[platform_sales > platform_sales.mean()].index


# Filtramos el DataFrame para solo incluir las plataformas seleccionadas
filtered_popular_data = game_data[game_data['platform'].isin(popular_platforms)]


# Agrupamos los datos por plataforma y año y sumamos las ventas
platform_yearly_sales = filtered_popular_data.groupby(['platform', 'year_of_release'])['total_sales'].sum().unstack().fillna(0)


# Mostramos las plataformas que solían ser populares pero que ahora no tienen ventas
print("Plataformas que solían ser populares pero que ahora no tienen ventas:")
for platform in popular_platforms:
    if platform_yearly_sales.loc[platform].tail(1).values[0] == 0:
        print(platform)


# Tiempo de aparición y desaparición de plataformas
# Calculamos el tiempo de vida de cada plataforma
platform_lifespan = filtered_popular_data.groupby('platform')['year_of_release'].agg(['min', 'max'])
platform_lifespan['lifespan'] = platform_lifespan['max'] - platform_lifespan['min']


# Mostramos el tiempo de vida de las plataformas
print("Tiempo de vida de las plataformas:")
print(platform_lifespan)


# Creación de un diagrama de caja para las ventas globales de todos los juegos, desglosados por plataforma
# Crear un diagrama de caja para las ventas globales de todos los juegos, desglosados por plataforma
plt.figure(figsize=(12, 6))
sns.boxplot(data=game_data, x='platform', y='total_sales')
plt.title('Diagrama de caja de ventas globales por plataforma')
plt.xlabel('Plataforma')
plt.ylabel('Ventas globales')
plt.xticks(rotation=90)
plt.show()


# Análisis de reseñas de usuarios y profesionales
# Graficar la relación entre las reseñas y las ventas para una plataforma popular
popular_platform = 'PS4'  # Ejemplo de plataforma popular
platform_data = game_data[game_data['platform'] == popular_platform]


plt.figure(figsize=(12, 6))
sns.scatterplot(data=platform_data, x='critic_score', y='total_sales', label='Critic Score')
sns.scatterplot(data=platform_data, x='user_score', y='total_sales', label='User Score')
plt.title(f'Relación entre reseñas y ventas para {popular_platform}')
plt.xlabel('Score')
plt.ylabel('Ventas Totales')
plt.legend()
plt.show()


# Calcular la correlación entre las reseñas y las ventas
correlation_critic = platform_data['critic_score'].corr(platform_data['total_sales'])
correlation_user = platform_data['user_score'].corr(platform_data['total_sales'])

print(f"Correlación entre reseñas de críticos y ventas para {popular_platform}: {correlation_critic}")
print(f"Correlación entre reseñas de usuarios y ventas para {popular_platform}: {correlation_user}")


# Distribución general de los juegos por género
genre_distribution = game_data['genre'].value_counts()


# Graficar la distribución de juegos por género
plt.figure(figsize=(12, 6))
sns.barplot(x=genre_distribution.index, y=genre_distribution.values)
plt.title('Distribución de juegos por género')
plt.xlabel('Género')
plt.ylabel('Número de juegos')
plt.xticks(rotation=90)
plt.show()


#Análisis de los géneros más rentables:
# Sumamos las ventas totales por género
genre_sales = game_data.groupby('genre')['total_sales'].sum().sort_values(ascending=False)


# Graficar las ventas totales por género
plt.figure(figsize=(12, 6))
sns.barplot(x=genre_sales.index, y=genre_sales.values)
plt.title('Ventas totales por género')
plt.xlabel('Género')
plt.ylabel('Ventas Totales')
plt.xticks(rotation=90)
plt.show()


#Paso 4. Crea un perfil de usuario para cada región
# Define las regiones
regions = ['na_sales', 'eu_sales', 'jp_sales']
# Inicializa diccionarios para almacenar resultados
top_platforms_per_region = {}
platform_market_share = {}


# Calcula las cinco plataformas principales y las cuotas de mercado para cada región
for region in regions:
    # Calcula las ventas totales por plataforma en la región
    platform_sales = game_data.groupby('platform')[region].sum().sort_values(ascending=False)
    # Selecciona las cinco plataformas principales
    top_platforms = platform_sales.head(5)
    # Almacena los resultados
    top_platforms_per_region[region] = top_platforms
    # Calcula las cuotas de mercado
    market_share = (top_platforms / platform_sales.sum()) * 100
    platform_market_share[region] = market_share


# Mostrar resultados
for region in regions:
    print(f"\nLas cinco plataformas principales en {region.upper()}:")
    print(top_platforms_per_region[region])
    print(f"\nCuotas de mercado en {region.upper()}:")
    print(platform_market_share[region])


#Los cinco géneros principales y su comparación entre regiones
# Inicializa diccionarios para almacenar resultados
top_genres_per_region = {}
# Calcula los cinco géneros principales para cada región
for region in regions:
    # Calcula las ventas totales por género en la región
    genre_sales = game_data.groupby('genre')[region].sum().sort_values(ascending=False)
    # Selecciona los cinco géneros principales
    top_genres = genre_sales.head(5)
    # Almacena los resultados
    top_genres_per_region[region] = top_genres


# Mostrar resultados
for region in regions:
    print(f"\nLos cinco géneros principales en {region.upper()}:")
    print(top_genres_per_region[region])


#Impacto de las clasificaciones de ESRB en las ventas en regiones individuales
# Inicializa diccionarios para almacenar resultados
esrb_sales_per_region = {}

# Calcula las ventas por clasificación ESRB para cada región
for region in regions:
    # Calcula las ventas totales por clasificación ESRB en la región
    esrb_sales = game_data.groupby('rating')[region].sum().sort_values(ascending=False)
    # Almacena los resultados
    esrb_sales_per_region[region] = esrb_sales


# Mostrar resultados
for region in regions:
    print(f"\nVentas por clasificación ESRB en {region.upper()}:")
    print(esrb_sales_per_region[region])


#Gráficos de las cinco plataformas principales por región
plt.figure(figsize=(15, 8))
for i, region in enumerate(regions):
    plt.subplot(2, 2, i+1)
    sns.barplot(x=top_platforms_per_region[region].index, y=top_platforms_per_region[region].values)
    plt.title(f'Plataformas principales en {region.upper()}')
    plt.xlabel('Plataforma')
    plt.ylabel('Ventas totales')

plt.tight_layout()
plt.show()


#Gráficos de los cinco géneros principales por región
plt.figure(figsize=(15, 8))
for i, region in enumerate(regions):
    plt.subplot(2, 2, i+1)
    sns.barplot(x=top_genres_per_region[region].index, y=top_genres_per_region[region].values)
    plt.title(f'Géneros principales en {region.upper()}')
    plt.xlabel('Género')
    plt.ylabel('Ventas totales')

plt.tight_layout()
plt.show()


#Gráficos de las ventas por clasificación ESRB por región
plt.figure(figsize=(15, 8))
for i, region in enumerate(regions):
    plt.subplot(2, 2, i+1)
    sns.barplot(x=esrb_sales_per_region[region].index, y=esrb_sales_per_region[region].values)
    plt.title(f'Ventas por clasificación ESRB en {region.upper()}')
    plt.xlabel('Clasificación ESRB')
    plt.ylabel('Ventas totales')

plt.tight_layout()
plt.show()


# Paso 5. Prueba de hipótesis
"""
Hipótesis 1: Calificaciones promedio de usuarios para Xbox One y PC

Hipótesis Nula (H0): Las calificaciones promedio de los usuarios para las plataformas Xbox One y PC son iguales.
Hipótesis Alternativa (H1): Las calificaciones promedio de los usuarios para las plataformas Xbox One y PC son diferentes.

Hipótesis 2: Calificaciones promedio de usuarios para los géneros Acción y Deportes

Hipótesis Nula (H0): Las calificaciones promedio de los usuarios para los géneros Acción y Deportes son iguales.
Hipótesis Alternativa (H1): Las calificaciones promedio de los usuarios para los géneros Acción y Deportes son diferentes.

Estableciendo el umbral alfa
Se Usará un valor de alfa de 0.05, que es comúnmente utilizado en pruebas de hipótesis.

Realización de las pruebas
Para probar estas hipótesis, utilizaremos la prueba t de Student para muestras independientes. Esta prueba es adecuada para comparar 
las medias de dos grupos independientes.
"""


# Hipótesis 1: Xbox One vs PC
xbox_one_scores = game_data[game_data['platform'] == 'XOne']['user_score']
pc_scores = game_data[game_data['platform'] == 'PC']['user_score']

'''
Hipótesis 1: Calificaciones promedio de usuarios para Xbox One y PC

La hipótesis nula (H0) es una declaración de que no hay efecto o diferencia. H0 asume que las calificaciones promedio de los usuarios 
para las plataformas Xbox One y PC son iguales, es decir, cualquier diferencia observada se debe al azar.
La hipótesis alternativa (H1) es una declaración que propone una diferencia o efecto. H1 sugiere que las calificaciones promedio de los 
usuarios para las plataformas Xbox One y PC no son iguales y que la diferencia observada es significativa.
'''

# Realización de la prueba t
t_stat_xbox_pc, p_val_xbox_pc = ttest_ind(xbox_one_scores, pc_scores, equal_var=False)
print(f"Prueba t para Xbox One vs PC: t-stat = {t_stat_xbox_pc}, p-value = {p_val_xbox_pc}")


# Hipótesis 2: Acción vs Deportes
accion_scores = game_data[game_data['genre'] == 'Action']['user_score']
deportes_scores = game_data[game_data['genre'] == 'Sports']['user_score']

'''
Hipótesis 2: Calificaciones promedio de usuarios para los géneros Acción y Deportes

H0 asume que no hay diferencia significativa entre las calificaciones promedio de los usuarios para los géneros de Acción y Deportes.
H1 sugiere que hay una diferencia significativa entre las calificaciones promedio de los usuarios para los géneros de Acción y Deportes.
'''

# Realización de la prueba t
t_stat_action_sports, p_val_action_sports = ttest_ind(accion_scores, deportes_scores, equal_var=False)
print(f"Prueba t para Acción vs Deportes: t-stat = {t_stat_action_sports}, p-value = {p_val_action_sports}")


'''
Se usa la prueba t de Student para muestras independientes debido a la independencia de las muestras, la suposición de normalidad para 
grandes muestras, y la consideración de varianzas desiguales. El valor p se comparó con un umbral alfa de 0.05 para determinar si se 
rechaza la hipótesis nula.
'''


# Resultados
if p_val_xbox_pc < 0.05:
    print("Rechazamos la hipótesis nula para Xbox One vs PC: Las calificaciones promedio de los usuarios son diferentes.")
else:
    print("No podemos rechazar la hipótesis nula para Xbox One vs PC: Las calificaciones promedio de los usuarios son iguales.")

if p_val_action_sports < 0.05:
    print("Rechazamos la hipótesis nula para Acción vs Deportes: Las calificaciones promedio de los usuarios son diferentes.")
else:
    print("No podemos rechazar la hipótesis nula para Acción vs Deportes: Las calificaciones promedio de los usuarios son iguales.")


'''
Criterio para Probar las Hipótesis

Criterio Utilizado: Prueba t de Student para Muestras Independientes

La prueba t de Student para muestras independientes es adecuada para comparar las medias de dos grupos independientes para determinar si 
hay una diferencia significativa entre ellas. En este caso, los grupos son:

Plataformas: Xbox One y PC.
Géneros: Acción y Deportes.

Los datos de calificaciones de usuarios para Xbox One y PC, así como para los géneros de Acción y Deportes, son independientes entre sí.
La prueba t asume que las muestras provienen de poblaciones con una distribución aproximadamente normal, lo cual es razonable suponer 
para grandes muestras.
Se usa la versión "igual_var=False" de la prueba t para tener en cuenta la posible diferencia en las varianzas de los dos grupos.

El valor p nos indica la probabilidad de obtener un resultado tan extremo como el observado, suponiendo que la hipótesis nula sea 
verdadera. Un valor p bajo sugiere que la hipótesis nula es poco probable.
Establecemos alpha en 0.05. Si el valor p es menor que alpha, rechazamos la hipótesis nula a favor de la hipótesis alternativa.

Calcular las medias y desviaciones estándar de las calificaciones de usuarios para cada grupo.
Realizar la prueba t de Student para muestras independientes para comparar las medias de los grupos.
Observar el valor p resultante.
Comparar el valor p con el umbral alfa:
Si el valor p < 0.05, rechazamos la hipótesis nula.
Si el valor p ≥ 0.05, no podemos rechazar la hipótesis nula.
'''

# Paso 6. Escribe una conclusión general
'''
En este proyecto, se llevó a cabo un análisis exhaustivo del conjunto de datos de videojuegos, abarcando aspectos cruciales como las 
ventas, la popularidad de las plataformas y géneros, así como las calificaciones de usuarios y críticos. Los hallazgos obtenidos 
proporcionan información valiosa sobre las dinámicas del mercado de videojuegos y las preferencias de los consumidores.

Ventas por Plataforma:

Las plataformas con mayores ventas totales incluyen PS2, X360 y PS3. Estas consolas han demostrado ser las preferidas por los jugadores 
durante su ciclo de vida.
A lo largo de los años, se observa que algunas plataformas han desaparecido del mercado, mientras que otras han mantenido su popularidad. 
Este patrón es crucial para entender la evolución y el ciclo de vida de las consolas.

Ventas por Género:

Los géneros de Acción, Deportes y Disparos son los más populares en términos de ventas. Estos géneros dominan el mercado y reflejan las 
preferencias de los jugadores en múltiples regiones.
La distribución de géneros muestra variaciones regionales en las preferencias de los jugadores, lo que sugiere la necesidad de estrategias 
de marketing adaptadas a cada región.

Análisis de Calificaciones

En la plataforma PS4, se observó una correlación positiva entre las calificaciones de críticos y las ventas totales, mientras que la 
correlación con las calificaciones de usuarios fue menor. Esto indica que las calificaciones de críticos pueden tener un mayor impacto en 
las ventas.

Los hallazgos de este análisis proporcionan una comprensión profunda de las tendencias del mercado de videojuegos y las preferencias de 
los consumidores. Las conclusiones obtenidas pueden ser utilizadas por desarrolladores y publicistas para:

Lanzamiento de Nuevos Juegos: Tomar decisiones informadas sobre en qué plataformas lanzar nuevos juegos.

Estrategias de Marketing Regionales: Entender las preferencias regionales y adaptar las estrategias de marketing en consecuencia.

Influencia de las Calificaciones: Evaluar el impacto de las calificaciones de críticos y usuarios en las ventas y ajustar sus estrategias 
de relaciones públicas.

En conclusión, este análisis no solo revela las dinámicas actuales del mercado de videojuegos, sino que también ofrece una base sólida 
para futuras investigaciones y decisiones estratégicas en la industria. Los datos y métodos utilizados permiten a los actores del sector 
identificar oportunidades y responder de manera efectiva a las cambiantes demandas del mercado.
'''