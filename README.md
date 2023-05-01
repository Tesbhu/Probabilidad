# Probabilidad
-----------------------

La probabilidad es una medida numérica que describe la posibilidad de que ocurra un evento específico. Se utiliza para cuantificar la incertidumbre y la aleatoriedad en diferentes situaciones. La probabilidad se expresa como un número entre 0 y 1, donde 0 indica que el evento es imposible de ocurrir y 1 indica que el evento es seguro de ocurrir.

En términos más formales, la probabilidad se define como el cociente entre el número de eventos favorables y el número total de posibles resultados. Por ejemplo, si lanzamos un dado justo, el dado tiene 6 caras y cada cara tiene la misma probabilidad de aparecer. Entonces, la probabilidad de que salga un 3 es 1/6, ya que hay una cara con el número 3 y seis posibles resultados en total.

La probabilidad también puede expresarse como un porcentaje o una fracción. Por ejemplo, una probabilidad del 50% se puede expresar como 0.5 o como 1/2.

Existen diferentes tipos de probabilidades, como la probabilidad condicional, la probabilidad conjunta y la probabilidad marginal, que se utilizan para describir la probabilidad de eventos relacionados o múltiples eventos.

La teoría de la probabilidad es una rama fundamental de las matemáticas y tiene aplicaciones en una amplia variedad de campos, como la estadística, la ciencia, la ingeniería, la economía, la toma de decisiones, la teoría de juegos, la inteligencia artificial y más. Se utiliza para analizar y predecir eventos aleatorios, tomar decisiones basadas en incertidumbre, diseñar experimentos y modelar fenómenos complejos.

En mi opinión se debe seguir un plan de conocimientos como el siguiente:

1. Espacio muestral: Es el conjunto de todos los posibles resultados de un experimento aleatorio. Por ejemplo, al lanzar un dado, el espacio muestral sería {1, 2, 3, 4, 5, 6}.

2. Evento: Es un subconjunto del espacio muestral, que consiste en uno o más resultados. Puede ser un evento simple (un solo resultado) o un evento compuesto (varios resultados). Por ejemplo, en el lanzamiento de un dado, el evento "obtener un número par" sería {2, 4, 6}.

3. Probabilidad de un evento: Es una medida numérica que representa la posibilidad de que ocurra un evento. Se denota como P(evento) y se encuentra entre 0 y 1. Un evento imposible tiene una probabilidad de 0, mientras que un evento seguro tiene una probabilidad de 1.

4. Regla de la suma: Esta regla establece que la probabilidad de que ocurra uno de dos eventos mutuamente excluyentes (no pueden ocurrir simultáneamente) es igual a la suma de las probabilidades de cada evento individual. Por ejemplo, la probabilidad de obtener un número par o un número impar en un lanzamiento de dado justo es 1/2 + 1/2 = 1.

5. Regla del complemento: Esta regla establece que la probabilidad de que ocurra un evento complementario (el evento contrario al evento original) es igual a 1 menos la probabilidad del evento original. Por ejemplo, si la probabilidad de obtener un número impar en un lanzamiento de dado justo es 1/2, entonces la probabilidad de obtener un número par sería 1 - 1/2 = 1/2.

6. Regla del producto: Esta regla se aplica cuando se calcula la probabilidad conjunta de dos eventos independientes. La probabilidad conjunta de dos eventos independientes es igual al producto de las probabilidades de cada evento individual. Por ejemplo, la probabilidad de obtener un 4 en el primer lanzamiento de un dado justo y un 2 en el segundo lanzamiento sería 1/6 * 1/6 = 1/36.

Estos son solo algunos de los temas básicos de probabilidad. A medida que se profundiza en el estudio de la probabilidad, se exploran conceptos más avanzados como la probabilidad condicional, la independencia de eventos, las distribuciones de probabilidad, el teorema de Bayes y mucho más.

Antes de abordar temas más complejos veamos como implementar las probabilidades en python:

- Problema: Crea un programa en python que me diga cual es la probabilidad de sacar una bola azul de una urna con 10 bolas blancas, 6 rojas y 4 azules

Código: 

```Python
# Importar la biblioteca random para realizar la selección aleatoria
import random

# Definir el número de simulaciones
num_simulaciones = 100000

# Contadores para el número de veces que se saca una bola azul y el número total de simulaciones
contador_azul = 0
contador_total = 0

# Realizar las simulaciones
for _ in range(num_simulaciones):
    urna = ['blanca'] * 10 + ['roja'] * 6 + ['azul'] * 4
    
    # Seleccionar una bola aleatoriamente
    bola_seleccionada = random.choice(urna)
    
    # Verificar si la bola seleccionada es azul
    if bola_seleccionada == 'azul':
        contador_azul += 1
    
    contador_total += 1

# Calcular la probabilidad de sacar una bola azul
probabilidad_azul = contador_azul / contador_total

# Imprimir el resultado
print("La probabilidad de sacar una bola azul es:", probabilidad_azul)
```
Si has ejecutado el programa resulta que:


La probabilidad de sacar una bola azul de una urna con 10 bolas blancas, 6 bolas rojas y 4 bolas azules se puede calcular dividiendo el número de bolas azules entre el número total de bolas en la urna.

En este caso, hay 4 bolas azules y un total de 10 bolas blancas + 6 bolas rojas + 4 bolas azules, es decir, 20 bolas en total.

La probabilidad de sacar una bola azul es:

probabilidad_azul = 4 / 20 = 0.2 = 20%

Por lo tanto, **la probabilidad de sacar una bola azul de la urna es del 20% o 0.2**.

------------------

El Teorema de Bayes es una herramienta fundamental en la teoría de la probabilidad y se utiliza para actualizar la probabilidad de un evento basado en nueva información o evidencia. La fórmula general del Teorema de Bayes es la siguiente:

$$P(A|B) = (P(B|A) * P(A)) / P(B)$$

Donde:

- $P(A|B)$ es la probabilidad condicional de $A$ dado $B$.
- $P(B|A)$ es la probabilidad condicional de $B$ dado $A$.
- $P(A)$ y $P(B)$ son las probabilidades marginales de A y B, respectivamente.

El Teorema de Bayes se puede utilizar para realizar inferencias y tomar decisiones en una amplia gama de problemas, desde la medicina hasta la inteligencia artificial.

Ejemplo de cómo aplicar el Teorema de Bayes en Python para dos urnas con bolas de 3 colores distintos (rojo, verde y azul):

Supongamos que tenemos dos urnas: Urna 1 y Urna 2. En Urna 1 hay 4 bolas rojas, 3 bolas verdes y 3 bolas azules. En Urna 2 hay 2 bolas rojas, 6 bolas verdes y 2 bolas azules. Seleccionamos una bola al azar de una de las urnas y resulta ser roja.

Queremos calcular la probabilidad de que la bola haya sido seleccionada de la Urna 1, dado que es roja.
```Python
# Definir las probabilidades a priori de seleccionar una urna
p_urna1 = 0.5  # Probabilidad a priori de seleccionar Urna 1
p_urna2 = 0.5  # Probabilidad a priori de seleccionar Urna 2

# Definir las probabilidades condicionales de seleccionar una bola roja dada cada urna
p_roja_dado_urna1 = 4 / 10  # Probabilidad de seleccionar una bola roja dado Urna 1
p_roja_dado_urna2 = 2 / 10  # Probabilidad de seleccionar una bola roja dado Urna 2

# Calcular la probabilidad a posteriori de que la bola haya sido seleccionada de Urna 1
p_urna1_dado_roja = (p_roja_dado_urna1 * p_urna1) / ((p_roja_dado_urna1 * p_urna1) + (p_roja_dado_urna2 * p_urna2))

# Imprimir el resultado
print("La probabilidad de que la bola haya sido seleccionada de la Urna 1, dado que es roja, es:", p_urna1_dado_roja)

```
En este ejemplo, se aplicó el Teorema de Bayes para calcular la probabilidad a posteriori de que la bola haya sido seleccionada de la Urna 1, dado que es roja. Se utilizaron las probabilidades a priori de seleccionar cada urna y las probabilidades condicionales de seleccionar una bola roja dado cada urna. El resultado se muestra al final del programa.

Es importante destacar que el ejemplo anterior asume que la selección de la urna es equiprobable. En aplicaciones reales, las probabilidades a priori pueden variar y deben ajustarse según la información y el contexto específico del problema.

## Temas un poco más avanzados

5. Distribuciones de probabilidad: En probabilidad y estadística, existen diferentes distribuciones de probabilidad que describen las posibles ocurrencias y sus probabilidades asociadas en un experimento o fenómeno. Algunas de las distribuciones más comunes incluyen la distribución binomial, la distribución normal y la distribución de Poisson.

6. Variables aleatorias: Una variable aleatoria es una variable que toma valores numéricos según los resultados de un experimento aleatorio. Puede ser discreta o continua, y su función de probabilidad o densidad de probabilidad describe cómo se distribuyen sus valores posibles.

7. Esperanza matemática y varianza: La esperanza matemática, o valor esperado, es una medida de la tendencia central de una variable aleatoria. La varianza, por otro lado, mide la dispersión de los valores alrededor de la esperanza matemática. Estos conceptos son fundamentales en el análisis de probabilidades y proporcionan información importante sobre las características de una distribución de probabilidad.

### Ejemplo de la distribución de Poisson

La distribución de Poisson es una distribución de probabilidad discreta que describe la probabilidad de que ocurra un número determinado de eventos en un intervalo de tiempo o espacio específico, dado que los eventos ocurren a una tasa constante e independiente del tiempo. En Python, se puede usar la función poisson.pmf() del módulo scipy.stats para calcular la probabilidad de ocurrencia de un número específico de eventos.

Un ejemplo de aplicación de la distribución de Poisson en la vida real es el conteo de accidentes automovilísticos en una ciudad en un período de tiempo determinado. Supongamos que en promedio ocurren 5 accidentes por día en la ciudad. Podemos modelar el número de accidentes en un solo día como una distribución de Poisson con una tasa lambda de 5. Luego, podemos usar la función poisson.pmf() para calcular la probabilidad de que ocurran 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 o más accidentes en un día específico.

A continuación, se muestra un ejemplo de código en Python que calcula la probabilidad de que ocurran 2 accidentes en un día con una tasa lambda de 5:

```Python
from scipy.stats import poisson

lam = 5  # tasa lambda
k = 2  # número de eventos

probabilidad = poisson.pmf(k, lam)
print("La probabilidad de que ocurran", k, "accidentes en un día es:", probabilidad)
```

La salida del código será:

**La probabilidad de que ocurran 2 accidentes en un día es: 0.08422433748856833**

Una probilidad bastabte alta y desgraciadamente realista.

Para comparar la distribución de Poisson con una distribución normal, podemos graficar ambas distribuciones para el mismo valor de lambda. A continuación, se muestra un ejemplo de código en Python que genera y grafica ambas distribuciones para una tasa lambda de 5:

```Python
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, poisson

# tasa lambda
lam = 5

# valores de k para la distribución de Poisson
k_poisson = np.arange(0, 15)

# valores de x para la distribución normal
x_normal = np.linspace(0, 15, 100)

# cálculo de las probabilidades para ambas distribuciones
probs_poisson = poisson.pmf(k_poisson, lam)
probs_normal = norm.pdf(x_normal, lam, np.sqrt(lam))

# graficar ambas distribuciones
plt.plot(k_poisson, probs_poisson, 'bo', label='Poisson')
plt.plot(x_normal, probs_normal, 'r-', label='Normal')

plt.xlabel('k/x')
plt.ylabel('Probabilidad')
plt.legend()
plt.show()d)
```
<div align="center">
    <h1></h1>
    <img src="Readme_images/fig 1.png" width="">
</div>
En la gráfica, se puede observar que ambas distribuciones son similares cuando la tasa lambda es grande (como en este caso), pero difieren más notablemente en valores más pequeños de lambda.

### Ejemplo de la distribución binomial 


La distribución binomial es una distribución de probabilidad discreta que modela el número de éxitos en una serie de ensayos independientes, donde cada ensayo tiene solo dos resultados posibles: éxito o fracaso. La distribución binomial se caracteriza por dos parámetros: el número de ensayos (n) y la probabilidad de éxito en cada ensayo (p).

**Un ejemplo de aplicación de la distribución binomial en la vida real es el análisis de la efectividad de una campaña publicitaria en línea. Supongamos que se envían 1000 anuncios a través de una plataforma de publicidad y se sabe que la tasa de clics promedio es del 5%. Podemos modelar el número de clics como una distribución binomial con n = 1000 y p = 0.05. Luego, podemos usar la distribución binomial para calcular la probabilidad de obtener un número específico de clics, como 50, 100, 200, etc.**

Para implementar la distribución binomial en Python, podemos utilizar la función `binom.pmf()` del módulo `scipy.stats`. Sin embargo, también puedes usar la biblioteca pandas para generar muestras de la distribución binomial y realizar cálculos adicionales. Pandas proporciona una forma conveniente de trabajar con datos tabulares y realizar análisis estadísticos.

Ejemplo de implementación de la distribución binomial en pandas:
```Python
import pandas as pd
from scipy.stats import binom

n = 1000  # número de ensayos
p = 0.05  # probabilidad de éxito

# Crear un DataFrame con una columna que contiene muestras de la distribución binomial
df = pd.DataFrame({'Clics': binom.rvs(n, p, size=1000)})

# Calcular la probabilidad de obtener exactamente 50 clics
prob_50_clics = binom.pmf(50, n, p)
print("La probabilidad de obtener exactamente 50 clics es:", prob_50_clics)

# Calcular la probabilidad acumulada de obtener menos de 100 clics
prob_menos_100_clics = binom.cdf(100, n, p)
print("La probabilidad acumulada de obtener menos de 100 clics es:", prob_menos_100_clics)

# Calcular la media y la desviación estándar de la distribución binomial
media = binom.mean(n, p)
desviacion_estandar = binom.std(n, p)
print("Media:", media)
print("Desviación estándar:", desviacion_estandar)
```
En este ejemplo, se crea un DataFrame de pandas con una columna llamada "Clics" que contiene muestras de la distribución binomial con n = 1000 y p = 0.05. Luego, se calcula la probabilidad de obtener exactamente 50 clics utilizando binom.pmf(), la probabilidad acumulada de obtener menos de 100 clics utilizando binom.cdf(), y se calcula la media y la desviación estándar de la distribución binomial utilizando binom.mean() y binom.std() respectivamente.

