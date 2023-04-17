<h2>RODIUCAN</h2>
<p>Siglas del modelo <strong>RODIUCAN</strong> obtenidas de "OPTIMIZACIÓN DE LA RESOLUCIÓN DE IMÁGENES DIGITALES UTILIZANDO AUTÓMATAS CELULARES Y REDES NEURONALES ARTIFICIALES".
</p>
<p>Es un trabajo de tesis que propone un modelo para aumentar la resolución de imágenes utilizando autómatas celulares y redes neuronales convolucionales, considerando un conjunto de datos reducido para optimizar el uso de los recursos computacionales.</p>

<h3>Códio fuente y ejemplos</h3>
<p>
<a target="_blank" href="https://github.com/rolandex25g/rodiucan.git">RODIUCAN github</a>
En github están los modelos en .H5, los cuadernos, el conjunto de datos original. El mínimo para ejecutar el modelo ya entrenado.
Tamaño aproximado: 600 MB
</p>

<p>
<a target="_blank" href="https://gitlab.com/rolandex25/rodiucan.git">RODIUCAN gitlab</a>
En gitlab están los modelos en .H5, los cuadernos, el dataset original, el dataset preprocesado, las pruebas. todo lo que necesario
para repetir el experimento con el mismo u otro conjunto de datos.
Tamaño aproximado: 4.6GB
</p>

<p>
Un ejemplo demostrativo también está disponible en Hugging Face:
<a target="_blank" href="https://huggingface.co/spaces/rolandex25/RODIUCAN-Demo">RODIUCAN example Hugging Face</a>
</p>

<p>Los cuadernos de colab también están disponibles en línea para su revisión. Y se enumeran segun su orden de desarrollo.</p>

<p>
1. Prueba inicial con una sola imagen. 
<a target="_blank" href="https://colab.research.google.com/drive/1u71QG7u_CsVgDqMQdw10Z6k7vFD-Up9c/view?usp=sharing">Cuaderno 1</a>
</p>
<p>
2. Pre-procesamiento de todas las imágenes. 
<a target="_blank" href="https://colab.research.google.com/drive/1LavGy5UX8Vc7iYs-3OVay6CwRBr68RPC/view?usp=sharing">Cuaderno 2</a>
</p>
<p>
3. Entrenar el modelo para imágenes en escala de grises. 
<a target="_blank" href="https://colab.research.google.com/drive/195EtfZRL-Grkdopl67l0tCnWVOl675BF/view?usp=sharing">Cuaderno 3</a>
</p>
<p>
4. Entrenar el modelo para imágenes a color. 
<a target="_blank" href="https://colab.research.google.com/drive/12f3jdIrYLPgY2lPSOL6gzwZ4U46EzDh6/view?usp=sharing">Cuaderno 4</a>
</p>
<p>
5. Comparar resultados del modelo propuesto RODIUCAN con el modelo ESRGAN. 
<a target="_blank" href="https://colab.research.google.com/drive/186MRR2JzVXVXx-CdtpoDPZgeW3mog4Sd/view?usp=sharing">Cuaderno 5</a>
</p>
<p>
6. <b>Ejemplo de uso</b> del modelo h5 en colores. 
<a target="_blank" href="https://colab.research.google.com/drive/1_mJ_Sy0HtPsGu-g72wqrdvW8tFL1gYEE/view?usp=sharing">Cuaderno 6</a>
</p>

<h3>Entrenamiento</h3>

<p>El entrenamiento se realizó con el conjunto de datos "General100" y con parches de 128x128 píxeles.</p>

<h3>Descripción del modelo</h3>
<p>Se utilizaron autómatas celulares para detectar los bordes y fondos de las imágenes, obteniendo así una representación simplificada pero muy descriptiva de los bordes de los objetos en las imágenes.</p>

<p>Estas imágenes luego se usaron como un espacio intermedio de transformación entre dos redes neuronales convolucionales para escalar la resolución de la imagen. Las redes neuronales se entrenaron por separado y luego se combinaron usando un esquema similar a una red generativa condicional para refinar el modelo.</p>


<h3>Resultados de la evaluación</h3>

<p>Se utilizaron seis conjuntos de datos diferentes para evaluar el modelo. Se consideraron las métricas: MSE, PSNR, SSIM y DISTS. Y la misma evaluación se realizó con el modelo ESRGAN para tener un punto de comparación entre ambos modelos.</p>

<p>Se consideraron seis conjuntos de datos diferentes que normalmente se usan en las pruebas de modelos de muestreo superior, como: Set5, Set14, history, BSD100, Urban100, Manga109. El conjunto de datos “histórico” de solo diez imágenes fue reemplazado por un conjunto de datos más grande de fotos históricas antiguas de la ciudad de La Paz (Bolivia) que se denominó Históricas40. Las imágenes se redujeron en un factor de 4 veces su tamaño original.</p>

<p>La siguiente tabla muestra los resultados de comparar las imágenes originales con sus versiones optimizadas.</p>

<table>
<tr>
	<th>Data set</th>
	<th colspan="2">MSE (óptimo ideal 0)</th>
	<th colspan="2">PSNR (óptimo ideal, más alto mejor)</th>
	<th colspan="2">SSIM (óptimo ideal 1)</th>
	<th colspan="2">DISTS (óptimo ideal 0)</th>
</tr>
<tr>
	<td></td>
	<td>optimized ESRGAN</td>
	<td>optimized RODIUCAN</td>
	<td>optimized ESRGAN</td>
	<td>optimized RODIUCAN</td>
	<td>optimized ESRGAN</td>
	<td>optimized RODIUCAN</td>
	<td>optimized ESRGAN</td>
	<td>optimized RODIUCAN</td>
</tr>
<tr>
	<td>BSDS100</td><td>0.0072</td><td>0.0068</td><td>22.174</td><td>22.390</td><td>0.8032</td><td>0.8272</td><td>0.1059</td><td>0.1684</td>
</tr>
<tr>
	<td>Historicas40</td><td>0.0099</td><td>0.0086</td><td>21.180</td><td>21.812</td><td>0.8441</td><td>0.8586</td><td>0.1142</td><td>0.1386</td>
</tr>
<tr>
	<td>Manga109</td><td>0.0067</td><td>0.0073</td><td>22.585</td><td>22.055</td><td>0.8983</td><td>0.8911</td><td>0.0763</td><td>0.1118</td>
</tr>
<tr>
	<td>Set5</td><td>0.0029</td><td>0.0021</td><td>26.405</td><td>27.489</td><td>0.9472</td><td>0.9592</td><td>0.0592</td><td>0.0402</td>
</tr>
<tr>
	<td>Set14</td><td>0.0068</td><td>0.0066</td><td>22.793</td><td>22.808</td><td>0.8528</td><td>0.8721</td><td>0.0994</td><td>0.1287</td>
</tr>
<tr>
	<td>Urban100</td><td>0.0103</td><td>0.0092</td><td>20.858</td><td>21.399</td><td>0.8216</td><td>0.8413</td><td>0.1011</td><td>0.1618</td>
</tr>
</table>

<h3>Conclusiones</h3>
<p>El modelo propuesto al usar las imágenes generadas por el autómata es lo suficientemente eficiente como para poder ser entrenado con un conjunto reducido de datos de 100 imágenes y es factible su entrenamiento sin aceleración GPU. Dicho entrenamiento sin GPU duro 41 horas, tuvo un consumo eléctrico de 12.18 kilo-vatios y un equivalente de 5.26 kilogramos de emisiones de dióxido de carbono.</p>
<p>En las evaluaciones se verifico que el modelo propuesto, optimiza las imágenes de baja calidad principalmente respecto a la reconstrucción de bordes, y en general tiene un grado de mejora aproximado al modelo ESRGAN y superior en cuatro de los seis conjuntos evaluados. Esta mejora se hace evidente con las métricas MSE, PSNR y SSIM, pero no con la métrica DISTS.</p>
