<h3>RODIUCAN</h3>

<p>The <strong>RODIUCAN</strong> model acronyms obtained from "RESOLUTION OPTIMIZATION OF DIGITAL IMAGES USING CELLULAR AUTOMATA AND ARTIFICIAL NEURAL NETWORKS".
It is a model of increased image resolution.</p>

<p>It is a thesis work which proposes a model for increasing image resolution using cellular automata and convolutional neural networks, considering a reduced data set to optimize the use of computational resources.</p>

<p>The training was carried out with the data set "General100" and with patches of 128x128 pixels.</p>

<h3>Model Description</h3>
<p>Cellular automata were used to detect the edges and backgrounds in the images, thus obtaining a simplified but very descriptive representation of the edges of objects in the images.</p>

<p>These images were then used as a transformation buffer between two convolutional neural networks to scale up the image resolution. The neural networks were trained separately and then combined using a scheme similar to a conditional generative network to refine the model.</p>


<h3>Evaluation results</h3>

<p>Six different data sets were used to evaluate the model. The metrics were considered: MSE, PSNR, SSIM and DISTS. And the same evaluation was carried out with the ESRGAN model to have a point of comparison between both models.</p>

<p>Six different datasets typically used in upsampling model testing were considered, such as: Set5, Set14, historical, BSD100, Urban100, Manga109. The “historical” dataset of only ten images was replaced by a larger dataset of old historical photos of the city of La Paz (Bolivia) that was named Historicas40. The images were reduced by a factor of 4 times their original size.</p>

<p>The following table shows the results of comparing the original images with their optimized versions.</p>

<table>
<tr>
	<th>Data set</th>
	<th colspan="2">MSE (optimum ideal 0)</th>
	<th colspan="2">PSNR (optimal ideal, higher better)</th>
	<th colspan="2">SSIM (optimum ideal 1)</th>
	<th colspan="2">DISTS (optimum ideal 0)</th>
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
	<td>BSDS100</td><td>0.0072</td><td>0.0070</td><td>22.174</td><td>22.272</td><td>0.8032</td><td>0.8250</td><td>0.1059</td><td>0.1666</td>
</tr>
<tr>
	<td>Historicas40</td><td>0.0099</td><td>0.0087</td><td>21.180</td><td>21.798</td><td>0.8441</td><td>0.8592</td><td>0.1142</td><td>0.1370</td>
</tr>
<tr>
	<td>Manga109</td><td>0.0067</td><td>0.0078</td><td>22.585</td><td>21.872</td><td>0.8983</td><td>0.8878</td><td>0.0763</td><td>0.1127</td>
</tr>
<tr>
	<td>Set5</td><td>0.0029</td><td>0.0023</td><td>26.405</td><td>27.234</td><td>0.9472</td><td>0.9579</td><td>0.0592</td><td>0.0381</td>
</tr>
<tr>
	<td>Set14</td><td>0.0068</td><td>0.0073</td><td>22.793</td><td>22.494</td><td>0.8528</td><td>0.8704</td><td>0.0994</td><td>0.1310</td>
</tr>
<tr>
	<td>Urban100</td><td>0.0103</td><td>0.0093</td><td>20.858</td><td>21.328</td><td>0.8216</td><td>0.8390</td><td>0.1011</td><td>0.1627</td>
</tr>
</table>
