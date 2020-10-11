<h1>MarsImagesAnalysis</h1>
<p><strong>Python Dependencies:</strong> numpy, keras, tensorflow, pil, matplotlib</p>
<ul>
<li>Objective 1: Find subjective quality for images in the&nbsp;<em>qualitydata</em>&nbsp;folder</li>
<li>Objective 2: Classify images in the&nbsp;<em>classdata</em>&nbsp;folder</li>
<li>Objective 3: 3D projections for images in the&nbsp;<em>2D</em>&nbsp;folder</li>
</ul>
<p><strong>Run the following command:</strong></p>
<ul>
<li>&gt; python EvaluateQuality.py</li>
</ul>
<p><strong>Once the code executed you will see the following menu:</strong></p>
<p>------------------------------ MENU ------------------------------</p>
<ol>
<li>Option 1: Image aesthetic quality</li>
<li>Option 2: Image classification</li>
<li>Option 3: 2D to 3D approximate projection</li>
<li>Exit</li>
</ol>
<p><strong>For demonstration purposes, please select the number 1 to determine the quality of the images in your dataset:</strong></p>
<p>Enter your choice [1-4]: 1</p>
<p>Menu 1 (Image aesthetic quality) has been selected Enter a threshold between 1 and 10. suggested - 5: 5</p>
<ul>
<li>Finding Image aesthetic quality ... ... with threshold: 5.0 ... ... 1/1 
<li>  [==============================] - 2s 2s/step 1/1</li>
<li>[==============================] - 1s 1s/step 1/1</li>
<li>:</li>
<li>[==============================] - 1s 1s/step 1/1</li>
<li>[==============================] - 1s 1s/step 1/1</li>
<li>[==============================] - 1s 1s/step</li>
</ul>
<p>Number of images above threshold 60 Number of images below threshold 10</p>
<p><strong>Once the images with acceptable quality are identified, now it is the time to select option 2 which will separate the images automatically based on various environmental conditions:</strong></p>
<p>------------------------------ MENU ------------------------------</p>
<ol>
<li>Option 1: Image aesthetic quality</li>
<li>Option 2: Image classification</li>
<li>Option 3: 2D to 3D approximate projection</li>
<li>Exit</li>
</ol>
<p>-------------------------------------------------------------------</p>
<p>Enter your choice [1-4]: 2</p>
<p>Menu 2 (Image classification) has been selected</p>
<p>&nbsp;Classifying Images ...&nbsp; ...</p>
<p>... ...</p>
<ul>
<li>Precision for Aeolian: 0.86&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Recall for Aeolian: 0.72</li>
<li>Precision for Dry: 1.00&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Recall for Dry: 0.88</li>
<li>Precision for Glacial: 0.73&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Recall for Glacial: 0.78</li>
<li>Precision for Volcanic: 0.73&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Recall for Volcanic: 1.00</li>
</ul>
<p>... ...<br /> <strong>After classifying images into environmental conditions classes, the researcher could also convert these images into 3D images for better exploration by selecting option 3:</strong><br /> </p>
<p>~~~ Finished writing to the file named imagesclassification.csv ~~~</p>
<p>------------------------------ MENU ------------------------------</p>
<ol>
<li>Option 1 : Image aesthetic quality</li>
<li>Option 2 : Image classification</li>
<li>Option 3 : 2D to 3D approximate projection</li>
<li>Exit</li>
</ol>
<p>-------------------------------------------------------------------</p>
<p>Enter your choice [1-4]: 3</p>
<p>Menu 3 (2D to 3D projection) has been selected</p>
<p>&nbsp;Converting Images ...&nbsp; ...</p>
<p>... ...</p>
<p><strong>Please check the samples of D2 and 3D images in the corresponding folders.<br /> </strong></p>
<p>------------------------------ MENU ------------------------------</p>
<ol>
<li>Option 1: Image aesthetic quality</li>
<li>Option 2: Image classification</li>
<li>Option 3: 2D to 3D approximate projection</li>
<li>Exit</li>
</ol>
<p>-------------------------------------------------------------------</p>
<p>Enter your choice [1-4]: 4 <strong>(to exit the demo)</strong></p>
