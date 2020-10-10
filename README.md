# MarsImagesAnalysis

Python Dependencies: numpy, keras, tensorflow, pil, matplotlib

- Objective 1 :  Find subjective quality for images in the *qualitydata* folder
- Objective 2 :  Classify images in the *classdata* folder
- Objective 3 :  3D projections for images in the *2D* folder

Run:
>python EvaluateQuality.py
Using TensorFlow backend.

# Once the code executed you will see the following menu:
------------------------------ MENU ------------------------------
1. Option 1 : Image aesthetic quality
2. Option 2 : Image classification
3. Option 3 : 2D to 3D approximate projection
4. Exit
-------------------------------------------------------------------
Enter your choice [1-4]: 1
Menu 1 (Image aesthetic quality) has been selected
Enter a threshold between 1 and 10. suggested - 5:
5

Finding Image aesthetic quality ...  ... with threshold : 5.0
... ...
1/1 [==============================] - 2s 2s/step
1/1 [==============================] - 1s 1s/step
1/1 [==============================] - 1s 1s/step
:
1/1 [==============================] - 1s 1s/step
1/1 [==============================] - 1s 1s/step
Number of images above threshold 60
Number of images below threshold 10
~~~ Finished writing to the file named imagesquality.csv ~~~
------------------------------ MENU ------------------------------
1. Option 1 : Image aesthetic quality
2. Option 2 : Image classification
3. Option 3 : 2D to 3D approximate projection
4. Exit
-------------------------------------------------------------------
Enter your choice [1-4]: 2
Menu 2 (Image classification) has been selected
 Classifying Images ...  ...
... ...
... ...
aeo_tp : 13  aeo_tn : 37  aeo_fp : 5  aeo_fn : 2
dry_tp : 15  dry_tn : 35  dry_fp : 2  dry_fn : 0
gla_tp : 11  gla_tn : 39  gla_fp : 3  gla_fn : 4
vol_tp : 11  vol_tn : 39  vol_fp : 0  vol_fn : 4
Precision for Aeolian  : 0.86    Recall for Aeolian : 0.72
Precision for Dry  : 1.00        Recall for Dry : 0.88
Precision for Glacial  : 0.73    Recall for Glacial : 0.78
Precision for Volcanic  : 0.73   Recall for Volcanic : 1.00
... ...
~~~ Finished writing to the file named imagesclassification.csv ~~~
------------------------------ MENU ------------------------------
1. Option 1 : Image aesthetic quality
2. Option 2 : Image classification
3. Option 3 : 2D to 3D approximate projection
4. Exit
-------------------------------------------------------------------
Enter your choice [1-4]: 3
Menu 3 (2D to 3D projection) has been selected
 Converting Images ...  ...
... ...
Working on 2D/test1.jpg- 1
Working on 2D/test2.jpg- 2
Working on 2D/test3.jpg- 3
Working on 2D/test4.jpg- 4
------------------------------ MENU ------------------------------
1. Option 1 : Image aesthetic quality
2. Option 2 : Image classification
3. Option 3 : 2D to 3D approximate projection
4. Exit
-------------------------------------------------------------------
Enter your choice [1-4]:

