# Traffic Flow Prediction

This is the final project of my course in Computer Science. The theme was traffic flow prediction and the result is this monography.

## Abstract

Traffic flow prediction is a complex task. There are many challenging factors that must be taken into consideration, like flow oscillation, accidents and the local weather (random factors). The literature focus on highways and freeways, which doesn't have obstructions. This monography focus on road crossings, where there are traffic lights and obstructions, what makes the prediction even more complex. That means this work focus in verifying if the models, usually applied in freeways, can be sucessfully used in crossings too. To calculate the flow prediction in this scenario, four machine learning techniques were tested. Two of them use deep learning, **LSTM** and **GRU** and two of them use a more traditional approach, **SVM** and **RF**. The results of the experiment were promising, because all of the models had better performance than the proposed baselines. It was also shown that the traditional models stood out against the deep learning techniques, in which the **SVM** had the best results for all the predicitons (15, 30, 45 and 60 minutes in future).

## Structure of Repository

This project has 4 parts:

+ **Monograph** is the folder with the latex files that made the monograph;
+ **Presentation** is the folder with the latex files that made the presentation of this monograph;
+ **Project** is the folder with the code and data used;
+ **WPOS** is the folder with the latex files of the early presentation of this monograph in the WPOS event;
