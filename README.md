# ia-bdx-ts-project3
Avec le modèle persistant naïf et ARMA, nous avons une baseline comme référence pour des modèles plus sophistiqués...
Suite de l'introduction aux séries temporelles avec les techniques "modernes" venant du Machine Learning :
  * RNN/LSTM/GRU
  * Convolution 1D
  * Méthodes ensemblistes du style Random Forest/XGBoost
  * Modèles dit avec "attention" venant des problèmes seq2seq du NLP (transformers) 
  
 Rappel : le théorème No Free Lunch indique qu'aucun modèle n'est universellement le meilleur, d'où l'intérêt d'essayer diverses approches sur les mêmes données.

## Atelier
* Le livrable aura la forme d'un notebook Python qui fera l'objet d'une auto-évaluation croisée
* On pourra au choix utiliser Keras ou PyTorch
* A minima, il faudra avoir utilisé un RNN, un LSTM + une autre méthode de son choix sur le cas du sinus puis sur la série temporelle déjà choisie (refaire valider éventuellement par le formateur)
  * Partie 1
    * Produire une série suivant un sinus
    * Entraîner et évaluer un RNN simple sur cette série
    * Varier la méthode d'évaluation (par exemple Walk-forward à fenêtre fixe ou à fenêtre croissante)
    * Ajouter différents niveaux de bruit et re-tester le modèle précédent
    * Ré-entraîner, tester
    * Refaire ce qui précède avec un LSTM et comparer
    * Refaire ce qui précède avec un GRU et comparer
  * Partie 2
    * Tout refaire avec la série choisie la semaine précédente (éventuellement prendre une série plus longue pour les LSTM/GRU si moins de 1000 exemples. Les données de consommation électrique sont une bon matériau)
    * Tout refaire avec une méthode ensembliste et comparer
    * Tout refaire avec une conv-1D et comparer
    * BONUS : remplacer le LSTM par un modèle à attention
    

## Mise en commun

## Récréation
* La FFT dans Python et son usage  (si pas le temps, se reporter [ici](https://www.ritchievink.com/blog/2017/04/23/understanding-the-fourier-transform-by-example/))
* Pour se détendre, lire les articles dans la rubrique Ressources/Les architectures au-delà des LSTM

## Méthodes plus avancées
* Modèles à  seq2seq https://jeddy92.github.io/JEddy92.github.io/ts_seq2seq_intro/

## Ressources

* RNN simple sur un sinus avec Keras : [ici](https://fairyonice.github.io/Understand-Keras%27s-RNN-behind-the-scenes-with-a-sin-wave-example.html)
* RNN simple sur un sinus avec TensorFlow : [ici](https://medium.com/@jkim2718/time-series-prediction-using-rnn-in-tensorflow-738e2dcfca96)
* Notebook de Guillaume [ici](https://github.com/guitoo/TimeSeries)

* RNN vs GRU vs LSTM : [ici](http://dprogrammer.org/rnn-lstm-gru)

* Des explications sur les LSTM : [ici](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

* LSTM sur un sinus : 
[ici](https://medium.com/@krzysztofbalka/training-keras-lstm-to-generate-sine-function-2e3c0ca42c3b)
[Une discussion à lire](https://datascience.stackexchange.com/questions/31923/training-an-lstm-to-track-sine-waves)

* LSTM sur série univarié [ici](https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/)

* Deep learning sur série univarié : [ici](https://machinelearningmastery.com/how-to-develop-deep-learning-models-for-univariate-time-series-forecasting/)

* Un diagramme très complet sur la forme des entrée pour un LSTM dans Keras : https://github.com/MohammadFneish7/Keras_LSTM_Diagram

* Une analyse de série temporelle très complète d'un point de vue statistique sur un cas réel avec LSTM : [ici](https://towardsdatascience.com/time-series-analysis-visualization-forecasting-with-lstm-77a905180eba)
Avec le [notebook](https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/LSTM%20Time%20Series%20Power%20Consumption.ipynb)
Tiré de [ça](https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/)

* Que faire si val loss décroit plus vite loss : [ici](https://www.pyimagesearch.com/2019/10/14/why-is-my-validation-loss-lower-than-my-training-loss/)

* ARIMA vs LSTM (vs Prophet): [ici](https://medium.com/@cdabakoglu/time-series-forecasting-arima-lstm-prophet-with-python-e73a750a9887)

* Time series generator dans Keras : [ici](https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/) [et là](https://www.dlology.com/blog/how-to-use-keras-timeseriesgenerator-for-time-series-data/)

* Changer la fréquence d'une série temporelle avec Pandas :
http://benalexkeen.com/resampling-time-series-data-with-pandas/

* LSTM et Time Series avec PyTorch : [ici](https://www.jessicayung.com/lstms-for-time-series-in-pytorch/)
[là](https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/)

* Tuto PyTorch : [ici](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
[ici aussi](https://www.analyticsvidhya.com/blog/2019/09/introduction-to-pytorch-from-scratch/)

* Utiliser la convolution 1D pour les séries temporelles : [ici](https://machinelearningmastery.com/how-to-develop-convolutional-neural-networks-for-multi-step-time-series-forecasting/)
[là](https://towardsdatascience.com/how-to-use-convolutional-neural-networks-for-time-series-classification-56b1b0a07a57)
https://jeddy92.github.io/JEddy92.github.io/ts_seq2seq_conv/
https://fr.slideshare.net/PyData/1d-convolutional-neural-networks-for-time-series-modeling-nathan-janos-jeff-roach

* XGBoost pour les séries temporelles : [ici](https://towardsdatascience.com/using-gradient-boosting-for-time-series-prediction-tasks-600fac66a5fc)

* Un notebook Kaggle qui utilise XGBoost pour de la prédiction de séries temporelles : [ici](https://www.kaggle.com/furiousx7/xgboost-time-series)

* Une étude comparative sur github entre ARIMA, XGBOOST et RNN: [ici](https://github.com/Jenniferz28/Time-Series-ARIMA-XGBOOST-RNN)

* Les architectures au-delà des LSTM : [ici](https://towardsdatascience.com/the-fall-of-rnn-lstm-2d1594c74ce0)
[ici aussi](https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3)
[là tiens](https://medium.com/@adityathiruvengadam/transformer-architecture-attention-is-all-you-need-aeccd9f50d09)

* Un article assez incroyable où l'auteur a quasiment essayé toutes les techniques de ML sur le problème de la prédiction des cours, très "enrichissant" à lire : [ici](https://towardsdatascience.com/aifortrading-2edd6fac689d)


* J'ajouterai des exemples de notebooks sur PyTorch, sur le modèle avec attention et sur le test de marche aléatoire avec vectorisation
  * Pytorch : https://github.com/JTreguer/pytorch_examples
  * Modèle avec attention : https://github.com/JTreguer/lstm_attention
  * Random-walk test : https://github.com/JTreguer/random_walk_test

## Grille d'évaluation mutuelle du livrable

* Pour la partie sur ARMA/ARIMA, pour chaque point 3 degrés ("peut mieux faire", "Job done!", "Wow!!!").
* Si "peut mieux faire", mettre les suggestions d'amélioration ou des propositions de code dans le notebook revu
* Points à vérifier :
    * Visualisation de la série temporelle de départ
    * Commentaires sur l'aspect de la série
    * Pré-traitements éventuels
    * Décomposition de la série temporelle (avec seasonal_decompose ou pas)
    * Vérification de la stationnarité des résidus par le test augmenté de Dickey-Fuller (+ autres méthodes empiriques éventuellement)
    * Visualisation des fonctions d'auto-corrélation (ACF/PACF)
    * Choix des ordres p et q du modèle ARMA/ARIMA
    * Fit du modèle
    * Evaluation du modèle
    * Prédictions en walk-forward
   
* Pour la partie méthodes de Machine Learning
* Mêmes recommandations de bienveillance et de critiques constructives
* Points à vérifier :
   * Mise en forme de la série sous forme (X,Y) 
   * RNN sur sinus avec courbe d'apprentissage et évaluation
   * LSTM sur sinus avec courbe d'apprentissage et évaluation
   * RNN sur données choisies avec courbe d'apprentissage et évaluation
   * LSTM sur données choisies avec courbe d'apprentissage et évaluation
   * En bonus : GRU, Conv-1D, XGBoost
   
## Récap time series

1. Visualiser la série temporelle
2. Se faire une idée sur sa dynamique (tendance ? Moyenne ? Cycles ? Extrema qui s'écartent ?)
3. Evaluer les baselines naïves
  * prédire avec la valeur courante
  * prédire avec la valeur moyenne
4. Préparer l'application d'un modèle classique style ARMA en décomposant la série temporelle en :
  * niveau moyen
  * tendance (seasonal_decompose ou appliquer diff ou soustraire la moyenne mobile qui va bien ou soustraire/diviser par un modèle de tendance linéaire, polynomial, exponentiel, etc.)
  * cycles / saisons (seasonal_decompose ou faire la différence à n intervalles où n = période ou soustraire un sinus fitté ou utiliser une FFT pour identifier les principales périodes)
5. Vérifier que ce qu'il reste (les résidus) est bien stationnaire :
  * Tracer la moyenne mobile et l'écart-type mobile : sont-ils plats ?
  * Calculer la moyenne et l'écart-type sur différents intervalles de taille au moins 30 unités : sont-ils à peu près constants ?
  * Appliquer un test statistique de type Dickey-Fuller augmenté (on veut la statistique bien négative et p-value < 0.05)
6. Tracer les fonctions d'auto-corrélation (ACF/PACF) et en déduire les ordres optimaux du modèle ARMA
7. Fitter le modèle ARMA
8. Valider le modèle ARMA en walk-forward
9. Si assez de données, essayer un LSTM en faisant varier son architecture et la fenêtre temporelle. Attention à bien mettre les données sous forme (X,Y) où X = les n valeurs précédentes. Penser à TimeSeriesGenerator dans Keras s'il y a bcp de données.
10. Essayer avec XGBoost
11. Essayer la convolution 1-D
12. Essayer un réseau de neurones classique en lui passant l'ensemble (X,Y) de training
13. Essayer des techniques plus avancées provenant du NLP comme le modèle à attention du style Transformers
14. Comparer ses modèles à des outils disponibles en ligne comme BigML ou Prophet
  
