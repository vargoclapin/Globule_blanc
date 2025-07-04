Classification automatique des globules blancs à partir d’images microscopiques

a) Description du projet

On a à disposition un jeu de données déjà bien pré-traité par Damien. Ce jeu était divisé en 3 jeux: entraînement, validation et test. Notre problème est le suivant: On doit , à partir d’une image microscopique, entraîner et tester un modèle qui classe l’image parmi 8 catégories de globules blancs différents. 
Pour se faire, on a utilisé un CNN (Convolutional Neural Networks). Damien nous a guidé vers ce type de modèle, très adapté et surtout très classique pour la reconnaissance d’images. Schématiquement, cela fonctionne comme un réseau de neurones “classiques”, simplement les différents poids qui sont appris durant l'entraînement sont en réalité des noyaux de convolution qui sont appliqués aux images.
Ainsi, nous avons donc , au début, essayé de prendre en main les librairies pyTorch et Lightning pour mieux comprendre la manière dont est réalisé l’apprentissage.
Ensuite, nous avons utilisé plusieurs méthodes. Une méthode “from scratch”, où on définit nous même les différentes couches du CNN, et une autre où on utilise des modèles disponibles sur internet et déjà entraînés mais sur d’autres types d’images (transfer learning). 

 b) Choix techniques

On a décidé d’utiliser différentes méthodes pour réaliser notre projet afin de pouvoir comparer les performances de ces modèles. On a utilisé des algorithmes de deep-learning utilisant des réseaux de convolution. Le premier modèle à avoir été implémenté est celui qu’on a réalisé “from scratch”. On définit toutes nos couches manuellement en utilisant la bibliothèque pytorch puis on entraîne notre modèle sur notre jeu d’entraînement pendant un certain nombre d'époques.  On calcule ensuite un score ( on a choisi l’accuracy pour comparer nos différents modèles) sur le jeu de test.
On a ensuite utilisé des modèles de classification déjà entraînés sur d’autres jeux d'entraînement bien plus massif en utilisant la bibliothèque lightning. On les adapte ensuite à  notre problème de classification à 8 classes en changeant la dernière couche du réseau de neurones (La couche complètement connectée). On entraîne alors une première fois notre modèle, puis on dégèle la dernière couche du réseau de convolution pour que le modèle colle plus à notre problème de classification. (la première étape s’appelle transfer learning et la deuxième fine tuning).
On a utilisé 3 différents modèles : Resnet, Alexnet et VGG16 qui donnent chacun des résultats différents.

c) Imports à faire

Pour l'entraînement, Damien nous a déjà donné un fichier qui crée les bonnes classes pour bien charger les données dans le bon format pour torch. 
Ainsi, si vous souhaitez entraîner le modèle, il faudrait importer les 2 classes de datasets: BarcelonaDataModule et BarcelonaDataSet.

Pour le reste, on a besoin de différents outils de torch: nn, optim, models, transforms.
On a aussi besoin de lightning: pytorch_lightning.

Concernant la visualisation, on a utilisé seaborn et matplotlib, et aussi sklearn pour l’accuracy et la matrice de confusion.


d) Lien du commit
****
