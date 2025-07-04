# (21) Classification automatique des globules blancs à partir d’images microscopiques

Il faut exécuter le notebook_final pour voir le projet

## Description du projet

Nous avons à disposition un jeu de données comportant des images classifiées de globules blancs (disponibles dans le dossier barcelona de notre git). Celui-ci a déjà été pré-traité par Damien. Ces données sont sous-divisées en 3 jeux : entraînement, validation et test. 
  
__Notre problème est le suivant. On doit, à partir de ces données, entraîner et tester un modèle qui classe une image de globule blanc parmi 8 catégories différentes de globules blancs__.
  
Pour ce faire, on a utilisé un CNN (_Convolutional Neural Network_). Damien nous a guidé vers ce type de modèle, très adapté et surtout très classique pour la reconnaissance d’images (_computer vision_). 
  
Schématiquement, cela fonctionne comme un réseau de neurones “classique”. Simplement, les différents poids qui sont appris durant l'entraînement sont ici des noyaux de convolution (matrices 2 x 2).
  
Nous avons dû, au début, prendre en main les librairies de réseaux de neurones `PyTorch` et `Lightning` pour mieux comprendre la manière dont est réalisé l’apprentissage.
  
Pour l'apprentissage, nous avons utilisé plusieurs méthodes. Une méthode “from scratch”, où l'on définit nous même les différentes couches du CNN, et une autre où l'on utilise des modèles disponibles sur internet et déjà entraînés sur d’autres types d’images (_transfer learning_). 

## Choix techniques

On a décidé d’utiliser différentes méthodes pour réaliser notre projet afin de pouvoir comparer les performances de ces modèles. Nous avons utilisé des algorithmes de _deep-learning_ utilisant des réseaux de convolution.
  
Le premier modèle à avoir été implémenté est celui qu’on a réalisé “from scratch”. On définit toutes nos couches manuellement en utilisant la bibliothèque `PyTorch` puis on entraîne notre modèle sur notre jeu d’entraînement pendant un certain nombre d'époques.  On calcule ensuite un score (on a choisi l’_accuracy_ pour comparer nos différents modèles) sur le jeu de test.
  
On a ensuite utilisé des modèles de classification déjà entraînés sur d’autres jeux d'images bien plus massifs (comme _ImageNet_) en utilisant la bibliothèque `Lightning`.
  
On adapte ces modèles à notre problème de classification à 8 classes en changeant la dernière couche du réseau de neurones (la couche complètement connectée). On entraîne alors une première fois notre modèle, puis on dégèle la dernière couche du réseau de convolution pour que le modèle colle plus à notre problème de classification. La première étape s’appelle _transfer learning_ et la deuxième étape est le _fine tuning_.

Nous avons eu recours à 3 modèles préexistants : _ResNet_, _AlexNet_ et _VGG16_, qui donnent chacun des résultats différents.

## Imports à faire

Pour l'entraînement, Damien nous a fourni un fichier pour charger les données dans le bon format pour `PyTorch`.

Ainsi, si vous souhaitez entraîner le modèle, il faudrait importer les 2 classes de datasets: `BarcelonaDataModule` et `BarcelonaDataSet`.
  
Pour le reste, on a besoin de différents packages de `PyTorch`: `torch.nn`, `torch.optim`, `torchvision.models`, `torchvision.transforms`.
On a également besoin de Lightning`.

Concernant la visualisation, on a utilisé `seaborn`, `matplotlib` mais aussi `sklearn` pour l’accuracy et la matrice de confusion.

## Interface utilisateur

Une application à été créée dans le dossier interface du repo git. Il s'agit d'une interface utilisant le modèle _VGG16_ que nous avons entraîné. Ce faisant, il permet de déterminer la catégorie du globule blanc sur une image, de __manière plus intuitive__ et __accessible en ligne__.

Pour exécuter le serveur, il faut avoir installé `Flask`
dans l'environnement d'exécution.

## Lien du git
https://github.com/vargoclapin/Globule_blanc
