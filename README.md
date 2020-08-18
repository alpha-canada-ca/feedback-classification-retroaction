(La version française suit la version anglaise)

# Tagging web feedback automatically by training a classification algorithm - Supervised training

These scripts were developed to help auto-tag web feedback gathered on Canada.ca pages.

There are 2 scripts:
- **process_feedback.py:** gets the data from AirTable, pre-processes the data, trains classification algorithms based on already tagged feedback (where tags have been confirmed by a human), and saves the necessary files in the data folder. Allows for training several models, depending on the topic of the page the feedback was gathered from.
- **suggest.py:** gets the feedback text to classify, gets the right files (the right model to use and its associated vectorizer and list of possible tags)

## How it works
The training and prediction happens in several phases.

### Data importing, splitting and pre-processing
- Data gets imported from a source (AirTable, CSV file, etc.), and is split between English and French
- Only samples that have a model assigned to them and confirmed tags are kept
- Sample are separated in different models
- Text is preprocessed to improve training accuracy:
  - special characters are removed
  - text is changed to all lower case
  - text is stemmed, so only the roots are kept (e.g.: swim and swimming are considered the same words)

### Preparing the data for a classification algorithm
- Classification algorithms work with numbers, not words, so some preparation is needed
- Possible tags are turned into a sparse matrix: each tag is possible class, and each sample gets a 1 for tags that are assigned to it (and 0 for tags not assigned to it)
- Feedback text is changed into a matrix, where all words in the corpus are a possible attribute. Using a Term Frequency - Inverse document frequency (TF-IDF) vectorizer, each sample is turned into a vector, with a number assigned to the words (attributes) contained in the sample

### Train the algorithms
- Now that the sample text and the possible classes are numbers, a classification algorithm can be trained
- Several options are possible, but this script uses a simple multinomial Naive Bayes classification algorithm
- For each model, a separate algorithm is trained for each possible tag, calculating the probability for each class
- The model, the vectorizer data and the possible tags for each model is saved in a pickle file.

### Get probabilities for each tag
- The prediction script launches a local web app
- The script looks for several attributes (language, model and text to classify), that are passed in the local URL
- The proper model, vectorizer and list of possible tags are downloaded, and a probability is calculated for each possible tag
- The script returns the tag with the highest probability, plus any other tag over a certain threshold
- The script in this repo works locally (localhost), but can be used to generate predictions on a live server


## Progressive improvement: re-training algorithm periodically

The Canada.ca feedback implementation records new feedback into an AirTable, and tags it automatically (if there's an existing model assigned to it).

Subject matter experts go over the new feedback, adjust the tag(s) if it's needed, and confirms the tag when they feel confident.

Every day, the prediction algorithm gets retrained with all confirmed tags. In effect, the prediction algorithm should get better with time, as more confirmed tags are used for training the algorithm.

***

#Classer la rétroaction Web automatiquement par l'entraînement d'un algorithme de classification - Entraînement supervisé

Ces scripts ont été développés pour aider à étiqueter automatiquement les commentaires recueillis sur les pages de Canada.ca.

Il y a deux scripts :
- **process_feedback.py:** ce script les données de AirTable, traite les données, entraîne des algorithmes de classification basés sur les commentaires déjà étiquetés (lorsque les étiquettes ont été confirmées par un humain), et enregistre les fichiers nécessaires dans le dossier appelé "data". Il permet d'entraîner plusieurs modèles, selon le sujet de la page d'où provient la rétroaction.
- **suggest.py:** ce script obient le texte de la rétroaction à classer, récupère les bons fichiers (le bon modèle à utiliser et son vectorisateur associé, ainsi que la liste d'étiquettes possibles)

## Comment ça marche
L'entraînement et la prédiction se déroulent en plusieurs phases.

### Importation, séparation et traitement des données
- Les données sont importées d'une source (AirTable, fichier CSV, etc.), et sont séparées entre l'anglais et le français
- Seuls les échantillons auxquels un modèle a été attribué et dont les étiquettes ont été confirmées sont conservés
- Les échantillons sont séparés selon différents modèles
- Le texte est traité pour en améliorer la précision de l'entraînement:
  - les caractères spéciaux sont supprimés
  - le texte est modifié pour être en lettres minuscules
  - le texte est tronqué afin que seules les racines soient conservées (par exemple : nage et nager sont considérés comme le mêmes mot)

### Préparer les données pour un algorithme de classification
- Les algorithmes de classification fonctionnent avec des chiffres et des nombres, et non avec des mots, ce qui nécessite une certaine préparation
- Les étiquettes possibles sont transformées en une matrice clairsemée : chaque étiquette est une classe possible, et chaque échantillon obtient un 1 pour les étiquettes qui lui sont attribuées (et un 0 pour les étiquettes qui ne lui sont pas attribuées)
- Le texte de la rétroaction est transformé en une matrice, où tous les mots du corpus sont un attribut possible. À l'aide d'un vectorisateur de fréquence de terme - Fréquence inverse du document (TF-IDF), chaque échantillon est transformé en un vecteur, avec un chiffre attribué aux mots (attributs) contenus dans l'échantillon

### Entraîner les algorithmes
- Maintenant que le texte modèle et les classes possibles sont des nombres, un algorithme de classification peut être entraîné
- Plusieurs options sont possibles, mais ce script utilise un simple algorithme multinomial de classification naïve de Bayes
- Pour chaque modèle, un algorithme distinct est formé pour chaque étiquette possible, en calculant la probabilité pour chaque classe
- Le modèle, les données vectorielles et les étiquettes possibles pour chaque modèle sont enregistrés dans un fichier pickle.

### Obtenir les probabilités pour chaque étiquette
- Le script de prédiction lance une application web locale
- Le script obtient plusieurs attributs (langue, modèle et texte à classer), qui sont passés dans l'URL locale
- Le modèle approprié, le vectorisateur et la liste des étiquettes possibles sont téléchargés, et une probabilité est calculée pour chaque étiquette possible
- Le script renvoie l'étiquette ayant la plus grande probabilité, ainsi que toute autre étiquette dépassant un certain seuil
- Le script de ce référentiel fonctionne localement (localhost), mais peut être utilisé pour générer des prédictions sur un serveur en direct


## Amélioration progressive : ré-entraînement périodique de l'algorithme

La mise en œuvre de la rétroaction de Canada.ca enregistre les nouvelles réactions dans un tableau Airtable, et attribue une étiquette automatiquement (si un modèle existant lui a été attribué).

Des experts en la matière passent en revue les nouveaux commentaires, ajustent les étiquettes si nécessaire et les confirment lorsqu'ils sont satisfaits de l'étiquetage.

Chaque jour, l'algorithme de prédiction est ré-entraîné avec toutes les étiquettes confirmées. Ainsi, l'algorithme de prédiction devrait s'améliorer avec le temps, car un plus grand nombre d'étiquettes confirmées sont utilisées pour l'entraînement de l'algorithme.
