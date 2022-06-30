# Self-SupervisedLearning-Onderzoek

De inhoud van de files is als volgt:

- `docker_image`, de dockerfile die de environmet beschrijft die nodig is voor het trainen van de modellen. Deze is vastgezet op de versie die toendertijd het nieuwste was. 
- `3D scans`, hier zijn we niet zo ver mee gekomen, memory issues speelden hier een rol. Mooie stap in dit notebook is de datagenerator die ervoor zorgt dat je als het goed is niet alle data in geheugen hoeft te hebben maar alleen alles van een bepaalde batch aan data. 
- `Aantal patches`, onderzoek naar invloed van het aantal patches. Hier vind je ook een goede versie van de business understanding. 
- `Scans inlezen en verwerken.ipynb` maken van joblib bestand op basis van de ruwe scans. 
- `Grote training 1/2/3`. 10.000 epochs trainen met goed model. Grote trainning 3 intropduceert ReLu, dit bleek een groot positief effect te hebben. Dit is het beste model (nr 3). 
- Split analyzer en reconstructuer. Hier staat hoe we de het analyzer (encoder) en de reconstructuer (decoder) van elkaar gesplitst hebben. Dit in voorbereiding dat de analyzer hergebruikt wordt in een andere trainingstaak. 
- `show layers` visualsiatie van de weights per laag in het model, [gebaseerd op dit github repo](https://github.com/gabrielpierobon/cnnshapes/blob/master/README.md). Een typisch patroon wat je terug ziet is steeds vierkantes met het ct scan beeld. 

- Triplets laten zien, definitie van de code om triplet van origineel, corrupt en reconstructed te laten zien. 
- Triplet video, om video te maken.  


# Opmerkingen
- Tensorboard draait standaard op 6006. Denk erom dat je ook in de docker container deze poort open moet hebben. 
- Scans worden typisch ingelezen vanuit een joblib bestand. Hoe je dit maakt kan je vinden in `Scans inlezen en verwerken.ipynb`. 
- NUMBER_OF_SWITCHES en PATCH_SIZE hebben een onderling verband. Als ze samen te groot zijn krijg je overfitting, als je de één klein maakt kun je de ander groter maken. 
