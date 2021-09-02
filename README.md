# Covid Map Hanoi using Machine Learning
My on-going project, implementing Recurrent Neural Network for Named Entity Recognition task on Covid newspaper.

The project use BERT-multilingual, trained specifically on ~ 200 covid newspaper with tags following BILOU tags format (CoNLL-U 2003)

The model will read and extract location info, along with other infos like patient ID (if have) and date of publish.

After extraction, an automated Python script will run to add the new information into MySQL Database.

Here's a demo of the map on web:

![Covid Map Hanoi](demo/covid_map.gif)
