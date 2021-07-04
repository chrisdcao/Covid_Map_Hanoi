# Covid Map Hanoi
My on-going project, implementing Recurrent Neural Network for Named Entity Recognition task on Covid newspaper.

The project use BERT-multilingual, trained specifically on ~ 200 covid newspaper with tags following BILOU tags format (CoNLL-U 2003)

The model will read and extract location info, along with other infos like patient ID (if have) and date of publish.

After extraction, an automated Python script will run to add the new information into MySQL Database.

Here's a demo of the map on web:

![Covid Map Hanoi](demo/covid_map.gif)

The model is stopping at accuracy of 89.61% evaluation loss at 4 training epochs. Any more epochs signals overfitting thus further conductions must wait for more data.

The model has not implemented Conditional Random Field, which might slightly increase the accuracy.
