# cti-text ripped out of Docker build from Contextual Topic Identification by Stveshawn
## Result

Visualizations (2D UMAP) of clustering results with different vectorization methods with `n_topic=10`

| TF-IDF | BERT | LDA_BERT |
|---|---|---|
![Model](https://github.com/Stveshawn/contextual_topic_identification/raw/master/docs/images/tfidf.png) | ![Model](https://github.com/Stveshawn/contextual_topic_identification/raw/master/docs/images/bert.png) | ![Model](https://github.com/Stveshawn/contextual_topic_identification/raw/master/docs/images/lda_bert.png)|


Evaluation of different topic identification models with `n_topic=10`


| Metric\Method | TF-IDF + Clustering | LDA | BERT + Clustering | LDA_BERT + Clustering |
|---|---|---|---|---|
|C_Umass|__-2.161__|-5.233|-4.368|-3.394|
|CV|0.538|0.482|0.547|__0.551__|
|Silhouette score|0.025|/|0.063|__0.234__|
