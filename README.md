## Simple Flask API to generate Semantic Embedding Vectors

To download new articles, you will need to set up the embedding API that will encode the new articles coming in to Zeeguu. This allows them to be inferred into topic categories.

To do this create a new docker image by cloning the `semantic-emb-api` and running the command:

```
docker build -f Dockerfile -t zeeguu_api_sem_emb .
```

This will install the image which is part of the dependencies of the `dev_server` docker-compose in the zeeguu/api repository

With this you can run:

```
docker-compose up dev_init_es
```
