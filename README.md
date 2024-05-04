# RAG Network Project

## Overview

This project introduces a web application built with Flask in a Docker environment, utilizing the RAG (Retrieval-Augmented Generation) mechanism to extract the most relevant articles from a dataset available at [this Kaggle link](https://www.kaggle.com/datasets/meruvulikith/1300-towards-datascience-medium-articles-dataset). The application is designed to display the retrieved documents to the user, facilitating easier testing and interaction. The core functionalities leverage the `langchain` library and open models from `HuggingFace` for embedding and enhanced document retrieval. All environmental dependencies are managed with `Poetry`.

## Requirements

- Tools:
  - Docker
  - Docker Compose
  - Poetry

## Setup

### Data Preparation

Ensure that the dataset from Kaggle is downloaded and placed in the main project directory:

```
./1300-towards-datascience-medium-articles-dataset/medium.csv
```

By default, this dataset folder, along with the dataset itself, is included in the GitHub project. The `RAG/chroma_db` directory will host files allowing for the persistence of the vector database state (Chroma). The GitHub codebase already contains embeddings for the articles. However, if the folder is emptied, the code will regenerate it (note that this will increase runtime as each article needs to be re-embedded).

### Environment Setup

Ensure all environmental dependencies are installed and managed via Poetry. Follow the steps below to set up your Docker environment which encapsulates all necessary dependencies and settings.

## Running the Application

To start the application, execute the following commands:

1. Initialize the Docker containers:
   ```
   docker-compose up
   ```
2. Access the application through your web browser by navigating to:
   ```
   http://localhost:5000
   ```

This setup ensures that the application is ready to use with minimal setup required from the end-user, providing a seamless experience for testing and use.
