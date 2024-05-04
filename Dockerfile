# Use an official Python runtime as a parent image
FROM python:3.11-slim-buster

# Set the working directory in the container
WORKDIR /usr/src/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc libffi-dev g++ wget make
RUN apt-get update && apt-get install -y sqlite3
RUN sqlite3 --version

# install version od sqlite 3.35 that is compatible with chroma
RUN wget https://www.sqlite.org/2021/sqlite-autoconf-3350000.tar.gz \
    && tar xzf sqlite-autoconf-3350000.tar.gz \
    && cd sqlite-autoconf-3350000 \
    && ./configure \
    && make \
    && make install \
    && ldconfig


# Install Poetry
RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN pip install --no-use-pep517 pypika==0.48.9
RUN pip install --upgrade pip setuptools wheel


# Copy the current directory contents into the container at /usr/src/app
COPY . /usr/src/app

ENV POETRY_HTTP_TIMEOUT=600
ENV PEP517_BUILD_BACKEND=1
ENV FLASK_APP=front_app/app.py

# Install application dependencies
RUN poetry install --no-dev

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["flask", "run", "--host=0.0.0.0"]
