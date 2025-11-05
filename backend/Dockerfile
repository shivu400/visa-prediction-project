
FROM python:3.11-slim


WORKDIR /code


COPY ./requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


COPY ./app /code/app
COPY ./models /code/models
COPY ./uploads /code/uploads
COPY ./tests /code/tests


EXPOSE 8000


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]