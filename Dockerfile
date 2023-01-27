# Base Image
FROM python:3.9

# Set working directory
WORKDIR /captcha-app

COPY ./requirements.txt ./setup.py ./setup.cfg ./

RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir

COPY ./src ./src
COPY ./artifact ./artifact

RUN pip install -e .

EXPOSE 8000

CMD ["uvicorn", "src.fastapi_backend.main:app", "--host", "0.0.0.0", "--port", "8000"]





