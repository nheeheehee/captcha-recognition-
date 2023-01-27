#!/bin/bash

docker build -t captcha .
docker tag captcha nheeheehee/captcha:latest
docker push nheeheehee/captcha:latest
docker run -p 8000:8000 captcha:latest