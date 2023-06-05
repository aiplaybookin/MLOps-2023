FROM python:3.9-slim-bullseye

WORKDIR /workspace

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt \
    && rm -rf /root/.cache/pip

COPY train.py /workspace/

CMD ["python3", "train.py"]