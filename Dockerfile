FROM python:3.10.14-bookworm

WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY * ./
CMD ["python", "beta_emulator"]