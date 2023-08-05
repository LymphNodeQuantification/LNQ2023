FROM python:3.9-slim

ENV PYTHONUNBUFFERED 1

WORKDIR /tmp

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

COPY --chown=user:user ground_truth /tmp/ground_truth

COPY --chown=user:user requirements.txt /tmp

RUN python3 -m pip install -rrequirements.txt

COPY --chown=user:user evaluation.py /tmp

ENTRYPOINT ["python", "evaluation.py"]
