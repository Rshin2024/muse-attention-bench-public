ARG BASE_IMAGE=lmsysorg/sglang@sha256:a0ae9cefecc0fb68e74a6f3b7f338eb80a53e3e6df4637ec2f024abc5fec07af
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
LABEL org.opencontainers.image.title="Muse attention benchmark" \
      org.opencontainers.image.base.name="${BASE_IMAGE}" \
      org.opencontainers.image.description="Local reconstruction of the public competition protocol; not the hosted judge"
ENV PYTHONPATH=/opt/muse-attention-bench/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /opt/muse-attention-bench
COPY src ./src
COPY configs ./configs
COPY third_party ./third_party
ENTRYPOINT ["python3", "-m", "muse_attention_bench"]
