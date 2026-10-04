FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
ENV HYDRA_MODE=server HYDRA_DATA_DIR=/data
VOLUME /data
EXPOSE 8765
CMD ["hydra-sim", "--mode", "server", "--no-browser"]
