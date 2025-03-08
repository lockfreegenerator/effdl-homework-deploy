FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose ports for HTTP, metrics, and gRPC
EXPOSE 8080 9090

# Run the HTTP and gRPC services
ENTRYPOINT ["sh", "-c", "python http_server.py & python grpc_server.py"]
