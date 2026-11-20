FROM python:3.12-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir "mcp>=2.3,<3" "httpx>=0.27" "pydantic>=2.7" "boto3>=1.34"
ENV MCP_TRANSPORT=streamable-http \
    MCPFORGE_POLICY=read_write
EXPOSE 8000
USER 1000
CMD ["python", "server.py"]
