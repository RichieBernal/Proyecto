FROM python:3.11-slim

WORKDIR / 
COPY . .
RUN pip3 install --no-cache-dir -r requirements.txt 
RUN apt-get update && apt-get install -y vim 

EXPOSE 8000

# Add the volume configuration to copy logs to the local hard drive
VOLUME /logs

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" , "--reload"]