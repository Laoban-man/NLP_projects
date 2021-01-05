FROM ubuntu:latest
RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev build-essential
COPY . .
RUN pip3 install -r requirements.txt
RUN pip3 install lightgbm

# Launch app
EXPOSE 5000
CMD ["python3", "app.py"]
