#Python version 
FROM python:3.11-slim

#Working directory is app
WORKDIR /app

#We require all dependencies from requirements files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#We require all the files from our projects
COPY . .

#storing the pdf files
RUN mkdir -p data/source_data/naive_rag

#Streamlit app at port 8501
EXPOSE 8501

#Default command to run
CMD ["streamlit", "run", "app.py", "--server.port = 8501", "--server.address=0.0.0.0"]