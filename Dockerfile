FROM python:3.10

COPY . /app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install PyYAML explicitly
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir setuptools==58.0.4
# RUN pip install --no-cache-dir --only-binary :all: PyYAML==6.0

# Upgrade pip
RUN pip install --upgrade pip

# Install all dependencies
RUN pip --no-cache-dir install -r requirements.txt

# Expose port 80
EXPOSE 80

# Run the Streamlit app
ENTRYPOINT ["streamlit", "run", "--server.port=80", "--server.address=0.0.0.0"]
CMD ["src/main.py"]

