# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy source code
COPY ./*py /app/
COPY ./notebooks/*.py /app/notebooks/

RUN apt-get update && apt-get install -y tree && apt-get clean

RUN tree /app
RUN ls -R /app

# Install dependencies
COPY requirements_combined.txt .

RUN pip install --no-cache-dir -r requirements_combined.txt


# Expose port
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
