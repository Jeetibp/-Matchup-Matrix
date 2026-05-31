FROM python:3.11.9-slim

WORKDIR /app

# Create non-root user (required by HuggingFace Spaces)
RUN useradd -m -u 1000 appuser

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Give ownership to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# HuggingFace Spaces requires port 7860
EXPOSE 7860

# Run with gunicorn on port 7860
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "--preload"]
