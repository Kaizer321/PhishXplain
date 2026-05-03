FROM python:3.10-slim

# Create a non-root user (Hugging Face requirement)
RUN useradd -m -u 1000 user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy the requirements file and install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the application files
COPY --chown=user . .

# Hugging Face exposes port 7860
EXPOSE 7860

# Start server on 7860
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
