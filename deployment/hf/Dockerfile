FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 PORT=7860 YOLO_DEVICE=cpu \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    GRADIO_ANALYTICS_ENABLED=False MPLCONFIGDIR=/home/user/app/project_work/matplotlib
WORKDIR /home/user/app
COPY --chown=user:user requirements*.txt ./
RUN pip install --no-cache-dir torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-gradio.txt
COPY --chown=user:user . .
RUN python -B deployment/hf/start.py --check
EXPOSE 7860
CMD ["python", "-B", "deployment/hf/start.py"]
