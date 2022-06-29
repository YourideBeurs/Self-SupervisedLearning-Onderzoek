FROM tensorflow/tensorflow:2.9.1-gpu

RUN pip install \
  numpy==1.19.5 \
  matplotlib==3.3.4 \
  jupyterlab==3.2.9 \
  pandas==1.1.5 \
  scikit-learn==0.24.2 \
  plotnine==0.8.0 \
  seaborn==0.11.2 \
  pydicom==2.3.0 \
  tqdm==4.64.0

CMD jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
