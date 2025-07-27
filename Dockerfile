# For more information, please refer to https://aka.ms/vscode-docker-python
# FROM nvcr.io/nvidia/pytorch:21.05-py3

FROM nvcr.io/nvidia/pytorch:21.05-py3

WORKDIR /workdir

## Copy all your files of the current folder into Docker Container
COPY ./ /workdir
RUN chmod a+x /workdir/inference.py

## Install requirements
RUN pip3 install -r requirements.txt

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
ENTRYPOINT ["/opt/conda/bin/python", "inference.py"]
# ENTRYPOINT python -m process $0 $@

