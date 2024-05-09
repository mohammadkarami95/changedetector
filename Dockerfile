FROM python:3

# ADD script.py /script.py
COPY . .
ADD requirements.txt /requirements.txt

RUN pip3 install --no-cache-dir -r /requirements.txt
RUN pip3 install huggingface_hub
RUN huggingface-cli download MohammadKarami/simple-roberta
RUN huggingface-cli download MohammadKarami/medium-roberta
RUN huggingface-cli download MohammadKarami/medium-electra
RUN huggingface-cli download MohammadKarami/whole-roBERTa
RUN huggingface-cli download MohammadKarami/hard-roberta
RUN huggingface-cli download MohammadKarami/hard-electra

ENTRYPOINT ["python3", "/script.py", "--input", "$inputDataset", "--output", "$outputDir" ]

