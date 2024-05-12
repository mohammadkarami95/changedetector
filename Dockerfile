# Has already torch and cuda installed
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# ADD script.py /script.py

RUN pip3 install --no-cache-dir transformers argparse
RUN pip3 install huggingface_hub
RUN huggingface-cli download MohammadKarami/simple-roberta
RUN huggingface-cli download MohammadKarami/medium-roberta
RUN huggingface-cli download MohammadKarami/medium-electra
RUN huggingface-cli download MohammadKarami/medium-bert
RUN huggingface-cli download MohammadKarami/whole-roBERTa
RUN huggingface-cli download MohammadKarami/hard-roberta
RUN huggingface-cli download MohammadKarami/hard-electra
RUN huggingface-cli download MohammadKarami/whole-electra

ADD script.py /

ENTRYPOINT ["python3", "/script.py", "--input", "$inputDataset", "--output", "$outputDir" ]

