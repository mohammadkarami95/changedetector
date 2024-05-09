FROM python:3

# ADD script.py /script.py
COPY . .
ADD requirements.txt /requirements.txt

RUN pip3 install --no-cache-dir -r /requirements.txt

ENTRYPOINT ["python3", "/script.py", "--input", "$inputDataset", "--output", "$outputDir" ]

