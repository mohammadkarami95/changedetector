from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

def download_model( model_name):
    """Download a Hugging Face model and tokenizer to the specified directory"""
    # Check if the directory already exists
    model_dir = model_name.split('/')[-1]
    if not os.path.exists(model_dir):
        # Create the directory
        os.makedirs(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Save the model and tokenizer to the specified directory
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
#'MohammadKarami/simple-roberta', 'MohammadKarami/medium-roberta', 'MohammadKarami/medium-electra', 'MohammadKarami/medium-bert', 'MohammadKarami/whole-roBERTa', 'MohammadKarami/whole-electra', 'MohammadKarami/hard-roberta',
MODELS= ['MohammadKarami/simple-roberta', 'MohammadKarami/medium-roberta', 'MohammadKarami/medium-electra', 'MohammadKarami/medium-bert', 'MohammadKarami/whole-roBERTa', 'MohammadKarami/whole-electra', 'MohammadKarami/hard-roberta', 'MohammadKarami/hard-electra']
for model in MODELS:
    download_model(model)