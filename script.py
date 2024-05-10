import os
import argparse
import json
import glob
import re
# import model_downloader
from transformers import pipeline #, AutoModelForSequenceClassification, AutoTokenizer

def pars_args():
    parser= argparse.ArgumentParser(description= "PAN 2024 Style Change Detection Task.")
    parser.add_argument("--input", type= str, help= "Folder containing input files for task(.txt)", required= True)
    parser.add_argument("--output", type= str, help= "Folder containing output/solution files(.json)", required= True)
    args = parser.parse_args()
    return args

def read_problem_files(problems_folder):
    problems = {}
    
    solution_files = glob.glob(f'{problems_folder}/problem-*.txt') \
        + glob.glob(f'{problems_folder}/**/problem-*.txt') \
        + glob.glob(f'{problems_folder}/**/**/problem-*.txt')
    # solution_files = glob.glob(f'{problems_folder}/test/problem-*.txt')
    print(f'Some of solution_files are: {solution_files[:5]})
    
    for file in solution_files:
        file_num= re.findall(r'\d+', str(file))[0]
        with open(file, 'r') as f:
            data= []
            problem= f.readlines()
            for i in range(len(problem)-1):
                data.append(problem[i] + ' ' + problem[i + 1])
    #             # print('append : '+ str(lines[i]) + ' ' + str(lines[i + 1]))
            problems[f'problem-{file_num}']= data
    return problems
            


def easy_test(problems, output_path):
    # model = AutoModelForSequenceClassification.from_pretrained('MohammadKarami/simple-roberta')
    # tokenizer = AutoTokenizer.from_pretrained('MohammadKarami/simple-roberta')
    print('EASY functions')
    print(f'Number of EASY files:{len(problems)}')
    classifier= pipeline('text-classification', model= 'MohammadKarami/simple-roberta', tokenizer='MohammadKarami/simple-roberta', max_length= 512, truncation= True)
    print('MODEL LOADED')
    print('working of easy files...')
    os.makedirs(output_path, exist_ok= True)
    for problem in problems:
        preds= []
        for text in problems[problem]:
            output= classifier(text)
            if output[0]['label']== 'isnt':
                preds.append(0)
            else:
                preds.append(1)
        file_num= re.findall(r'\d+', problem)[0]
        with open(output_path+"/solution-"+file_num+".json", 'w') as out:
            prediction = {'changes': preds}
            out.write(json.dumps(prediction)) 
    print('Easy files predicted\n ------------------------------------\n')

def most_frequent(List):
    return max(set(List), key = List.count)

def medium_test(problems, output_path):
    print('medium function')
    print(f'Number of MEDIUM files:{len(problems)}')
    print('medium models is running...')
    # roberta_model = AutoModelForSequenceClassification.from_pretrained('./medium-roberta')
    # roberta_tokenizer = AutoTokenizer.from_pretrained('./medium-roberta')
    roberta_classifier= pipeline('text-classification', model='MohammadKarami/medium-roberta', tokenizer='MohammadKarami/medium-roberta', max_length= 512, truncation= True)

    # electra_model = AutoModelForSequenceClassification.from_pretrained('./medium-electra')
    # electra_tokenizer = AutoTokenizer.from_pretrained('./medium-electra')
    electra_classifier= pipeline('text-classification', model='MohammadKarami/medium-electra', tokenizer='MohammadKarami/medium-electra', max_length= 512, truncation= True)

    # bert_model = AutoModelForSequenceClassification.from_pretrained('./medium-bert')
    # bert_tokenizer = AutoTokenizer.from_pretrained('./medium-bert')
    bert_classifier= pipeline('text-classification', model='MohammadKarami/medium-bert', tokenizer='MohammadKarami/medium-bert', max_length= 512, truncation= True)

    # whole_roberta_model = AutoModelForSequenceClassification.from_pretrained('./whole-roBERTa')
    # whole_roberta_tokenizer = AutoTokenizer.from_pretrained('./whole_roBERTa')
    whole_roberta_classifier= pipeline('text-classification', model='MohammadKarami/whole-roBERTa', tokenizer='MohammadKarami/whole-roBERTa', max_length= 512, truncation= True)

    # whole_electra_model = AutoModelForSequenceClassification.from_pretrained('./whole_electra')
    # whole_electra_tokenizer = AutoTokenizer.from_pretrained('./whole-electra')
    whole_electra_classifier= pipeline('text-classification', model='MohammadKarami/whole-electra', tokenizer='MohammadKarami/whole-electra', max_length= 512, truncation= True)

    # electra_classifier= pipeline('text-classification', model='MohammadKarami/medium-electra', tokenizer="MohammadKarami/medium-electra", max_length= 512, truncation= True)
    # bert_classifier= pipeline('text-classification', model='MohammadKarami/medium-bert', tokenizer="MohammadKarami/medium-bert", max_length= 512, truncation= True)
    # whole_roberta_classifier= pipeline('text-classification', model='MohammadKarami/whole-roBERTa', tokenizer="MohammadKarami/whole-roBERTa", max_length= 512, truncation= True)
    # whole_electra_classifier= pipeline('text-classification', model='MohammadKarami/whole-electra', tokenizer="MohammadKarami/whole-electra", max_length= 512, truncation= True)
    print('medium models is loaded')
    os.makedirs(output_path, exist_ok= True)
    print('working on medium fils...')
    for problem in problems:
        # print(f'{problem} file is working...')
        preds= []
        sample_pred= []

        for text in problems[problem]:
            roberta_output= roberta_classifier(text)
            if roberta_output[0]['label']== 'isnt':
                sample_pred.append(0)
            else:
                sample_pred.append(1)

            electra_output= electra_classifier(text)
            if electra_output[0]['label']== 'isnt':
                sample_pred.append(0)
            else:
                sample_pred.append(1)

            bert_output= bert_classifier(text)
            if bert_output[0]['label']== 'isnt':
                sample_pred.append(0)
            else:
                sample_pred.append(1)

            whole_roberta_output= whole_roberta_classifier(text)
            if whole_roberta_output[0]['label']== 'isnt':
                sample_pred.append(0)
            else:
                sample_pred.append(1)

            whole_electra_output= whole_electra_classifier(text)
            if whole_electra_output[0]['label']== 'isnt':
                sample_pred.append(0)
            else:
                sample_pred.append(1)
            preds.append(most_frequent(sample_pred))
        file_num= re.findall(r'\d+', problem)[0]
        with open(output_path+"/solution-"+file_num+".json", 'w') as out:
            prediction = {'changes': preds}
            out.write(json.dumps(prediction))
    print('medium files predicted\n ------------------------------------\n')
def hard_test(problems, output_path):
    print('Hard function')
    print(f'Number of HARD files:{len(problems)}')
    print('Hard models is loading...')
    # roberta_model = AutoModelForSequenceClassification.from_pretrained('./hard-roberta')
    # roberta_tokenizer = AutoTokenizer.from_pretrained('./hard-roberta')
    roberta_classifier= pipeline('text-classification', model='MohammadKarami/hard-roberta', tokenizer='MohammadKarami/hard-roberta', max_length= 512, truncation= True)

    # electra_model = AutoModelForSequenceClassification.from_pretrained('./hard-electra')
    # electra_tokenizer = AutoTokenizer.from_pretrained('./hard-electra')
    electra_classifier= pipeline('text-classification', model='MohammadKarami/hard-electra', tokenizer='MohammadKarami/hard-electra', max_length= 512, truncation= True)

    # whole_roberta_model = AutoModelForSequenceClassification.from_pretrained('./whole-roBERTa')
    # whole_roberta_tokenizer = AutoTokenizer.from_pretrained('./whole-roBERTa')
    whole_roberta_classifier= pipeline('text-classification', model='MohammadKarami/whole-roBERTa', tokenizer='MohammadKarami/whole-roBERTa', max_length= 512, truncation= True)
    print('Hard models loaded')
    os.makedirs(output_path, exist_ok= True)

    print('working on medium files...')
    for problem in problems:
        # print(f'{problem} file is working...')
        preds= []
        sample_pred= []

        for text in problems[problem]:
            roberta_output= roberta_classifier(text)
            if roberta_output[0]['label']== 'isnt':
                sample_pred.append(0)
            else:
                sample_pred.append(1)

            electra_output= electra_classifier(text)
            if electra_output[0]['label']== 'isnt':
                sample_pred.append(0)
            else:
                sample_pred.append(1)

            whole_roberta_output= whole_roberta_classifier(text)
            if whole_roberta_output[0]['label']== 'isnt':
                sample_pred.append(0)
            else:
                sample_pred.append(1)

            preds.append(most_frequent(sample_pred))

        file_num= re.findall(r'\d+', problem)[0]
        with open(output_path+"/solution-"+file_num+".json", 'w') as out:
            prediction = {'changes': preds}
            out.write(json.dumps(prediction))

    print('Hard files predicted!')
    
def main():
    args= pars_args()
    print('TESTING TIME')
    # model_downloader()
    for subtask in ['easy', 'medium', 'hard']:
        if subtask =='easy':
            print('EASY files are reading...')
            problems= read_problem_files(args.input+f"/{subtask}")
            easy_test(problems, args.output+f"/{subtask}")
        elif subtask=='medium':
            print('MEDIUM files are reading...')
            problems= read_problem_files(args.input+f"/{subtask}")
            print('MEDIUM files are read')
            medium_test(problems, args.output+f"/{subtask}")
        else:
            print('HARD files are reading...')
            problems= read_problem_files(args.input+f"/{subtask}")
            print('HARD files are read')
            hard_test(problems, args.output+f"/{subtask}")


def file_nanmes():
    arg= pars_args()
    dires= [x[0] for x in os.walk(arg.input+"/*")]
    print(f"list of directors:{dires}")


if __name__=='__main__':
    main()
