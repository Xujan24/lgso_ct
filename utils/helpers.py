import os
from typing import Any, Dict, List
import pickle as pkl
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def load_from_pkl(file_path: str) -> Any:
    '''
    Loads data from pickle

    Arguments:
    --------------
    file_path (`any`) - path for the pickle file

    Returns:
    --------------
    `any` - the data object
    '''
    with open(file_path, 'rb') as f:
        return pkl.load(f)


def save_to_pkl(file_path: str, data: Any) -> None:
    '''
    Save obj to file

    Arguments:
    --------------
    file_path (`str`) - path to save the file \\
    data (`any`) - the data object

    Returns:
    --------------
    `None`
    '''

    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    with open(file_path, 'wb') as f:
        pkl.dump(data, f)


def gen_training_samples(x: List) -> Dict:
    '''
    Generates a traing sample with the query as the input and the ground-truth as the output
    ** Note: Used just for the testing **
    '''
    return {
        'input': gen_query(x[0], x[1]),
        'output': x[2]
    }


def gen_query(term: str, relation: str) -> str:
    '''
    Creates query given the input term and the relation.

    Arguments:
    -----------------
    term (`str`) - the input term \\
    relation (`str`) - the relation

    Returns:
    -----------
    `str` - query string.
    '''
    VALID_RELATIONS = ['finding_site', 'associated_morphology', 'synonyms', 'causative_agent', 'interprets', 'severity']
    if relation not in VALID_RELATIONS:
        raise ValueError(f"Invalid relation: {relation}")
    
    prompt = ''
    if relation == VALID_RELATIONS[0]:
        ## finding_site
        prompt = """Answer the question using the format shown in the context.
                    [Question] What is the finding site for 'hemiplegia of nondominant side due to and following embolic cerebrovascular accident'?
                    [Answer] central nervous system structure
                    [Question] What is the finding site for 'threat to breathing due to cave-in, falling earth and other substances'?
                    [Answer] thoracic structure
                    [Question] What is the finding site for '%s'?
                """%(term)
    elif relation == VALID_RELATIONS[1]:
        ## associated_morphology
        prompt = """Answer the question using the format shown in the context.
                    [Question] What is the associated morphology for 'threat to breathing due to cave-in, falling earth and other substances'?
                    [Answer] compression
                    [Question] What is the associated morphology for 'late effect of radiation'?
                    [Answer] traumatic abnormality
                    [Question] What is the associated morphology for '%s'?
                """%{term}
    elif relation == VALID_RELATIONS[2]:
        ## synonyms
        prompt = """Answer the question using the format shown in the context.
                    [Question] What is the synonym for 'hydroxyurea poisoning'?
                    [Answer] hydroxycarbamide poisoning
                    [Question] What is the synonym for 'transperitoneal migration'?
                    [Answer] external migration
                    [Question] What is the synonym for '%s'?
                """%(term)
    elif relation == VALID_RELATIONS[3]:
        ## causative_agent
        prompt = """Answer the question using the format shown in the context.
                    [Question] What is the causative agent for 'hiv disease resulting in multiple infections'?
                    [Answer] human immunodeficiency virus
                    [Question] What is the causative agent for 'familial dementia british type'?
                    [Answer] amyloid beta peptide
                    [Question] What is the causative agent for '%s'?
                """%(term)
    elif relation == VALID_RELATIONS[4]:
        ## interprets
        prompt = """Answer the question using the format shown in the context.
                    [Question] What is 'short of breath dressing/undressing' interprets as?
                    [Answer] respiratory function
                    [Question] What is 'bacterial colony morphology, erose margin' interprets as?
                    [Answer] patient evaluation procedure 
                    [Question] What is the '%s' interprets as?
                """%(term)
    elif relation == VALID_RELATIONS[5]:
        ## severity
        prompt = """Answer the question using the format shown in the context.
                    [Question] What is the severity for 'hyperemesis gravidarum with metabolic disturbance unspecified'?
                    [Answer] severe
                    [Question] What is the severity for 'better eye: moderate visual impairment, lesser eye: total visual impairment'?
                    [Answer] moderate
                    [Question] What is the severity for '%s'?
                """%(term)

    return prompt


def get_responses_from_ref_model(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, query: List[str], max_iter = 15, max_new_tokens: int = 256) -> List[List[str]]:
    res = []

    inputs = tokenizer(query, return_tensors="pt", padding=True, padding_side='left').to(model.device)  # Move inputs to the same device as the model

    for i in range(max_iter):
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens) # Adjust max_new_tokens as needed

        generated_texts = [tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]
        generated_texts = list(map(__format_response, generated_texts))

        if i == 0:
            res = generated_texts
            del outputs, generated_texts
            continue

        for j in range(len(query)):
            res[j] = [*res[j], *generated_texts[j]]
        
        del outputs, generated_texts
    
    return res


def __format_response(x: str) -> List[str]:
    return [x.split('\n')[6].strip().split(']')[-1].strip().lower()]