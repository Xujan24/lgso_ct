"""
Script to generate ranked training dataset.
The input data must be of format:
[
    {'chosen': [
        {
            'role': 'user', 
            'content': <query>
        }, {
            'role': 'agent', 
            'content': <winning option>
        }
    ],
    'rejected': [
        {
            'role': 'user', 
            'content': <query>
        }, {
            'role': 'agent', 
            'content': <losing option>
        }
    ]},
    ...
]

The formatted data structure is as follows:
[
    {
        'query': <query>,
        'chosen': <wining option>,
        'rejected': <losing option>,
        'lambda' : <ranked score>    
    },
    ...
]
"""


from tqdm import tqdm
from typing import List, Tuple, Union
import numpy as np
import argparse
from AlignScore.alignscore import AlignScore
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import torch
from utils.helpers import load_from_pkl, save_to_pkl


def calc_score_weights(
        alignscores: List[float], 
        simscores: List[float], 
        tau: Union[float, int] = 0.2) -> Tuple[float, float]:

    ## calc the mean and standard deviation of the corresponding scores
    mu_alignscore = np.mean(alignscores)
    std_alignscore = np.std(alignscores)

    mu_simscore = np.mean(simscores)
    std_simscore = np.std(simscores)

    ## calculate the coefficient of variance (CoV)
    c_alignscore = std_alignscore/mu_alignscore
    c_simscore = std_simscore/mu_simscore
    del mu_alignscore, std_alignscore, mu_simscore, std_simscore

    scores = torch.tensor(np.array([c_alignscore, c_simscore])/tau)
    del c_alignscore, c_simscore

    alpha_align, alpha_sim = F.softmax(scores, dim=-1).tolist()
    del scores

    return alpha_align, alpha_sim


def calc_scores(
        scorer: AlignScore, 
        sbert: SentenceTransformer, 
        chosen_options: List[str], 
        rejected_options: List[str]
    ) -> Tuple[List[float], List[float]]:
    """
    Calculates scores using the SBERT and AlignScore.

    Keyword arguments: \\
    scorer (AlignScore) - AlignScore instance \\
    sbert (SentenceTransformers) - Sentence transformers instance \\
    chosen_options (List[str]) - list of chosen options \\
    rejected_options (List[str]) - list of rejected options

    returns:\\
    Tuple[List[float], List[float]] - tuple of scores
    """
    ## calculate the alignment score using AlignScore
    ## we use all the chosen options as the context and all the rejected options as the claims
    ## the current implementation of AlignScore does not support calculating score between each contexts and each claims
    ## it calculates the score between corresponding contexts and claims
    ## we extend the contexts and claims so that we can calculate the alignment score on a single pass
    adjusted_chosen_options = chosen_options * len(rejected_options)
    adjusted_rejected_options = [item for item in rejected_options for _ in range(len(chosen_options))]

    align_scores = scorer.score(contexts=adjusted_chosen_options, claims=adjusted_rejected_options)
    align_scores = np.reshape(np.array(align_scores), (len(rejected_options), -1)).T
    align_scores = np.mean(align_scores, axis=0).tolist()

    del adjusted_chosen_options, adjusted_rejected_options

    ## calculate the similarity score using SBERT
    embs = sbert.encode([*chosen_options, *rejected_options])
    cos_scores = sbert.similarity(embs[:len(chosen_options)], embs[len(chosen_options):])
    cos_scores = cos_scores.mean(axis=0).tolist()

    del embs

    return align_scores, cos_scores


def calc_sim_scores(
        sbert: SentenceTransformer, 
        chosen_options: List[str], 
        rejected_options: List[str]
    ) -> List[float]:
    """
    Calculates the scores using SBERT only

    Keyword arguments: \\
    sbert - an instance of Sentence transformer model \\
    chosen_options - list of chosen options \\
    rejected_options - list of rejected options

    returns:\\
    List[float] - a list of cosine similarity scores
    """
    ## calculate scores using only the SBERT
    embs = sbert.encode([*chosen_options, *rejected_options])
    cos_scores = sbert.similarity(embs[:len(chosen_options)], embs[len(chosen_options):])
    cos_scores = cos_scores.mean(axis=0).tolist()

    del embs

    return cos_scores


def gen_triplets(
        options: List[str], 
        scores: List[float]) -> Union[Tuple [str, str, float], None]:
    """Generates triplets from the given list of options and corresponding scores. Important: the lists must be sorted in decreasing order 
    
    Keyword arguments: \\
    options (List) -- list of options (generally responses from a LM, sorted by their corresponding scores) \\
    scores (List) -- list of sorted scores corresponding to each options. \\
    Return: (Tuple) -- Tuple of the form (y_w, y_l, lambda_score)
    """
    
    triplets = []
    if len(options) == 1:
        return None
    
    for i in range(len(options)):
        if i == len(options) - 1:
            break
        for j in range(i+1, len(options)):
            y_w = options[i]
            y_l = options[j]

            s_w = scores[i]
            s_l = scores[j]

            r_w = i+1
            r_l = j+1

            lambda_wl = s_w * delta_MMR(r_w, r_l) + s_l * delta_MMR(r_l, r_w)

            triplets.append((y_w, y_l, lambda_wl))

            del y_w, y_l, s_w, s_l, r_w, r_l, lambda_wl
    
    return triplets


def delta_MMR(r1: int, r2: int) -> float:
    """Computes the mean reciprocal rank (MMR).
    
    Keyword arguments: \\
    r1, r2 (int) -- ranks \\
    Return: (float) -- the MMR between r1 and r2
    """
    
    return (1/r1 -1/r2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to the data file')
    parser.add_argument('--output', type=str, required=True, help='Path to the output file')
    parser.add_argument('--tau', type=float, default=0.2, help='tau value for calculating score weights')
    args = parser.parse_args()

    data = load_from_pkl(args.input)

    ## transforming the data in the format {query, [winning values], [losing values]}

    formatted_data = {}
    tqdm.write('Formatting data...')

    for d in tqdm(data[:5]):
        ## get current query
        chosen = d.get('chosen')
        rejected = d.get('rejected')
        
        query = chosen[0].get('content')

        ## check if the current query is already in the formatted data
        if not query in formatted_data.keys():
            formatted_data[query] = {
                'chosen' : [],
                'rejected' : []
            }
        
        chosen = chosen[1].get('content')
        rejected = rejected[1].get('content')
        
        formatted_data.get(query).get('chosen').append(chosen.lower())
        formatted_data.get(query).get('rejected').append(rejected.lower())

        del chosen, rejected

        formatted_data.get(query)['chosen'] = list(set(formatted_data.get(query)['chosen']))
        formatted_data.get(query)['rejected'] = list(set(formatted_data.get(query)['rejected']))

        del query

    ## using SBERT only
    tqdm.write('Calculating individual scores ...')
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    scorer = AlignScore()

    align_scores = []
    cos_scores = []

    rejected_options_scores = {}
    for k, v in tqdm(formatted_data.items()):
        chosen_options = list(filter(lambda x: len(x.strip()) > 0, v['chosen']))
        rejected_options = list(filter(lambda x: len(x.strip()) > 0, v['rejected']))
        _align_scores, _cos_scores = calc_sim_scores(scorer=scorer, sbert=sbert, chosen_options=chosen_options, rejected_options=rejected_options)

        ## store all the scores to calculate the score weights later
        align_scores.extend(_align_scores)
        cos_scores.extend(_cos_scores)

        rejected_options_scores[k] = {
            'rejected_options': rejected_options,
            'align_scores': _align_scores,
            'cos_scores': _cos_scores
        }

        del chosen_options, rejected_options, _align_scores, _cos_scores
    
    
    tqdm.write('Calculating score weights ...')
    alpha_align, alpha_cos = calc_score_weights(alignscores=align_scores, simscores=cos_scores, tau=args.tau)


    tqdm.write('Generating ranked dataset')
    final_dataset = []
    for k, v in tqdm(rejected_options_scores.items()):
        wt_scores = alpha_align * np.array(v['align_scores']) + alpha_cos * np.array(v['cos_scores'])
        response_score_pairs = list((x, y) for x, y in zip(v['rejected_options'], wt_scores))
        response_score_pairs.sort(key=lambda x: x[1], reverse=True)

        sorted_responses = [pair[0] for pair in response_score_pairs]
        sorted_response_scores = [pair[1] for pair in response_score_pairs]

        del response_score_pairs, wt_scores

        triplets = gen_triplets(sorted_responses, sorted_response_scores)

        for triplet in triplets:
            if triplet == None:
                continue
            
            y_w, y_l, lambda_wl = triplet
            final_dataset.append({
                'query': k,
                'chosen': y_w,
                'rejected': y_l,
                'lambda': lambda_wl.item()
            })

            del y_w, y_l, lambda_wl, triplet

        del triplets
    
    save_to_pkl(file_path=args.output, data=final_dataset)
            