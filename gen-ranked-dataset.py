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


def calc_score(
        alignscore: AlignScore, 
        sbert: SentenceTransformer, 
        y_w: str, 
        y_l: str) -> Tuple[float, float]:
    """Computes the score between two texts `y_w` and `y_l`

    TODO: Implement as the combination of the AlignScore and the cosine similarity score.

    
    Keyword arguments: \\
    y_w (str) -- ground truth (or the golden option) \\
    y_l (str) -- losing option (generally the response from a LM) \\
    
    Return: (float) score between `y_w` and `y_l`
    """

    # for debuging
    # print(f'{y_w} / {y_l}')
    
    ## calculate alignscore
    a_s = alignscore.score(contexts=[y_w], claims=[y_l])[0]

    ## calc cosine sim. using sbert
    embs = sbert.encode([y_w, y_l])
    emb_w, emb_l = embs[0], embs[1]
    s_s = sbert.similarity(emb_w, emb_l).item()

    return a_s, s_s


def gen_triplets(
        options: List[str], 
        scores: List[float]) -> Tuple [str, str, float]:
    """Generates triplets from the given list of options and corresponding scores. Important: the lists must be sorted in decreasing order 
    
    Keyword arguments: \\
    options (List) -- list of options (generally responses from a LM, sorted by their corresponding scores) \\
    scores (List) -- list of sorted scores corresponding to each options. \\
    Return: (Tuple) -- Tuple of the form (y_w, y_l, lambda_score)
    """
    
    triplets = []
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

    for d in tqdm(data):
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
    
    ## initialize the alignscore model and sbert
    tqdm.write('Calculating scores ...')
    alignscore = AlignScore()
    sbert = SentenceTransformer('all-MiniLM-L6-v2')

    rejected_options_scores = {}
    for k, v in tqdm(formatted_data.items()):
        ## for each rejected options, get the scores agains each chosen options
        chosen_options = v['chosen']
        rejected_options = v['rejected']

        option_scores = []

        for rejected_option in rejected_options:
            if len(rejected_option.strip()) == 0:
                ## wierd: sometimes the response is empty
                tqdm.write('found empty response')
                continue
            
            align_score, cos_sim = 0, 0

            for chosen_option in chosen_options:
                _as, _cs = calc_score(alignscore=alignscore, sbert=sbert, y_w=chosen_option, y_l=rejected_option)
                align_score += _as
                cos_sim += _cs

                del _as, _cs

            option_scores.append({
                    'response': rejected_option,
                    'alignscore': align_score/len(chosen_options), 
                    'simscore': cos_sim/len(chosen_options)
            })
        rejected_options_scores[k] = option_scores

    del alignscore, sbert, formatted_data


    align_scores = [entry["alignscore"] for responses in rejected_options_scores.values() for entry in responses]
    sim_scores = [entry["simscore"] for responses in rejected_options_scores.values() for entry in responses]

    tqdm.write('Calculating score weights')
    alpha_align, alpha_sim = calc_score_weights(alignscores=align_scores, simscores=sim_scores, tau=args.tau)


    tqdm.write('Generating ranked dataset')
    final_dataset = []
    for k, v in tqdm(rejected_options_scores.items()):
        response_score_pairs = []
        for item in v:
            response_score_pairs.append((item['response'], alpha_align * item['alignscore'] + alpha_sim * item['simscore']))
        ## sort the responses using the final scores
        response_score_pairs.sort(key=lambda x: x[1], reverse=True)

        sorted_responses = [pair[0] for pair in response_score_pairs]
        sorted_response_scores = [pair[1] for pair in response_score_pairs]

        del response_score_pairs

        triplets = gen_triplets(sorted_responses, sorted_response_scores)

        for triplet in triplets:
            y_w, y_l, lambda_wl = triplet
            final_dataset.append({
                'query': k,
                'chosen': y_w,
                'rejected': y_l,
                'lambda': lambda_wl
            })

            del triplet

        del triplets
    
    save_to_pkl(file_path=args.output, data=final_dataset)
            