import os
import glob
import parsing
from convert import html_to_numpy
from grading import table_similarity
from tqdm import tqdm

base_path = os.path.expanduser("~/data/human_table_benchmark/rd-tablebench")

def get_ground_html(json_path: str) -> str:
    ground_path = os.path.join(base_path, 'groundtruth', json_path[json_path.rfind(os.sep) + 1:-5] + '.html')
    with open(ground_path, 'r', encoding='UTF-8') as fp:
        return fp.read()

def grade_reducto():
    global base_path
    similarities = []
    for path in tqdm(glob.glob(os.path.join(base_path, 'reducto_nov1', '*.json'))):
        ground_html = get_ground_html(path)
        html = parsing.parse_reducto_response(path)[0]
        if html:
            np_html = html_to_numpy(html)
            np_ground = html_to_numpy(ground_html)
            similarity = table_similarity(ground_truth=np_ground, prediction=np_html)
            similarities.append(similarity)
        else:
            similarities.append(0)

    print(f'gpt4o score={sum(similarities)/len(similarities)}')


def grade_gpt4o():
    global base_path
    similarities = []
    for path in tqdm(glob.glob(os.path.join(base_path, 'gpt4o', '*.json'))):
        ground_html = get_ground_html(path)
        html = parsing.parse_gpt4o_response(path)[0]
        if html:
            np_html = html_to_numpy(html)
            np_ground = html_to_numpy(ground_html)
            similarity = table_similarity(ground_truth=np_ground, prediction=np_html)
            similarities.append(similarity)
        else:
            similarities.append(0)

    print(f'gpt4o score={sum(similarities)/len(similarities)}')

if __name__ == "__main__":
    grade_reducto()