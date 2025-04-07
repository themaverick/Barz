import re
import json
from bs4 import BeautifulSoup
import pandas as pd

def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

annots_list = []

with open("../dataset/genius-expertise/annotation_info.json", 'r') as f:
    for line in f:
        annots_list.append(json.loads(line))

drake_songs = []

for idx, i in enumerate(annots_list):
    if i['song'].startswith('Drake-'):
        if i['song'] not in drake_songs:
            drake_songs.append(i['song'])


song_dataset = [] 

for idx, i in enumerate(drake_songs):
    song_name = i
    annotated_lyrics = []
    for k in annots_list:
        if k['song'] == song_name and k['type'] == 'reviewed':
            annotation = remove_html_tags(k['edits_lst'][0]['content'])
            lyr_snip = k['lyrics']

            annotated_lyrics.append({
                'lyr_snip': lyr_snip,
                'annotation': annotation
            })
    element = {
        'song_name': song_name, 
        'annotated_lyrics': annotated_lyrics
    }
    song_dataset.append(element)
    print(f'phase 1 {idx}/{len(drake_songs)}')
    # print(f"song number: {idx+1}")

song_list = []

with open("../dataset/genius-expertise/lyrics.jl", 'r') as f:
    for line in f:
        song_list.append(json.loads(line))

print(song_list[0])

for idx, i in enumerate(song_list):
    for k in song_dataset:
        # print(k['song_name'])
        if i['song'] == k['song_name']:
            k['lyrics'] = i['lyrics']
            # print(i['song'])
            print(f'num songs matched: {idx+1}/{len(song_list)}')

with open('../dataset/dataset.json', 'w') as json_file:
    json.dump(song_dataset, json_file, indent=4) 