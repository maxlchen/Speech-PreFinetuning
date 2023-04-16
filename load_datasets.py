from collections import defaultdict
import os
import pandas
from datasets import load_dataset, load_metric, Dataset, DatasetDict
import soundfile as sf
import math
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

random.seed(123)
np.random.seed(123)

def load_iemocap_valence():
    return load_iemocap_base(label="valence")
def load_iemocap_emotion():
    return load_iemocap_base(label="emotion")
def load_iemocap_arousal():
    return load_iemocap_base(label="arousal")
def load_iemocap_dominance():
    return load_iemocap_base(label="dominance")

def load_iemocap_base(label="emotion"):
    if os.path.exists(f"{label}_iemocap_data.pkl"):
        with open(f"{label}_iemocap_data.pkl", "rb") as f:
            dataset = pickle.load(f)
        new_dataset = DatasetDict()        
        encoder = LabelEncoder
        encoder.classes_ = np.load(f"{label}_iemocap_label_encoder.npy")
        
    else:
        dataset = {
                'train': {
                    'audio' : [],
                    'label' : []
                },
                'evaluation': {
                    'audio' : [],
                    'label' : []
                },
                'test': {
                    'audio' : [],
                    'label' : []
                }
            }
        all_data = []
        unique_emotions = set()
        for session in ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']:
            for transcript in os.scandir("../IEMOCAP_full_release/" + session + '/dialog/EmoEvaluation/'):
                if transcript.is_dir(): continue
                transcript_name = transcript.path.split('/')[-1]
                if transcript_name.startswith("."): continue
                recording_session = transcript_name.replace('.txt', '')
                with open(transcript.path, 'r') as f:
                    line = f.readline()
                    while line:
                        if line.startswith("["):
                            chunks = line.split("\t")
                            soundfile = "../IEMOCAP_full_release/" + session + '/sentences/wav/' + recording_session + "/" + chunks[1] + ".wav"
                            VAD = [float(x) for x in line.split('\t')[-1].rstrip().strip('][').split(', ')]
                            valence = VAD[0]
                            arousal = VAD[1]
                            dominance = VAD[2]
                            emotion = chunks[2]
                            if label == "valence":
                                y = valence
                            elif label == "arousal":
                                y = arousal
                            elif label == "dominance":
                                y = dominance
                            elif label == "emotion":
                                y = emotion
                            else:
                                raise NotImplementedError
                            data, samplerate = sf.read(soundfile, always_2d=False, dtype='int16')
                            unique_emotions.add(y)
                            all_data.append((data, y, soundfile))
                        line = f.readline()                
                
        unique_emotions = list(unique_emotions)
        encoder = LabelEncoder()
        encoder.fit(unique_emotions)
        np.save(f"{label}_iemocap_label_encoder.npy", encoder.classes_)
        np.random.shuffle(all_data)
        train = all_data[:math.ceil(0.8 * len(all_data))]
        val = all_data[math.ceil(0.8 * len(all_data)) : math.ceil(0.9 * len(all_data))]
        test = all_data[math.ceil(0.9 * len(all_data)) :]
        for data, emotion, filename in train:
            dataset['train']['audio'].append(data.flatten())
            dataset['train']['label'].append(encoder.transform([emotion])[0])
        for data, emotion, filename in val:
            dataset['evaluation']['audio'].append(data.flatten())
            dataset['evaluation']['label'].append(encoder.transform([emotion])[0])
        for data, emotion, filename in test:
            dataset['test']['audio'].append(data.flatten())
            dataset['test']['label'].append(encoder.transform([emotion])[0])
        with open(f"{label}_iemocap_data.pkl", "wb") as f:
            pickle.dump(dataset, f)
    new_dataset = DatasetDict()
    for split in dataset.keys():
        new_dataset[split] = Dataset.from_dict(dataset[split])
    return new_dataset, encoder

def load_msp_improv():
    if os.path.exists("msp_improv_data.pkl"):
        with open("msp_improv_data.pkl", "rb") as f:
            dataset = pickle.load(f)
        new_dataset = DatasetDict()        
        encoder = LabelEncoder
        encoder.classes_ = np.load("msp_improv_label_encoder.npy")
        
    else:
        dataset = {
                'train': {
                    'audio' : [],
                    'label' : []
                },
                'evaluation': {
                    'audio' : [],
                    'label' : []
                },
                'test': {
                    'audio' : [],
                    'label' : []
                }
            }
        all_data = []
        unique_emotions = set()
        for session in ['session1', 'session2', 'session3', 'session4', 'session5', 'session6']:
            for folder in os.scandir("../MSP-IMPROV/" + session + '/'):
                folder_name = folder.path.split('/')[-1]
                speaker = folder_name[1:3]
                emotion = folder_name[-1]
                if emotion == "A":
                    emotion = "Angry"
                elif emotion == "H":
                    emotion = "Happy"
                elif emotion == "S":
                    emotion = "Sad"
                elif emotion == "N":
                    emotion = "Neutral"
                else:
                    raise NotImplementedError
                for recording_type in os.scandir("../MSP-IMPROV/" + session + '/' + folder_name + "/"):
                    recording_type_name = recording_type.path.split('/')[-1]
                    for recording in os.scandir("../MSP-IMPROV/" + session + '/' + folder_name + "/" + recording_type_name + '/'):
                        recording_name = recording.path.split('/')[-1]
                        data, samplerate = sf.read("../MSP-IMPROV/" + session + '/' + folder_name + "/" + recording_type_name + '/' + recording_name, always_2d=False, dtype='int16')
                        unique_emotions.add(emotion)
                        all_data.append((data, emotion, recording.path))
                
        unique_emotions = list(unique_emotions)
        encoder = LabelEncoder()
        encoder.fit(unique_emotions)
        np.save('msp_improv_label_encoder.npy', encoder.classes_)
        np.random.shuffle(all_data)
        train = all_data[:math.ceil(0.8 * len(all_data))]
        val = all_data[math.ceil(0.8 * len(all_data)) : math.ceil(0.9 * len(all_data))]
        test = all_data[math.ceil(0.9 * len(all_data)) :]
        for data, emotion, filename in train:
            dataset['train']['audio'].append(data.flatten())
            dataset['train']['label'].append(encoder.transform([emotion])[0])
        for data, emotion, filename in val:
            dataset['evaluation']['audio'].append(data.flatten())
            dataset['evaluation']['label'].append(encoder.transform([emotion])[0])
        for data, emotion, filename in test:
            dataset['test']['audio'].append(data.flatten())
            dataset['test']['label'].append(encoder.transform([emotion])[0])
        with open("msp_improv_data.pkl", "wb") as f:
            pickle.dump(dataset, f)
    new_dataset = DatasetDict()
    for split in dataset.keys():
        new_dataset[split] = Dataset.from_dict(dataset[split])
    return new_dataset, encoder

def load_mandarin_emotion():
    if os.path.exists("mandarin_aff_data.pkl"):
        with open("mandarin_aff_data.pkl", "rb") as f:
            dataset = pickle.load(f)
        new_dataset = DatasetDict()        
        encoder = LabelEncoder
        encoder.classes_ = np.load("mandarin_aff_label_encoder.npy")
        
    else:
        dataset = {
                'train': {
                    'audio' : [],
                    'label' : []
                },
                'evaluation': {
                    'audio' : [],
                    'label' : []
                },
                'test': {
                    'audio' : [],
                    'label' : []
                }
            }
        all_data = []
        unique_emotions = set()
        for folder_path in os.scandir("../Mandarin/man_aff_spch/data/"):
            if not folder_path.is_dir(): continue
            folder = folder_path.path.split("/")[-1]
            for emotion_path in os.scandir("../Mandarin/man_aff_spch/data/" + folder + "/"):
                emotion = emotion_path.path.split('/')[-1]
                for phrase in os.scandir("../Mandarin/man_aff_spch/data/" + folder + "/" + emotion + "/phrase/"):                
                    recording_name = phrase.path.split('/')[-1]
                    data, samplerate = sf.read("../Mandarin/man_aff_spch/data/" + folder + "/" + emotion + "/phrase/" + recording_name, always_2d=False, dtype="int16")
                    unique_emotions.add(emotion)
                    all_data.append((data, emotion, phrase.path))
                
        unique_emotions = list(unique_emotions)
        encoder = LabelEncoder()
        encoder.fit(unique_emotions)
        np.save('mandarin_aff_label_encoder.npy', encoder.classes_)
        np.random.shuffle(all_data)
        train = all_data[:math.ceil(0.8 * len(all_data))]
        val = all_data[math.ceil(0.8 * len(all_data)) : math.ceil(0.9 * len(all_data))]
        test = all_data[math.ceil(0.9 * len(all_data)) :]
        for data, emotion, filename in train:
            dataset['train']['audio'].append(data.flatten())
            dataset['train']['label'].append(encoder.transform([emotion])[0])
        for data, emotion, filename in val:
            dataset['evaluation']['audio'].append(data.flatten())
            dataset['evaluation']['label'].append(encoder.transform([emotion])[0])
        for data, emotion, filename in test:
            dataset['test']['audio'].append(data.flatten())
            dataset['test']['label'].append(encoder.transform([emotion])[0])
        with open("mandarin_aff_data.pkl", "wb") as f:
            pickle.dump(dataset, f)
    new_dataset = DatasetDict()
    for split in dataset.keys():
        new_dataset[split] = Dataset.from_dict(dataset[split])
    return new_dataset, encoder

def load_msp_podcast():
    if os.path.exists("msp_podcast_data.pkl"):
        with open("msp_podcast_data.pkl", "rb") as f:
            dataset = pickle.load(f)
        new_dataset = DatasetDict()        
        encoder = LabelEncoder
        encoder.classes_ = np.load("msp_podcast_label_encoder.npy")
        
    else:
        labels_mapping = pandas.read_csv("../MSP-PODCAST/Labels/labels_consensus.csv")
        unique_emotions = set(labels_mapping['EmoClass'])
        unique_emotions = list(unique_emotions)
        encoder = LabelEncoder()
        encoder.fit(unique_emotions)
        np.save('msp_podcast_label_encoder.npy', encoder.classes_)
        dataset = {
                'train': {
                    # "file" : [],
                    'audio' : [],
                    'label' : []
                },
                'evaluation': {
                    # "file" : [],
                    'audio' : [],
                    'label' : []
                },
                'test': {
                    # "file" : [],
                    'audio' : [],
                    'label' : []
                }
            }
        all_data = []        
        for i, row in labels_mapping.iterrows():
            filename = row['FileName']
            label = row['EmoClass']
            split = row['Split_Set']
            if 'Test' in split:
                split = 'test'
            if 'Dev' in split:
                split = 'evaluation'
            if 'Train' in split:
                split = 'train'
            data, samplerate = sf.read("../MSP-PODCAST/Audio/" + filename, dtype='int16')
            if np.max(data) > 2147483647:
                print(data)
                print(filename)
            dataset[split]['audio'].append(data)
            dataset[split]['label'].append(encoder.transform([label])[0])
        with open("msp_podcast_data.pkl", "wb") as f:
            pickle.dump(dataset, f)
    new_dataset = DatasetDict()
    for split in dataset.keys():
        new_dataset[split] = Dataset.from_pandas(pandas.DataFrame(dataset[split]))
    return new_dataset, encoder
    
def load_esd():
    text_labels = ['Angry', 'Happy', 'Neutral', "Sad", "Surprise"]
    english = ['0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020']    
    if os.path.exists("ESD_data.pkl"):
        with open("ESD_data.pkl", "rb") as f:
            dataset = pickle.load(f)
        encoder = LabelEncoder
        encoder.classes_ = np.load("ESD_label_encoder.npy")
    else:
        encoder = LabelEncoder()
        encoder.fit(text_labels)
        np.save('ESD_label_encoder.npy', encoder.classes_)
        dataset = {
            'train': {
                'audio' : [],
                'label' : []
            },
            'evaluation': {
                'audio' : [],
                'label' : []
            },
            'test': {
                'audio' : [],
                'label' : []
            }
        }
        for speaker in english:
            for file in os.scandir("../ESD/" + speaker + '/'):
                if file.is_dir():
                    label = file.path.split('/')[-1]
                    for model_split in os.scandir("../ESD/" + speaker + '/' + label + '/'):
                        split = model_split.path.split('/')[-1]
                        for soundfile in os.scandir("../ESD/" + speaker + '/' + label + '/' + split + '/'):
                            if len(soundfile.path.split('_')) > 2: continue
                            data, samplerate = sf.read(soundfile.path, dtype='int16')
                            dataset[split]['audio'].append(data)
                            dataset[split]['label'].append(encoder.transform([label])[0])
        with open("ESD_data.pkl", 'wb') as f:
            pickle.dump(dataset, f)
    new_dataset = DatasetDict()
    for split in dataset.keys():            
        new_dataset[split] = Dataset.from_dict(dataset[split])
    return new_dataset, encoder

def load_esd_specific(speaker='0011', emotion='Angry', k=2):
    '''
    speakers: 00 + (01 through 20)
    emotion: ['Angry', 'Happy', 'Neutral', "Sad", "Surprise"]
    k: num fewshot training examples
    - choose k=-1 for full shot
    '''
    text_labels = ['Angry', 'Happy', 'Neutral', "Sad", "Surprise"]
    assert emotion in text_labels
    english = ['0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020']    
    mandarin = ['0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010']    
    all_speakers = english + mandarin
    assert speaker in all_speakers
    dataset = {
        'train': {
            'audio' : [],
            'label' : []
        },
        'evaluation': {
            'audio' : [],
            'label' : []
        },
        'test': {
            'audio' : [],
            'label' : []
        }
    }
    pos_pool = []
    neg_pool = []
    for speaker_dir in all_speakers:
        if speaker_dir != speaker: continue
        for file in os.scandir("../ESD/" + speaker + '/'):
            if file.is_dir():
                label = file.path.split('/')[-1]
                numeric_label = int(label == emotion)
                for model_split in os.scandir("../ESD/" + speaker_dir + '/' + label + '/'):
                    split = model_split.path.split('/')[-1]
                    for soundfile in os.scandir("../ESD/" + speaker_dir + '/' + label + '/' + split + '/'):
                        if len(soundfile.path.split('_')) > 2: continue
                        data, samplerate = sf.read(soundfile.path, dtype='int16')
                        if split != "train":
                            dataset[split]['audio'].append(data)
                            dataset[split]['label'].append(numeric_label)
                        else:
                            if numeric_label == 0:
                                neg_pool.append((data, numeric_label))
                            else:
                                pos_pool.append((data, numeric_label))

    np.random.shuffle(pos_pool)
    np.random.shuffle(neg_pool)
    if k > -1:
        num_per_label = math.ceil(k / 2)
        pos_pool = pos_pool[:num_per_label]
        neg_pool = neg_pool[:num_per_label]
    train_pool = pos_pool + neg_pool
    np.random.shuffle(train_pool)
    for sound, emote in train_pool:
        dataset['train']['audio'].append(sound)
        dataset['train']['label'].append(emote)

    new_dataset = DatasetDict()
    for split in dataset.keys():            
        new_dataset[split] = Dataset.from_dict(dataset[split])
    print(new_dataset)
    return new_dataset

def load_esd_specific_filenames(speaker='0011', emotion='Angry', k=2):
    '''
    speakers: 00 + (01 through 20)
    emotion: ['Angry', 'Happy', 'Neutral', "Sad", "Surprise"]
    k: num fewshot training examples
    - choose k=-1 for full shot
    '''
    text_labels = ['Angry', 'Happy', 'Neutral', "Sad", "Surprise"]
    assert emotion in text_labels
    english = ['0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020']    
    mandarin = ['0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010']    
    all_speakers = english + mandarin
    assert speaker in all_speakers
    dataset = {
        'train': {
            'audio' : [],
            'label' : []
        },
        'evaluation': {
            'audio' : [],
            'label' : []
        },
        'test': {
            'audio' : [],
            'label' : []
        }
    }
    pos_pool = []
    neg_pool = []
    for speaker_dir in all_speakers:
        if speaker_dir != speaker: continue
        for file in os.scandir("../ESD/" + speaker + '/'):
            if file.is_dir():
                label = file.path.split('/')[-1]
                numeric_label = int(label == emotion)
                for model_split in os.scandir("../ESD/" + speaker_dir + '/' + label + '/'):
                    split = model_split.path.split('/')[-1]
                    for soundfile in os.scandir("../ESD/" + speaker_dir + '/' + label + '/' + split + '/'):
                        if len(soundfile.path.split('_')) > 2: continue
                        data, samplerate = sf.read(soundfile.path, dtype='int16')
                        if split != "train":
                            dataset[split]['audio'].append(soundfile.path)
                            dataset[split]['label'].append(numeric_label)
                        else:
                            if numeric_label == 0:
                                neg_pool.append((data, numeric_label))
                            else:
                                pos_pool.append((data, numeric_label))

    np.random.shuffle(pos_pool)
    np.random.shuffle(neg_pool)
    if k > -1:
        num_per_label = math.ceil(k / 2)
        pos_pool = pos_pool[:num_per_label]
        neg_pool = neg_pool[:num_per_label]
    train_pool = pos_pool + neg_pool
    np.random.shuffle(train_pool)
    for sound, emote in train_pool:
        dataset['train']['audio'].append(sound)
        dataset['train']['label'].append(emote)

    new_dataset = DatasetDict()
    for split in dataset.keys():            
        new_dataset[split] = Dataset.from_dict(dataset[split])
    print(new_dataset)
    return new_dataset

if __name__ == "__main__":
    print(sum(load_esd_specific(speaker='0011', emotion='Angry', k=32)['train'][2]['audio']))
    
