import os
os.environ["OMP_NUM_THREADS"] = "16" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "16" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "16" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "16" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "16" # export NUMEXPR_NUM_THREADS=1
import torch
from transformers import AutoFeatureExtractor, Wav2Vec2ForPreTraining
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices
from datasets import load_dataset, load_metric, Dataset, DatasetDict
import soundfile as sf
import pickle
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer, EarlyStoppingCallback
import numpy as np
import pandas
from sklearn.preprocessing import LabelEncoder
from load_datasets import load_esd, load_msp_improv, load_iemocap_valence, load_iemocap_emotion, load_iemocap_arousal, load_iemocap_dominance, load_mandarin_emotion, load_msp_podcast
from argparse import ArgumentParser
from utils import MultiFinetuningTrainer

os.environ["WANDB_DISABLED"] = "true"

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--corpus", type=str, default="iemocap_emotion,msp_improv,mandarin_aff")
    parser.add_argument("--model", type=str, default="facebook/wav2vec2-base")
    parser.add_argument("--max_duration", type=float, default=5.0) 
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    return parser.parse_args()
args = parse_args()
model_checkpoint = args.model
feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
max_duration = args.max_duration

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    print(predictions, eval_pred.label_ids)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
def preprocess_function(examples):
    audio_arrays = [x for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=int(feature_extractor.sampling_rate * max_duration), 
        truncation=True, 
        padding=True,
    )
    return inputs
    
def load_corpus(corpus):
    if corpus == "ESD":
        loader = load_esd
    elif corpus == "msp_improv":
        loader = load_msp_improv
    elif corpus == "iemocap_valence":
        loader = load_iemocap_valence
    elif corpus == "iemocap_emotion":
        loader = load_iemocap_emotion
    elif corpus == "iemocap_arousal":
        loader = load_iemocap_arousal
    elif corpus == "iemocap_dominance":
        loader = load_iemocap_dominance
    elif corpus == "mandarin_aff":
        loader = load_mandarin_emotion
    elif corpus == "msp_podcast":
        loader = load_msp_podcast
    else:
        raise NotImplementedError
    return loader

accuracy = load_metric("accuracy")

def main(args):    
    corpora = []
    for corpus in args.corpus.split(','):
        loader = load_corpus(corpus)
        ds, le = loader()
        ds = ds.map(preprocess_function, remove_columns=['audio'], batched=True)    
        corpora.append((corpus, ds, le))        

    batch_size = args.batch_size
 
   
    model_name = "scaled_" + model_checkpoint.split("/")[-1] 
    corpus_str = args.corpus.replace(",","-")
    output_name = f"{model_name}-finetuned-{corpus_str}-{args.epochs}epochs"
    trainingargs = TrainingArguments(
        output_name,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=args.epochs,
        warmup_ratio=0.1,
        logging_steps=3,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        save_total_limit=1,
    )

    early_stop = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.02)

    trainer = MultiFinetuningTrainer(
            AutoModelForAudioClassification.from_pretrained(
                model_checkpoint, 
            ),
            train_dataset={
                corpus[0] : corpus[1]['train'] for corpus in corpora
            },
            eval_dataset={
                corpus[0] : corpus[1]['evaluation'] for corpus in corpora
            },
            tokenizer=feature_extractor,
            args=trainingargs,
            compute_metrics=compute_metrics,
            callbacks = [early_stop]
        )
    print("Beginning training")
    old_collator = trainer.data_collator
    trainer.data_collator = lambda data: dict(old_collator(data))
    if args.resume_checkpoint:
        trainer.train(True)
    else:
        trainer.train()
    trainer.save_model("best_"+output_name)
    print(trainer.evaluate())

if __name__ == "__main__":   
    # global args 
    main(args)    
