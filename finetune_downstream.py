from argparse import ArgumentParser
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer, EarlyStoppingCallback
import numpy as np
import pandas
from sklearn.preprocessing import LabelEncoder
from load_datasets import load_esd_specific
from argparse import ArgumentParser
from sklearn.metrics import roc_auc_score, f1_score
import torch
from transformers import AutoFeatureExtractor, Wav2Vec2ForPreTraining
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices
from datasets import load_dataset, load_metric, Dataset, DatasetDict
import shutil
torch.set_num_threads(1)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--prefinetuned_model", type=str, default="facebook/wav2vec2-base")
    parser.add_argument("--max_duration", type=float, default=5.0) 
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    return parser.parse_args()

args = parse_args()
model_checkpoint = args.prefinetuned_model
feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
max_duration = args.max_duration

accuracy = load_metric("accuracy")
f1 = load_metric("f1")
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    acc = accuracy.compute(predictions=predictions, references=labels).get('accuracy')
    macro_f1_score = f1.compute(predictions=predictions, references=labels, average='macro').get('f1')
    micro_f1_score = f1.compute(predictions=predictions, references=labels, average='micro').get('f1')
    weighted_f1 = f1_score(predictions, labels, average='weighted')
    return {
        'accuracy':acc, 
        'macro_f1':macro_f1_score, 
        'micro_f1':micro_f1_score,
        'weighted_f1': weighted_f1
    }
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


def main(args):
    if 'facebook' in args.prefinetuned_model:
        corpora = "base"
    else:
        corpora = args.prefinetuned_model.split("wav2vec2-base-finetuned-")[1].split("-200epochs")[0]
    print("Corpora:", corpora)
    fewshot_samples = [2, 4, 8, 16, 24, 32, 64]
    speakers = [("Mandarin", "0001"),
    ("Mandarin", "0002"),
    ("Mandarin", "0003"),
    ("Mandarin", "0004"),
    ("Mandarin", "0005"),
    ("Mandarin", "0006"),
    ("Mandarin", "0007"),
    ("Mandarin", "0008"),
    ("Mandarin", "0009"),
    ("Mandarin", "0010"),
    ("English", "0011"),
    ("English", "0012"),
    ("English", "0013"),
    ("English", "0014"),
    ("English", "0015"),
    ("English", "0016"),
    ("English", "0017"),
    ("English", "0018"),
    ("English", "0019"),
    ("English", "0020")
    ]
    for language, speaker in speakers:
        for emotion in ["Angry", "Happy", "Neutral", "Sad", "Surprise"]:
                for num_fewshot in fewshot_samples:
                    for trial in range(3): 
                        tmp = pandas.read_csv("scaled_esd_downstream.csv", dtype={"speaker":int})
                        {corpora},{speaker},{language},{emotion},{num_fewshot},{trial}
                        tmp = tmp[tmp.corpora == corpora]
                        tmp = tmp[tmp.speaker == int(speaker)]
                        tmp = tmp[tmp.language == language]
                        tmp = tmp[tmp.emotion == emotion]
                        tmp = tmp[tmp.k == num_fewshot]
                        tmp = tmp[tmp.trial == trial]
                        if len(tmp) > 0:
                            print("Matched", f"{corpora},{speaker},{language},{emotion},{num_fewshot},{trial}")
                            continue
                        print("Language", language)
                        print("Speaker", speaker)
                        print("Emotion", emotion)
                        print("K =", num_fewshot)
                        print("Trial", trial)
                        data = load_esd_specific(speaker=speaker, emotion=emotion, k=num_fewshot)   
                        encoded_dataset = data.map(preprocess_function, remove_columns=["audio"], batched=True)
                        if 'facebook' in args.prefinetuned_model:
                            model_name = model_checkpoint.split("/")[-1]
                        else:
                            model_name = model_checkpoint
                        output_name = f"{corpora}-downstream-esd-speaker_{speaker}-emotion_{emotion}-k_{num_fewshot}-iteration_{trial}"
                        print(output_name)
                        training_epochs = 200
                        patience = 30
                        if num_fewshot == -1:
                            training_epochs = 10
                            patience = 3
                        elif num_fewshot > 8:
                            training_epochs = 50
                            patience = 5

                        trainingargs = TrainingArguments(
                            output_name,
                            evaluation_strategy = "epoch",
                            save_strategy = "epoch",
                            learning_rate=3e-5,
                            per_device_train_batch_size=args.train_batch_size,
                            gradient_accumulation_steps=4,
                            per_device_eval_batch_size=args.eval_batch_size,
                            warmup_ratio=0.1,
                            logging_steps=3,
                            load_best_model_at_end=True,
                            metric_for_best_model="macro_f1",
                            push_to_hub=False,
                            save_total_limit=2,
                            num_train_epochs=training_epochs
                        )
                        early_stop = EarlyStoppingCallback(early_stopping_patience=patience)
                        try:
                            model = AutoModelForAudioClassification.from_pretrained(
                                model_checkpoint, 
                            )
                        except:
                            #Manually correct layer dims during loading if needed
                            model_dims = torch.load(args.prefinetuned_model + "/pytorch_model.bin")
                            classifier_dimensions = model_dims['classifier.weight'].size()
                            bias_dimensions = model_dims['classifier.bias'].size()
                            model = AutoModelForAudioClassification.from_pretrained(
                                "facebook/wav2vec2-base", 
                            )
                            for layer in model_dims.keys():
                                model.state_dict()[layer] = model_dims[layer]
                            
                        print("Changing linear layer")
                        model.classifier = torch.nn.Linear(in_features=model.classifier.in_features, out_features=2, bias=True)
                        model.config.num_labels = 2

                        trainer = Trainer(
                            model,
                            trainingargs,
                            train_dataset=encoded_dataset["train"],
                            eval_dataset=encoded_dataset["evaluation"],
                            tokenizer=feature_extractor,
                            compute_metrics=compute_metrics,
                            callbacks=[early_stop]
                        )
                        print("Beginning training")
                        
                        trainer.train()
                        print(trainer.evaluate())
                        prediction, label_ids, metrics = trainer.predict(encoded_dataset['test'])
                        print(metrics)
                        with open("scaled_esd_downstream.csv","a") as f:
                            f.write(f"{corpora},{speaker},{language},{emotion},{num_fewshot},{trial},{metrics['test_accuracy']},{metrics['test_macro_f1']},{metrics['test_weighted_f1']}\n")
                        shutil.rmtree(output_name)

if __name__ == "__main__":   
    # global args 
    main(args)    
