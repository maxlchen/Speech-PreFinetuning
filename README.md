# Pre-Finetuning for Emotional Speech Recognition

Citation:

```
@article{chen2023pre,
  title={Pre-Finetuning for Few-Shot Emotional Speech Recognition},
  author={Chen, Maximillian and Yu, Zhou},
  journal={arXiv preprint arXiv:2302.12921},
  year={2023}
}
```

Paper Link: https://arxiv.org/abs/2302.12921

Request Access to Wav2Vec2.0 Base pre-finetuned on four corpora: https://drive.google.com/file/d/1N1JxqN8Ts2OWcoBTiHYt693DZF2sackV/view?usp=share_link

Repository under construction.

Please additionally cite the corresponding corpora if you use any of them for fine-tuning or pre-finetuning.

## Downstream Fine-tuning Corpora

Emotional Speech Dataset: https://github.com/HLTSingapore/Emotional-Speech-Data

## Pre-Finetuning Corpora

IEMOCAP: https://sail.usc.edu/iemocap/

Mandarin Affective Speech: https://catalog.ldc.upenn.edu/LDC2007S09

MSP-Podcast: https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html

MSP-Improv: https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Improv.html

## Requirements

transformers 4.18.0

## Notes

Currently, the Trainer class for multitask learning only has single GPU support.
