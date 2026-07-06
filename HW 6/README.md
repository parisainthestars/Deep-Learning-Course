# HW6 — Fine-Tuning mT5 with LoRA for Persian Informal-to-Formal Style Transfer

## Overview

This homework fine-tunes `google/mt5-base` for Persian informal-to-formal style transfer using **LoRA** and the **PEFT** library. The task is sequence-to-sequence generation: the model receives an informal Persian sentence and generates a more formal version.

The main goals were:

- preprocess the ParsMap informal/formal Persian corpus,
- tokenize the data using the mT5 tokenizer,
- fine-tune mT5 efficiently with LoRA,
- test greedy decoding on custom informal sentences,
- implement BLEU, perplexity, and stochastic decoding experiments.

---

## Dataset

The dataset used in this homework is the **ParsMap** informal-to-formal Persian corpus. The original columns were renamed as:

| Original Column | New Column | Meaning |
|---|---|---|
| `inFormalForm` | `input` | Informal Persian sentence |
| `formalForm` | `target` | Formal Persian sentence |

After preprocessing and splitting:

| Split | Number of Samples |
|---|---:|
| Train | 45,011 |
| Validation | 2,501 |
| Test | 2,501 |

The text was normalized using `hazm.Normalizer`, and rows with missing input or target values were removed.

---

## Token Length Statistics

Before padding and truncation, token-length statistics were computed using the `google/mt5-base` tokenizer.

| Text Type | Min | Max | Mean | 95th Percentile |
|---|---:|---:|---:|---:|
| Input | 3 | 146 | 22.70 | 45 |
| Target | 4 | 150 | 24.66 | 48 |

Based on these values, the maximum lengths were selected as:

| Parameter | Value |
|---|---:|
| `MAX_SOURCE_LEN` | 50 |
| `MAX_TARGET_LEN` | 53 |

These values keep most examples while avoiding unnecessary padding.

---

## Model

The base model was:

```text
google/mt5-base
```

Instead of fine-tuning all model parameters, **LoRA** was used for parameter-efficient fine-tuning.

The LoRA configuration was:

| Parameter | Value |
|---|---:|
| Rank `r` | 8 |
| `lora_alpha` | 32 |
| Target modules | `q`, `v` |
| LoRA dropout | 0.10 |
| Bias | `none` |
| Task type | `SEQ_2_SEQ_LM` |

The number of trainable parameters was:

| Parameter Type | Count |
|---|---:|
| Trainable parameters | 884,736 |
| All parameters | 583,286,016 |
| Trainable percentage | 0.1517% |

This shows the main benefit of LoRA: only a very small fraction of the model parameters are updated.

---

## Training Setup

The model was trained using `Seq2SeqTrainer`.

| Setting | Value |
|---|---:|
| Optimizer setup | HuggingFace Seq2SeqTrainer default optimizer |
| Learning rate | 5e-4 |
| Train batch size | 16 |
| Eval batch size | 16 |
| Epochs | 4 |
| Weight decay | 0.01 |
| Mixed precision | FP16 |
| Evaluation strategy | Each epoch |
| Save strategy | Each epoch |

---

## Training Results

The validation loss decreased consistently during training.

| Epoch | Training Loss | Validation Loss |
|---:|---:|---:|
| 1 | 1.0597 | 0.5435 |
| 2 | 0.5668 | 0.3616 |
| 3 | 0.4949 | 0.3164 |
| 4 | 0.4437 | 0.3047 |

The decreasing validation loss shows that the LoRA-adapted mT5 model learned the informal-to-formal transformation pattern.

---

## Inference Examples

Greedy decoding was used for five custom informal Persian sentences.

| Informal Input | Model Output |
|---|---|
| واسه چی اینقدر دیر اومدی؟ | برای چه این قدر دیر آمده ام؟ |
| برو اونور وایسا! | برو آنور وایسا! |
| خیلی باحالی داداش! | خیلی باحالی داداش است. |
| نمیدونم چرا اینجوری شد. | نمی دانم چرا این جوری شد. |
| یه چیزی بپرسم؟ | یک چیزی بپرسم؟ |

The model successfully formalized some informal words, such as:

| Informal | Formalized |
|---|---|
| واسه چی | برای چه |
| نمیدونم | نمی دانم |
| یه | یک |
| اینجوری | این جوری |

Some outputs are still imperfect, especially for very colloquial expressions such as `داداش`, which shows that the model needs more tuning or better data coverage.

---

## Evaluation

The notebook includes code for computing:

| Metric | Meaning |
|---|---|
| BLEU | Measures overlap between generated formal sentences and reference formal sentences |
| Perplexity | Measures how confident the model is when predicting validation targets |

The full BLEU/perplexity evaluation was attempted, but generating predictions for the whole test set at once caused a CUDA out-of-memory error. A better solution would be to compute BLEU in smaller batches.

---

## Stochastic Decoding

The notebook also implements stochastic decoding using:

- temperature sampling,
- top-k sampling,
- top-p nucleus sampling.

Unlike greedy decoding, stochastic decoding can generate multiple different formal outputs for the same informal input. This is useful when there are several acceptable formal rewrites.

---

## Key Takeaways

<div align="center">

| Concept | Main Takeaway |
|---|---|
| mT5 | A multilingual text-to-text Transformer suitable for Persian generation |
| LoRA | Fine-tunes large models by training only small low-rank adapter matrices |
| PEFT | Makes fine-tuning large models more memory-efficient |
| Token statistics | Help choose practical max sequence lengths |
| Greedy decoding | Produces deterministic outputs |
| Stochastic decoding | Produces more diverse outputs |
| Limitation | Some informal expressions still need better handling |

</div>

---

## Possible Improvements

<div align="center">

| Improvement | Reason |
|---|---|
| Use batched BLEU evaluation | Avoids CUDA out-of-memory errors |
| Train for more epochs | May improve formalization quality |
| Tune LoRA rank and alpha | Can improve adaptation strength |
| Try target modules beyond `q` and `v` | May improve generation quality |
| Use beam search | Can improve deterministic generation |
| Clean the corpus more carefully | Reduces noisy mappings |
| Add more informal examples | Helps with slang and colloquial expressions |

</div>

---

## Conclusion

This homework fine-tuned `google/mt5-base` for Persian informal-to-formal style transfer using LoRA. The dataset was normalized, tokenized, and split into train, validation, and test sets. LoRA made the fine-tuning efficient by training only 0.1517% of the total model parameters.

The training loss and validation loss decreased steadily over 4 epochs, showing successful learning. Greedy decoding produced reasonable formal rewrites for several Persian informal inputs, although some colloquial expressions were still not fully formalized.

Overall, this assignment demonstrates how parameter-efficient fine-tuning can adapt a large multilingual model to a Persian text generation task with much lower computational cost than full fine-tuning.