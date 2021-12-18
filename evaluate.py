from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import pickle
from datasets import load_metric
import random
import pandas as pd


def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    print(df)

def map_to_result(batch):
    with open("processor_hy.pkl", "rb") as f:
        processor = pickle.load(f)

    model = Wav2Vec2ForCTC.from_pretrained("elizabethkeleshian-wav2vec2-base-hy")

    with torch.no_grad():
        input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_str"] = processor.batch_decode(pred_ids)[0]
    batch["text"] = processor.decode(batch["labels"], group_tokens=False)
    
    return batch


if __name__ == "__main__":
    wer_metric = load_metric("wer")

    with open("prepared_test_hy.pkl", "rb") as f:
        test = pickle.load(f)
    
    results = test.map(map_to_result, remove_columns=test.column_names)
    wer_score = wer_metric.compute(predictions=results["pred_str"],references=results["texts"])
    print(f"Test WER: {wer_score:.3f}")
    show_random_elements(results.remove_columns(["speech", "sampling_rate"]))
    
