from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, BertTokenizerFast
from optimum.onnxruntime import ORTModelForTokenClassification



model_fine_tuned = ORTModelForTokenClassification.from_pretrained("rubert-model-x5")
tokenizer = BertTokenizerFast.from_pretrained("tokenizer-rubert-model-x5")
nlp = pipeline("ner", model=model_fine_tuned, tokenizer=tokenizer, ignore_labels=[])



class PredictRequest(BaseModel):
    input: str

app = FastAPI()

def get_response(example):
    ans = []
    example = nlp(example)

    if example[0]["entity"] == '0':
        example[0]["entity"] = 'O'

    curr = {"start_index": example[0]["start"], "end_index":example[0]["end"], "entity": example[0]["entity"]}
    for i in example[1:]:
        start, end, ent = i["start"], i["end"], i["entity"]
        if curr["end_index"] == start:
            curr["end_index"] = end
        else:
            ans.append(curr)
            curr = {"start_index": start, "end_index": end, "entity": 'O' if ent == '0' else ent}
    ans.append(curr)
    return ans


@app.post("/api/predict")
def predict(request: PredictRequest):
    words = request.input
    words_cleaned = " ".join(words.split())
    if words_cleaned == "":
        return []
    else:
        response = get_response(words_cleaned)
    return response