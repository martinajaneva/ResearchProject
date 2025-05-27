import json
import torch
import os
import gc
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from openai import OpenAI
from API_KEY import API_KEY
from tqdm import tqdm
from more_itertools import chunked


client = OpenAI(api_key = API_KEY)
def bullet_list(text):
    response = client.chat.completions.create(
        model = "gpt-4.1-nano",
        messages = [
            {"role": "system", "content":f"You are a research assistant that summarizes academic limitation sections."},
            {"role": "user", "content":f"Convert the following text into 3-6 clear bullet points:\n {text}"}
        ],
        max_tokens=300
    )
    return (response.choices[0].message.content).strip()


def generate_lims(model_path, input_path, output_path, max_input_length=512, max_output_tokens=256, check_limitations=False):
    model_path = os.path.abspath(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model.to(device)
    model.eval()

    with open(input_path, "r", encoding="utf-8") as f:
        papers = json.load(f)
    res = []
    size_batch = 8
    paper_batches = list(chunked(papers, size_batch))
    for batch in tqdm(paper_batches):
        already_processed = set()
        for p in batch:
            if check_limitations and p["limitations"]:
                paper_id = p.get("paper", "")
                
                limitations = p["limitations"]
                decoded = bullet_list(limitations)
                res.append({"paper": paper_id, "generated": decoded})
                already_processed.add(paper_id)
        input_texts = []
        paper_ids = []
        for p in batch:
            paper_id = p.get("paper", "")
            if paper_id not in already_processed:
                input_texts.append(p["input"])
                paper_ids.append(paper_id)
        
        if not input_texts:
            continue
        
        inputs = tokenizer(input_texts, return_tensors="pt",
        truncation=True,
        max_length=max_input_length, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens = max_output_tokens,
                num_beams=2,
                early_stopping=True,
                pad_token_id = tokenizer.eos_token_id
            )
        decoded_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        torch.cuda.empty_cache()
        for paper_id, decoded in zip(paper_ids, decoded_batch):
            res.append({"paper": paper_id, "generated": decoded})
    print("Loop Finished")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in res:
            f.write(json.dumps(item) + "\n")