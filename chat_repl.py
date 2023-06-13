#!/usr/bin/env python3

import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig


model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
config = GenerationConfig(max_new_tokens=200)

while True:
    query = input("Query ? ").strip()
    tokens = tokenizer(query, return_tensors="pt")
    outputs = model.generate(**tokens, generation_config=config)
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(answer)
    print("=" * 80)
