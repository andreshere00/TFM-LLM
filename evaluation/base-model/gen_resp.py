from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import torch
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
import evaluate
import os
import time

# Configuración para cuantizar el modelo en 4 bits
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Cargar el modelo y el tokenizador con autenticación
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    quantization_config=bnb_config, 
    device_map="auto",
    use_auth_token=True
)

print(f"Modelo {model_name} cargado y cuantizado en 4 bits")

# Cargar el dataset de prueba con autenticación
dataset = load_dataset("andreshere/counsel_chat", split="test", use_auth_token=True)

print(f"Dataset {dataset} cargado")

# Seleccionar una muestra de 200 ejemplos
sample_size = 200
sampled_dataset = dataset.select(range(sample_size))

# Función para generar respuesta dada una entrada de contexto y medir el tiempo
def generate_response(context):
    prompt = f"### Instructions: Respond accordingly to the input with a maximum length of 1024 words. \n\n ### Input: {context} \n\n ### Response: "
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=1024)
    end_time = time.time()
    response_time = end_time - start_time
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response, response_time

# Aplicar inferencia sobre el campo "Context" del dataset de prueba
contexts = []
references = []
responses = []
response_times = []

for example in sampled_dataset:
    context = example["Context"]
    reference = example["Response"]  # La respuesta de referencia está en el campo "Response"
    response, response_time = generate_response(context)
    contexts.append(context)
    references.append(reference)
    responses.append(response)
    response_times.append(response_time)
    print(f"Context: {context}")
    print(f"Reference: {reference}")
    print(f"Model Response: {response}")
    print(f"Response Time: {response_time} seconds")
    print("="*50)

# Guardar las respuestas y los tiempos en un nuevo archivo
df = pd.DataFrame({
    "Context": contexts,
    "Reference": references,
    "Model response": responses,
    "Response time (seconds)": response_times
})

df.to_csv("dataset_sample.csv", index=False)

print("Respuestas y tiempos guardados en 'dataset_sample.csv'")

# Calcular las métricas
bleu_scores = []
rouge = evaluate.load('rouge')
meteor = evaluate.load('meteor')
bertscore = evaluate.load('bertscore')

for ref, resp in zip(references, responses):
    bleu_score = sentence_bleu([ref.split()], resp.split())
    bleu_scores.append(bleu_score)

rouge_results = rouge.compute(predictions=responses, references=references)
meteor_results = meteor.compute(predictions=responses, references=references)
bertscore_results = bertscore.compute(predictions=responses, references=references, lang="en")

# Promedio de las métricas
avg_bleu = sum(bleu_scores) / len(bleu_scores)
avg_rouge = {key: sum([score[key] for score in rouge_results[key]]) / len(rouge_results[key]) for key in rouge_results.keys()}
avg_meteor = meteor_results["meteor"]
avg_bertscore = {key: sum(bertscore_results[key]) / len(bertscore_results[key]) for key in bertscore_results.keys()}

# Mostrar resultados
print(f"Average BLEU: {avg_bleu}")
print(f"Average ROUGE: {avg_rouge}")
print(f"Average METEOR: {avg_meteor}")
print(f"Average BERTScore: {avg_bertscore}")

# Guardar métricas en un archivo
metrics_df = pd.DataFrame({
    "Metric": ["BLEU", "ROUGE-L", "METEOR", "BERTScore"],
    "Score": [avg_bleu, avg_rouge['rougeL'], avg_meteor, avg_bertscore['f1']]
})

metrics_df.to_csv("metrics.csv", index=False)

print("Métricas guardadas en 'metrics.csv'")
