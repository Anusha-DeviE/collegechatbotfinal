import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

default_context = """
Our college provides various undergraduate and postgraduate programs, including B.Tech, B.Sc, M.Tech, and MBA.
The hostel facilities include separate buildings for boys and girls, mess services, Wi-Fi, and 24/7 security.
The admission process usually begins in April. You can apply online through our website.
For fees and scholarships, please contact the accounts department or check the official fee structure online.
"""

def generate_answer(question: str, context: str = default_context):
    inputs = tokenizer(question, context, return_tensors="pt")
    outputs = model(**inputs)

    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
    )
    return answer
