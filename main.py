from fastapi import FastAPI, Request
from transformers_model import generate_answer

app = FastAPI()

@app.post("/webhook")
async def webhook(request: Request):
    req_data = await request.json()
    intent = req_data['queryResult']['intent']['displayName']
    query = req_data['queryResult']['queryText']

    if intent in ["ask_hostel", "ask_programs", "ask_admission", "ask_fees"]:
        answer = generate_answer(query)
    else:
        answer = "I'm still learning to answer that. Try again later!"

    return {
        "fulfillmentText": answer
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
