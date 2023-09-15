from flask import Flask, request, jsonify, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

@app.route("/", methods=["GET", "POST"])
def chatbot():
    if request.method == "POST":
        user_message = request.form["user_message"]
        
        input_ids = tokenizer.encode(user_message, return_tensors="pt")
        response_ids = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
        bot_response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        
        return render_template("index.html", user_message=user_message, bot_response=bot_response)
    
    return render_template("index.html", user_message="", bot_response="")

if __name__ == "__main__":
    app.run(debug=True)