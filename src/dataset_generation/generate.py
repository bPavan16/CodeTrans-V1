API_KEY = "Your_gooogle_gemini_key"
import pandas as pd
import google.generativeai as genai
import time

# Load the dataset
data = pd.read_csv("xlcost_text_to_code.csv")

# Configure Google Generative AI
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")
text=data['text']
dataset=pd.read_csv("Dataset.csv")


n = dataset.shape[0]
print(n)



# Generate Pseudo Code
for i in range(n,len(text)) :

    response = model.generate_content("Write a function to" + text[i] +"in Pseudo code, give only the function without comments also dont write main and dont use built in feature  ")
    print(response.text)
    pseudo_response = response
    pseudo_code = response.text.strip()  # Strip any extra whitespace
    time.sleep(5)

    # Generate Python code
    response = model.generate_content(pseudo_response.text +"Use this pseudo code to generate python code no comments just the  function Dont write main and write in such a way that it can be translated into other language dont use byilt in feature ")
    print(response.text)
    python_code = response.text.strip()  # Strip any extra whitespace
    time.sleep(5)


    # Generate Java code
    response = model.generate_content(pseudo_response.text +"Use this pseudo code to generate java code no comments just the function Dont write main and write in such a way that it can be translated into other language dont use byilt in feature")
    print(response.text)
    java_code = response.text.strip()  # Strip any extra whitespace

    # Prepare new data as a DataFrame
    new_data = pd.DataFrame({
        "Pseudo Code": [pseudo_code],
        "Java": [java_code],
        "Python": [python_code]  # Remove trailing space in the column name
    })

    # Append new data to the existing DataFrame
    dataset = pd.concat([dataset, new_data], ignore_index=True)
    time.sleep(5)
    dataset.to_csv("Dataset.csv",index=False)

print("This message is displayed after a 3-second delay.")


# Display the updated DataFrame
print(dataset)

