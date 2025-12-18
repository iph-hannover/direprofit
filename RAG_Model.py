from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain.schema.document import Document
import faiss, torch, time, os
from llama_cpp import Llama

import nltk
from transformers import GPT2Tokenizer
nltk.download('punkt_tab')

def getDatabank():
    # Schritt 1: Daten vorbereiten
    documents = os.listdir('usedQuellen')
    sentences = []
    quellen = []
    for quelle in documents:
        if '.txt' in quelle:
            text = open(os.path.join('usedQuellen', quelle), "r", encoding="utf-8")
            text = text.read()
        text = cleanText(text)
        text_splitter = CharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=100
            )
        
        text_splitter = TokenTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            encoding_name="gpt2"
        )

        chunks_alt = [Document(page_content=x) for x in text_splitter.split_text(text)]
        chunks = split_into_chunks_by_sentence(text, max_tokens=550)
        for item in chunks:
            sentences.append(item)
            quellen.append(quelle)
    return sentences, quellen

def split_into_chunks_by_sentence(text, max_tokens=550):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    sentences = nltk.sent_tokenize(text)  # Zerlege den Text in Sätze
    tokens = []
    chunks = []
    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence)
        # Prüfe, ob das Hinzufügen des Satzes den Token-Limit überschreitet
        if len(tokens) + len(sentence_tokens) > max_tokens:
            # Wenn der Token-Limit überschritten wird, füge den aktuellen Chunk zu Chunks hinzu und starte einen neuen Chunk
            chunks.append(tokenizer.decode(tokens))
            tokens = sentence_tokens  # Starte neuen Chunk mit dem aktuellen Satz
        else:
            # Wenn der Token-Limit nicht überschritten wird, füge den Satz zum aktuellen Chunk hinzu
            tokens.extend(sentence_tokens)
    
    # Den letzten Chunk hinzufügen
    if tokens:
        chunks.append(tokenizer.decode(tokens))
    
    return chunks

def cleanText(text):
    text = text.replace("\n ", "")
    text = text.replace("\n-", "")
    text = text.replace("\n", "")
    text = text.replace("<!-- image -->", " ")
    text = text.replace("<!-- formula-not-decoded -->", " ")
    test = 7
    return text

def getPrompt(question, sentences, quellen, factsIndex, memoryIndex, sourceNr=3):
    query_embedding = faissmodel.encode([question])
    distances, indices = factsIndex.search(query_embedding, sourceNr)
    minDistance = 40
    distances = distances[0].tolist()
    indices = indices[0].tolist()
    # for nr in reversed(range(len(distances))):
    #     if distances[nr] > minDistance:
    #         del distances[nr]
    #         del indices[nr]
    #     else:
    #         break
    benutzeQuellen = [quellen[i] for i in indices]
    retrieved_documents = [sentences[i] for i in indices]
    history = retrieve_memory(question, memoryIndex)
    prompt = (
        "Rolle: Du bist Chatbot, deine Aufgabe ist den Anwender bei seinen Fragen zu helfen. Es wird von dir erwartet auf die Frage vom Anwender kurz und bündig zu antworten, ausser der Anwender erlaubt dir ausfürlich zu antworten. "
        "Vorherige relevante Konversationen:\n " +
        "".join(history) +

        "Hier sind relevante Dokumente die dir Kontexliefern könnten:\n"+
        "".join(retrieved_documents) +

        "Frage: " + "".join(question) + "\n" +
        " Antwort:"
        )
    return prompt, benutzeQuellen, memoryIndex

def getfaissFactsIndex(sentences):
    global faissmodel
    faissmodel = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
    embeddings = faissmodel.encode(sentences)
    dim = embeddings.shape[1]
    factsIndex = faiss.IndexFlatL2(dim)  # L2-Abstand (euklidisch)
    factsIndex.add(embeddings)
    return factsIndex

def getFaissMemorryIndex():
    dimension = 768
    memoryIndex = faiss.IndexFlatL2(dimension)
    memoryIndex = add_to_memory_faiss('', '', memoryIndex)
    return memoryIndex

def getLocalCPPModel():
    cPath = 'model'
#    modelName = 'llama-2-13b.Q8_0.gguf'
    modelName = 'Llama-3.1-8B-Instruct-bf16_q6_k.gguf'
    modelPath = os.path.join(cPath, modelName)
    useModel = Llama(
        model_path=modelPath,
        n_gpu_layers=-1, # Uncomment to use GPU acceleration
        # seed=1337, # Uncomment to set a specific seed
        n_ctx=4096, # Uncomment to increase the context windowK
    )
    return useModel

def getAntwort(prompt, user_input, memoryIndex, useModel, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    response = useModel.generate(inputs.input_ids, max_length=len(prompt)+2500, num_return_sequences=1)
    generated_text = tokenizer.decode(response[0], skip_special_tokens=True)
    generated_text = 'Chatbot:'+generated_text[len(prompt):]
    memoryIndex = add_to_memory_faiss(user_input, generated_text, memoryIndex)
    return generated_text, memoryIndex

def getCPPAntwort(prompt, user_input, memoryIndex, useModel):
    generated_text = useModel(
        prompt,
        temperature=0.5,
        top_p=0.90,
        top_k=50, 
        max_tokens=550,
        echo=False,
        stop=["Antwort:"]
        )
    generated_text = generated_text['choices'][0]['text']
    memoryIndex = add_to_memory_faiss(user_input, generated_text, memoryIndex)
    return generated_text, memoryIndex

conversation_memory = []

def add_to_memory_faiss(user_question, bot_answer, index):
    """Speichert die Konversation in FAISS."""
    global conversation_memory
    
    # Konvertiere in Vektoren
    text = f"Anwender: {user_question} \n {bot_answer}"
    conversation_memory.append(text)
    embedding = faissmodel.encode([text])
    
    # FAISS-Speicherung
    index.add(embedding)
    return index

def retrieve_memory(user_question, index, k=3):
    """Sucht ähnliche vergangene Dialoge mit FAISS."""
    query_embedding = faissmodel.encode([user_question])
    
    # Ähnlichste Erinnerungen abrufen
    D, I = index.search(query_embedding, k)
    I = I[0].tolist()
    for nr in reversed(range(len(I))):
        if I[nr] == 0 or I[nr] == -1:
            del I[nr]
    relevant_memories = [conversation_memory[i] for i in I if i < len(conversation_memory)]
    
    return "\n".join(relevant_memories)

timeStart = time.time()
sentences, quellen = getDatabank()
factsIndex = getfaissFactsIndex(sentences)
memoryIndex = getFaissMemorryIndex()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
useModel = getLocalCPPModel()

timeende = time.time()
print('Dauer bis Programm start: ' + str(timeende-timeStart))
print("Chatbot: Hallo, ich bin der Retrofit-Chatbot. Wie kann ich dir helfen?")
while True:
    user_input = input("Du: ")
    timeStart = time.time()
    prompt, benutzeQuellen, factsIndex = getPrompt(user_input, sentences, quellen, factsIndex, memoryIndex, sourceNr=3)
    # prompt, benutzeQuellen = getPrompt(user_input, sentences, quellen, sourceNr=4)
    if user_input.lower() == "exit":
        print("Chatbot: Auf wiedersehen!")
        break

    # response, index = getAntwort(prompt, user_input, memoryIndex, useModel, tokenizer, device)
    response, index = getCPPAntwort(prompt, user_input, memoryIndex, useModel)
    print("Chatbot:", response)
    timeende = time.time()
    print('Dauer: ' + str(timeende-timeStart))
    print('Antwort länge: ' + str(len(response)))
    print("Gefundene Quellen:")
    for source in benutzeQuellen:
        print(f"- {source}")
test = 7