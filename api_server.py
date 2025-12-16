from fastapi import FastAPI, File, UploadFile, Form
from predict import translate_kor_to_vie
from predict_2 import translate_vie_to_kor
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import whisper
app = FastAPI()
model = whisper.load_model("small")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/kor2vie")
def kor2vie(text: str):
    result = translate_kor_to_vie(text)
    return {"result": result}

@app.get("/vie2kor")
def vie2kor(text: str):
    result = translate_vie_to_kor(text)
    return {"result": result}
    
@app.post("/voice2text")
async def voice2text(src_tts_lang: str = Form(...), file: UploadFile = File(...)):
    # nhận dữ liệu từ mobile
    print("Lang =", src_tts_lang)
    audio = await file.read()

    # lưu file tạm
    with open("voice.webm", "wb") as f:
        f.write(audio)

    # chuyển voice thành text
    result = model.transcribe("voice.webm", language=src_tts_lang)
    text = result["text"]
    
    

    return {"text": text}
    
    
if __name__ == "__main__":
    uvicorn.run(app, host=XXXX port=XXXX)
