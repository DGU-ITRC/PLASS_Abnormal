import os
import time
import json
import model
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/file/demo", StaticFiles(directory="static/demo"), name="input")
app.mount("/file/input", StaticFiles(directory="static/input"), name="input")
app.mount("/file/output", StaticFiles(directory="static/output"), name="output")

templates = Jinja2Templates(directory="template")

SERVER_INFO = {
    "name": "Abnormal Detection API Server",
    "version": "1.0.0",
    "port": 40000
}

@app.get("/")
async def demo(request: Request):
    """
    데모를 실행할 수 있는 페이지를 반환합니다.

    Html file: /templates/index.html

    Args:
        request (Request): FastAPI Request 객체

    Returns:
        TemplateResponse: HTML 파일
    """
    return templates.TemplateResponse("index.html",{"request":request})

@app.get("/infrence")
async def infrence():
    """
    데모(추론)를 실행합니다. 서버에 저장된 데모 동영상을 기반으로 추론을 시작합니다.

    Video file: /static/input/clip_short.mp4

    Args:
        request (Request): FastAPI Request 객체

    Returns:
        StreamingResponse: 추론 진행상태 및 결과를 반환하는 스트리밍 응답
    """
    return StreamingResponse(infrence_stream(), media_type='text/event-stream')

@app.post("/inference")
async def infrence_uploaded_video(video: UploadFile = File(...)):
    """
    데모(추론)를 실행합니다. 사용자가 서버에 업로드한 동영상을 기반으로 추론을 시작합니다.

    Video file: /static/demo/clip_short.mp4

    Args:
        video (UploadFile): 사용자가 업로드한 동영상 파일

    Returns:
        StreamingResponse: 추론 진행상태 및 결과를 반환하는 스트리밍 응답
    """
    timestamp = int(time.time() * 1000000)
    input_directory = "static/input"
    filename = f"{timestamp}_{video.filename}"
    filepath = os.path.join(input_directory, filename)
    content = await video.read()
    with open(filepath, "wb") as f:
        f.write(content)
    return StreamingResponse(infrence_stream(filepath), media_type='text/event-stream')

async def infrence_stream(videopath=None):
    """
    데모(추론)를 실행합니다. 서버에 업로드된 동영상을 기반으로 추론을 시작합니다.

    Args:
        videopath (str): 동영상 파일 경로, None일 경우 /static/demo/clip_short.mp4 사용

    Returns:
        추론 진행상태 및 결과 문자열
    """
    yield f"Target Video: {videopath}"
    start_time = time.time()
    yield "Infrence Started"
    async for response in model.inference(videopath):
        yield response
    yield "Infrence Done"
    end_time = time.time()
    yield f"Infrence Done in {end_time - start_time} seconds"
    duration = end_time - start_time
    yield json.dumps({"msg": f"Infrence Done in {duration} seconds", "data": response})