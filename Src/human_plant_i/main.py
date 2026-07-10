import os
import asyncio
import cv2
import async_reader
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn

from my_logger import Logger, LogLevel


app = FastAPI()
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
logger = Logger("WebStream", LogLevel.DEBUG)

@app.get("/", response_class=HTMLResponse)
async def get_web_page(request: Request):
    logger.log_debug("Serving index.html template to client.")
    return templates.TemplateResponse(request=request, name="index.html")

@app.websocket("/ws/{camera_id}")
async def websocket_endpoint(websocket: WebSocket, camera_id: str):
    accept_task = asyncio.create_task(websocket.accept())
    index = int(camera_id[3:]) - 1
    start_stream_task = asyncio.create_task(async_reader.reader_start(index = index))

    await accept_task
    ret = await start_stream_task
    if not ret:
        return
    logger.log_info(f"WebSocket channel connection opened for {camera_id}.")
    try:
        while True:
            # The read method is blocking, so we run it in a separate thread to avoid blocking the event loop
            get_frame_task = asyncio.create_task(async_reader.reader_read(index = index))
            
            frame = await get_frame_task
            if frame is None:
                if async_reader.is_reader_stop(index = index):
                    break
                await asyncio.sleep(0.1)  # Yield control to event loop when stream is offline
                continue             
            ret, jpeg_buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            await websocket.send_bytes(jpeg_buffer.tobytes())
    except WebSocketDisconnect:
        logger.log_info(f"WebSocket client disconnected from {camera_id}.")
    except Exception as e:
        logger.log_error(f"Exception on {camera_id} socket: {str(e)}")
    finally:
        stop_reader_task = asyncio.create_task(async_reader.reader_stop(index = index))
        websocket.stop()
        await stop_reader_task
        logger.log_info(f"WebSocket channel connection closed for {camera_id}.")
        
def main():
    logger.log_info("Starting FastAPI application server via Uvicorn.")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()