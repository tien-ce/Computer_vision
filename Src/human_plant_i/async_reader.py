import os
import asyncio
import numpy as np
from dotenv import load_dotenv
from my_inference import ModelInference
from my_reader import Reader
from my_logger import Logger, LogLevel

load_dotenv()
rtp_url_camera1 = os.environ.get("RTP_STREAM_URL1")
rtp_url_camera2 = os.environ.get("RTP_STREAM_URL2")
rtp_urls = [rtp_url_camera1, rtp_url_camera2]
stream_readers:list[Reader] = []
for rtp_url in rtp_urls:
    stream_readers.append(Reader(rtp_url, queue_size = 5, time_out= 0.5, frame_stride= 1, log_level= LogLevel.INFO))
logger = Logger("stream", LogLevel.INFO)

async def reader_start(index: int) -> bool:
    logger.log_info(f"Start the camera {index}")
    return await asyncio.to_thread(stream_readers[index].start)

async def reader_read(index: int) -> np.array:
    ret,frame = await asyncio.to_thread(stream_readers[index].read)
    if not ret or frame is None:
        return None
    else:
        return frame

async def reader_stop(index:int):
    await asyncio.to_thread(stream_readers[index].stop)

def is_reader_stop(index:int) -> bool:
    return stream_readers[index].is_stop()