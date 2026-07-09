# Reader Module Documentation

## 1. Purpose
Decouples frame acquisition from model inference by reading frames on a background thread and transferring them to the main thread via a thread-safe queue.

## 2. Queuing Theory & Stability
To prevent infinite queue growth and video lag, the effective frame arrival rate ($\lambda_{\text{eff}}$) must not exceed the processing service rate ($\mu$):

$$\lambda_{\text{eff}} \le \mu \implies \frac{\lambda}{S} \le \frac{1}{T_{\text{inference}}}$$

Where:
* $\lambda$ = Input frame rate (FPS) from the video/camera source.
* $S$ = Frame stride (skip factor).
* $T_{\text{inference}}$ = Inference processing time per frame (seconds).
* $\mu$ = Service rate ($1 / T_{\text{inference}}$).

**Minimum Stable Stride ($S$):**
$$S \ge \lambda \times T_{\text{inference}}$$

---

## 3. Key Concurrency & Multithreading Components

* **Background Worker Thread (`_update`)**: A daemon thread that continuously decodes frames from OpenCV and pushes them to the queue.
* **Thread-Safe Queue (`queue.Queue`)**: Holds the decoded frames. Pushing is done outside manual locks to prevent nested lock overhead.
* **Condition Variable (`self.cv`)**: Manages thread synchronization:
  * The worker thread blocks on `self.cv.wait()` when the queue is full.
  * The consumer thread wakes the worker using `self.cv.notify()` when the queue size drops below the low watermark threshold ($20\%$).
* **Safe Queue Reads (`read`)**: Uses a blocking read with a timeout (`q.get(block=True, timeout=...)`) outside of the lock block to avoid locking out the worker.
* **Thread Shutdown (`stop`)**: Acquires `self.cv` and calls `self.cv.notify_all()` to ensure the worker thread is not left sleeping forever.
* **Stream Reconnection**: Automatically releases the capture and reconnects if consecutive read failures occur, preventing permanent stream drops.
