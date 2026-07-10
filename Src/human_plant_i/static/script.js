// Keep track of the active WebSocket connection (starts as null/empty)
let socket = null;

// Get references to HTML elements so we can control them using their ID attributes
const displayElement = document.getElementById("stream-display"); // The <img> element that displays the video
const titleElement = document.getElementById("stream-title");     // The hidden header element (keeps track of stream titles)

/**
 * Connects to the FastAPI WebSocket endpoint for a specific camera.
 * @param {string} cameraId - The ID of the camera to connect to (e.g. 'cam1' or 'cam2')
 */
function connectWebSocket(cameraId) {
    // If we are already connected to a camera, close that connection first before starting a new one
    if (socket) {
        socket.close();
    }

    // Determine if the website is running on secure HTTPS or HTTP,
    // then choose the correct WebSocket protocol ('wss://' for secure, 'ws://' for normal)
    const wsProtocol = window.location.protocol === "https:" ? "wss://" : "ws://";
    
    // Construct the URL to connect to the backend (e.g., ws://localhost:8000/ws/cam1)
    const wsUri = `${wsProtocol}${window.location.host}/ws/${cameraId}`;
    
    // Create a new WebSocket connection to the backend server
    socket = new WebSocket(wsUri);
    
    // Tell the browser to treat incoming binary data as "blobs" (raw binary files/images)
    socket.binaryType = "blob";

    // This event handler runs automatically every time a new image frame arrives from the server
    socket.onmessage = (event) => {
        const frameBlob = event.data;           // Grab the raw binary image data (JPEG blob)
        const currentUrl = displayElement.src;  // Save the URL of the frame currently being shown on the screen
        
        // Convert the raw binary data into a temporary web URL that the <img> tag can load,
        // and set it as the new source of the display image
        displayElement.src = URL.createObjectURL(frameBlob);
        
        // Memory Cleanup: If the previous frame's URL was a temporary blob URL,
        // delete it from browser memory so the browser doesn't slow down or crash over time.
        if (currentUrl.startsWith("blob:")) {
            URL.revokeObjectURL(currentUrl);
        }
    };

    // This event handler runs automatically when the WebSocket connection is closed
    socket.onclose = () => {
        console.log(`WebSocket detached from channel: ${cameraId}`);
    };
}

/**
 * Handles switching between different cameras when a user clicks a navigation button.
 * @param {string} cameraId - The ID of the clicked camera ('cam1' or 'cam2')
 */
function switchCamera(cameraId) {
    // Save the current image URL so we can clean it up
    const currentUrl = displayElement.src;
    
    // Clear the screen instantly so the user doesn't see the last frame of the old camera
    displayElement.src = "";
    
    // Memory Cleanup: Delete the old frame's blob URL from the browser's memory
    if (currentUrl && currentUrl.startsWith("blob:")) {
        URL.revokeObjectURL(currentUrl);
    }

    // Update the Navigation UI buttons styling:
    // 1. Remove the 'active' (green highlight) class from all navigation buttons
    document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
    
    // 2. Add the 'active' class to the button that was just clicked
    if (cameraId === 'cam1') {
        document.getElementById('btn-cam1').classList.add('active');
        titleElement.textContent = "Live Pipeline: CAMERA 01"; // Update the text (hidden)
    } else {
        document.getElementById('btn-cam2').classList.add('active');
        titleElement.textContent = "Live Pipeline: CAMERA 02"; // Update the text (hidden)
    }

    // Finally, establish a new WebSocket connection to retrieve the new camera's stream
    connectWebSocket(cameraId);
}

// When the page loads for the first time, automatically connect to CAMERA 01
connectWebSocket('cam1');
