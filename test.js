async function sendUrlToServer(url) {
    const apiUrl = 'http://localhost:5001/content'; // Assuming your server is running on port 5001
    const requestData = {
        url: url
    };

    const requestOptions = {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
    };

    try {
        const response = await fetch(apiUrl, requestOptions);
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const responseData = await response.text();
        console.log('Server response:', responseData);
        // Handle the response data here
    } catch (error) {
        console.error('Error sending URL to server:', error);
        // Handle error here
    }
}

// Example usage:
const urlToSend = 'https://www.nasa.gov/humans-in-space/'; // Replace with the URL you want to send
sendUrlToServer(urlToSend);
