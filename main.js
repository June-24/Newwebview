window.addEventListener('DOMContentLoaded', (event) => {
    document.getElementById('Submit').addEventListener('click', Help);
    document.getElementById('userInput').addEventListener('keypress', function (e) {
        if (e.key === 'Enter') {
            Help();
        }
    });
});

let query = '';

function Help() {
    const inputField = document.getElementById('userInput');
    const message = inputField.value.trim();
    query = message;
    console.log(message);

    if (message === "") return;

    const chatWindow = document.getElementById('display');

    const userMessageDiv = document.createElement('div');
    userMessageDiv.classList.add('message', 'user-message');
    userMessageDiv.textContent = message;
    chatWindow.append(userMessageDiv);

    inputField.value = "";
    chatWindow.scrollTop = chatWindow.scrollHeight;

    fetchresponse();
}

//send url to proxy-server to receive div content

// async function sendUrlToServer(url) {
//     const apiUrl = 'http://localhost:5001/content'; // Assuming your server is running on port 5001
//     const requestData = {
//         url: url
//     };

//     const requestOptions = {
//         method: 'POST',
//         headers: {
//             'Content-Type': 'application/json'
//         },
//         body: JSON.stringify(requestData)
//     };

//     try {
//         const response = await fetch(apiUrl, requestOptions);
//         if (!response.ok) {
//             throw new Error(`HTTP error! Status: ${response.status}`);
//         }
//         const responseData = await response.text();
//         console.log('Server response:', responseData);
//         // Handle the response data here
//     } catch (error) {
//         console.error('Error sending URL to server:', error);
//         // Handle error here
//     }
// }




async function fetchresponse() {
    const url = `http://127.0.0.1:5000/api/send_message`;

    const formData = new FormData();
    formData.append('user_input', query);

    const options = {
        method: 'POST',
        body: formData
    };

    try {
        const response = await fetch(url, options);
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const result = await response.json();
        const data = result.response;
        console.log(result.response);

        const chatWindow = document.getElementById('display');
        const botMessageDiv = document.createElement('div');
        const buttonn = document.createElement('button');
        buttonn.innerHTML = "Click me";

        buttonn.addEventListener('click', async function () {
            // const link = 'https://www.nasa.gov/humans-in-space/'; // Replace with the actual URL
            // try {
            //     const response = sendUrlToServer(link);
            //     console.log('Server response:', response);
            //     // Handle server response here
            //     console.log(response);
            // } catch (error) {
            //     console.error('Error sending link to server:', error);
            //     // Handle error here
            // }

            vscode.postMessage({
                type: 'newWebView',
                //suppose we get div from proxy server, can just send it
                //for now just sending a text div to check if working or not
                value: `<div class="grid-col-12 desktop:grid-col-6 desktop:padding-right-5 margin-bottom-6 desktop:margin-bottom-0">
                                <div class="margin-bottom-2">
                                        <h3 class="subtitle-md">Destinations</h3>
                                </div>
                                <div class="margin-bottom-2">
                                        <h2 class="display-48">Earth, Moon, and Mars</h2>
                                </div>
                                <div class="margin-bottom-2">
                                        <p class="heading-18 line-height-md">With more than 20 years of operations in low Earth orbit, we are preparing our return to the Moon for long-term exploration and discovery before taking the next giant leap to Mars. </p>
                                </div>
                                <p class="p-md">Never has humanity endeavored to simultaneously architect multinational infrastructures in lunar orbit, on the lunar surface, and at Mars — all while maintaining high-demand government and private-sector operations in low Earth orbit.</p>
                                <a href="https://www.nasa.gov/humans-in-space/destinations/" target="_self" class="button-primary button-primary-md">
                                        <span class="line-height-alt-1">
                        Destinations <span class="usa-sr-only">about Earth, Moon, and Mars</span>                    </span>
                                        <svg viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg"><circle class="button-primary-circle" cx="16" cy="16" r="16"></circle><path d="M8 16.956h12.604l-3.844 4.106 1.252 1.338L24 16l-5.988-6.4-1.252 1.338 3.844 4.106H8v1.912z" class="color-spacesuit-white"></path></svg>     
                                </a>
                        </div>`,
            });

        });
        botMessageDiv.classList.add('message', 'bot-message');
        // botMessageDiv.textContent = "Chatbot" + data;
        botMessageDiv.innerHTML = `<strong>Bot:</strong><br>${data}`;
        chatWindow.appendChild(buttonn);
        // botMessageDiv.textContent = data;
        chatWindow.appendChild(botMessageDiv);

        chatWindow.scrollTop = chatWindow.scrollHeight;
    } catch (error) {
        console.error('Error:', error);
    }
}
