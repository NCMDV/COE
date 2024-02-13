let currentTopic = "";

function selectTopic(topic) {
    currentTopic = topic;
    document.getElementById('optionsContainer').style.display = 'none';
    document.getElementById('chatContainer').style.display = 'flex';
    displayMessage(`Welcome to the ${topic} chat!`);
}

function goBackToTopics() {
    document.getElementById('optionsContainer').style.display = 'flex';
    document.getElementById('chatContainer').style.display = 'none';
    clearChat();
}

function sendMessage() {
    const userInput = document.getElementById('userInput').value;
    if (userInput.trim() === "") return;

    displayMessage(`User: ${userInput}`);
    simulateAIResponse(userInput);
    document.getElementById('userInput').value = "";
}

function simulateAIResponse(userInput) {
    // Simulate AI response based on the selected topic
    let aiResponse = "AI: ";
    switch (currentTopic) {
        case "ticket_create":
            aiResponse += "Let's create a ticket. Please provide the details";
            break;
        case "training_doc":
            aiResponse += "Here is the link for the training documents";
            break;
        case "inquiry":
            aiResponse += "Ask me anything";
            break;
        default:
            aiResponse += "I'm not sure how to respond to that.";
    }

    displayMessage(aiResponse);
}

function displayMessage(message) {
    const chatContainer = document.getElementById('chat');
    const messageElement = document.createElement('div');
    messageElement.textContent = message;
    chatContainer.appendChild(messageElement);
    chatContainer.scrollTop = chatContainer.scrollHeight; // Auto-scroll to the bottom
}

function clearChat() {
    const chatContainer = document.getElementById('chat');
    while (chatContainer.firstChild) {
        chatContainer.removeChild(chatContainer.firstChild);
    }
}





