// MESSAGE INPUT
const textarea = document.querySelector('.chatbox-message-input');
const chatboxForm = document.querySelector('.chatbox-message-form');
const today = new Date();
let idleTimer;

textarea.addEventListener('input', function () {
    clearTimeout(idleTimer);
    const line = textarea.value.split('\n').length;

    textarea.rows = Math.min(line, 6);

    chatboxForm.style.alignItems = textarea.rows > 1 ? 'flex-end' : 'center';
});

// Listen for "Enter" key press
textarea.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        submitForm();
    }
});

// Initial idle time pag wala pang conversation
function startIdleTimer() {
    idleTimer = setTimeout(() => {
        submitAutomaticResponse();
    }, 5000); // 5 seconds
}

function resetIdleTimer() {
    clearTimeout(idleTimer);
    startIdleTimer();
}

// TOGGLE CHATBOX
const chatboxToggle = document.querySelector('.chatbox-toggle');
const chatboxMessage = document.querySelector('.chatbox-message-wrapper');
let chatbotDisplay = 0;
let isAutoResponseDisplayed = 0;
const csrfToken = document.querySelector('input[name=csrfmiddlewaretoken]').value;

function updateConvo() {
    fetch('/chatbot_display/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfToken,
        },
        body: JSON.stringify({ chatbotDisplay }),
    });
}

// Show options-container by default and hide chat-container
document.querySelector('.options-container').style.display = 'block';
document.querySelector('.chat-container').style.display = 'none';

document.querySelectorAll('.topic-button').forEach(button => {
    button.addEventListener('click', function () {
        // Toggle visibility of options-container and chat-container
        document.querySelector('.options-container').style.display = 'none';
        document.querySelector('.chat-container').style.display = 'block';

        if (chatbotDisplay == 0) {
            clearTimeout(idleTimer);
            if (isAutoResponseDisplayed == 0) {
                startIdleTimer();
                chatbotDisplay = 1;
                isAutoResponseDisplayed = 1;
                updateConvo();
                disableRadioButtons(false);
                resetRadioButtons();
            }
        } else {
            chatbotDisplay = 0;
            updateConvo();
        }
    });
});


// here is the part of the changes needed for script.js

// PROCESS FEEDBACK 
const radioButtons = document.getElementsByName('rating');

function processFeedback() {
    const feedbackInputs = document.querySelector("input[name='rating']:checked").value;
    
    if(feedbackInputs) {
        console.log(feedbackInputs);
        fetch('/get_feedback/', {
            method: "POST",
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken,
            },
            body: JSON.stringify({feedbackInputs}),
        })
        .then(response => {
            if (response.ok) {
                displayReceivedMessage("Thank you for your feedback!");
                disableRadioButtons(true);
                return "feedback success";
            } else {
                throw new Error('No Model found here in my JS code');
            }

        })
        updateConvo();
    }
}

function disableRadioButtons(disabled = true){
    for (i = 0; i < radioButtons.length; i++){
        if(radioButtons[i].type = "radio") {
            radioButtons[i].disabled = disabled;
        }
    }
}

function resetRadioButtons() {
    for (i = 0; i < radioButtons.length; i++){
        if(radioButtons[i].type = "radio") {
            radioButtons[i].checked = false;
        }
    }
}

// DROPDOWN TOGGLE (edited for metrics data)
const dropdownToggle = document.querySelector('.chatbox-message-dropdown-toggle');
const dropdownMenu = document.querySelector('.chatbox-message-dropdown-menu');

dropdownToggle.addEventListener('click', function () {
    dropdownMenu.classList.toggle('show');
});

document.addEventListener('click', function (e) {
    if (!e.target.matches('.chatbox-message-dropdown, .chatbox-message-dropdown *')) {
        dropdownMenu.classList.remove('show');
    }
});

// CHATBOX MESSAGE
// const csrfToken = document.querySelector('input[name=csrfmiddlewaretoken]').value;
const chatboxMessageWrapper = document.querySelector('.chatbox-message-content');
const chatboxNoMessage = document.querySelector('.chatbox-message-no-message');

chatboxForm.addEventListener('submit', function (e) {
    e.preventDefault();
    resetInactivityTimer();
    submitForm();
});

// Inactivity timer
let inactivityTimer;

// Flag to track if typing indicator is displayed
let isTypingIndicatorDisplayed = false;

// // Reset Idle time kapag may conversation na
// function resetInactivityTimer() {
//     clearTimeout(inactivityTimer);
//     // inactivityTimer = setTimeout(function () {
//     //     sendMessageToServer({ userMessage: '' });
//     // }, 5000); // 5 seconds
//     idleTimer = setTimeout(() => {
//         submitAutomaticResponse();
//     }, 5000); // 5 seconds
// }


function resetInactivityTimer() {
    if (isValid(textarea.value)) {
        clearTimeout(inactivityTimer);
        resetInactivityTimerWithFeedback();
    }
}

function submitForm() {
    if (isValid(textarea.value)) {
        const userMessage = textarea.value;
        displaySentMessage(userMessage);
        displayTypingIndicator();
        textarea.value = '';
        textarea.rows = 1;
        textarea.focus();
        chatboxNoMessage.style.display = 'none';

        fetch('/ai-response/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken,
            },
            body: JSON.stringify({ userMessage }),
        })
        .then(response => {
            if (response.ok) {
                return response.text();
            } else {
                throw new Error('No Model found here in my JS code');
            }
        })
        .then(data => {
            isAutoResponseDisplayed = 0;
            hideTypingIndicator();
            displayReceivedMessage(data);
            scrollBottom();
            disableRadioButtons(false);
            resetRadioButtons();
            resetInactivityTimerWithFeedback(); // Reset the idle timer after receiving AI response
            
        })
        .catch(error => {
            console.error('Error:', error);
            hideTypingIndicator();
            displayReceivedMessage('Sorry for the inconvenience. I am encountering an error in generating a response. Let me try again or escalate this to my support team for further assistance');
            scrollBottom();
        });
    }
}

// Function to send automatic response to django backend
function djangoIdleResponse(idle_response){
    fetch('/idle_response/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfToken,
        },
        body: JSON.stringify({idle_response}), 
    });
};

// Function to handle automatic response
function submitAutomaticResponse() {
    const automaticResponse = "Hello! I'm here and ready to chat whenever you are. What's on your mind?";
    displayReceivedMessage(automaticResponse);
    djangoIdleResponse(automaticResponse);
    scrollBottom();
}

function displaySentMessage(message) {
    const currentTime = new Date(); // Get the current time

    const userMessageElement = `
        <div class="chatbox-message-item sent">
            <div class="chatbox-message-item-text">
                ${message}
            </div>
            <span class="chatbox-message-item-time">Sent: ${addZero(currentTime.getHours())}:${addZero(currentTime.getMinutes())}</span>
        </div>
    `;

    chatboxMessageWrapper.insertAdjacentHTML('beforeend', userMessageElement);
    scrollBottom();
}

function displayReceivedMessage(message) {
    const currentTime = new Date(); // Get the current time

    const aiResponseElement = `
        <div class="chatbox-message-item received">
            <div class="bot">
                <div class="chatbox-message-item-text">
                    ${message}
                </div>
                <span class="chatbox-message-item-time">Replied: ${addZero(currentTime.getHours())}:${addZero(currentTime.getMinutes())}</span>
            </div>
        </div>
    `;

    chatboxMessageWrapper.insertAdjacentHTML('beforeend', aiResponseElement);
    scrollBottom();
}

function displayFeedbackMessage(message) {
    const currentTime = new Date(); // Get the current time

    const aiResponseElement = `
        <div class="chatbox-message-feedback">
            <div class="chatbox-message-item received">
                <div class="bot">
                    <div class="chatbox-message-item-text">
                        ${message}
                    </div>
                    <span class="chatbox-message-item-time">Replied: ${addZero(currentTime.getHours())}:${addZero(currentTime.getMinutes())}</span>
                </div>
            </div>
            <div class="feedback-wrapper">
                <label for="super-sad">
                    <input type="radio" name="rating" class="super-sad" id="super-sad" value="1" onclick="processFeedback()" />
                    <svg viewBox="0 0 24 24"><path d="M12,2C6.47,2 2,6.47 2,12C2,17.53 6.47,22 12,22A10,10 0 0,0 22,12C22,6.47 17.5,2 12,2M12,20A8,8 0 0,1 4,12A8,8 0 0,1 12,4A8,8 0 0,1 20,12A8,8 0 0,1 12,20M16.18,7.76L15.12,8.82L14.06,7.76L13,8.82L14.06,9.88L13,10.94L14.06,12L15.12,10.94L16.18,12L17.24,10.94L16.18,9.88L17.24,8.82L16.18,7.76M7.82,12L8.88,10.94L9.94,12L11,10.94L9.94,9.88L11,8.82L9.94,7.76L8.88,8.82L7.82,7.76L6.76,8.82L7.82,9.88L6.76,10.94L7.82,12M12,14C9.67,14 7.69,15.46 6.89,17.5H17.11C16.31,15.46 14.33,14 12,14Z" /></svg>
                </label>
        
                <label for="sad">
                    <input type="radio" name="rating" class="sad" id="sad" value="2" onclick="processFeedback()"  />
                    <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" width="100%" height="100%" viewBox="0 0 24 24"><path d="M20,12A8,8 0 0,0 12,4A8,8 0 0,0 4,12A8,8 0 0,0 12,20A8,8 0 0,0 20,12M22,12A10,10 0 0,1 12,22A10,10 0 0,1 2,12A10,10 0 0,1 12,2A10,10 0 0,1 22,12M15.5,8C16.3,8 17,8.7 17,9.5C17,10.3 16.3,11 15.5,11C14.7,11 14,10.3 14,9.5C14,8.7 14.7,8 15.5,8M10,9.5C10,10.3 9.3,11 8.5,11C7.7,11 7,10.3 7,9.5C7,8.7 7.7,8 8.5,8C9.3,8 10,8.7 10,9.5M12,14C13.75,14 15.29,14.72 16.19,15.81L14.77,17.23C14.32,16.5 13.25,16 12,16C10.75,16 9.68,16.5 9.23,17.23L7.81,15.81C8.71,14.72 10.25,14 12,14Z" /></svg>
                </label>
    
                <label for="neutral">
                    <input type="radio" name="rating" class="neutral" id="neutral" value="3" onclick="processFeedback()"  />
                    <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" width="100%" height="100%" viewBox="0 0 24 24"><path d="M8.5,11A1.5,1.5 0 0,1 7,9.5A1.5,1.5 0 0,1 8.5,8A1.5,1.5 0 0,1 10,9.5A1.5,1.5 0 0,1 8.5,11M15.5,11A1.5,1.5 0 0,1 14,9.5A1.5,1.5 0 0,1 15.5,8A1.5,1.5 0 0,1 17,9.5A1.5,1.5 0 0,1 15.5,11M12,20A8,8 0 0,0 20,12A8,8 0 0,0 12,4A8,8 0 0,0 4,12A8,8 0 0,0 12,20M12,2A10,10 0 0,1 22,12A10,10 0 0,1 12,22C6.47,22 2,17.5 2,12A10,10 0 0,1 12,2M9,14H15A1,1 0 0,1 16,15A1,1 0 0,1 15,16H9A1,1 0 0,1 8,15A1,1 0 0,1 9,14Z" /></svg>
                </label>

                <label for="happy">
                    <input type="radio" name="rating" class="happy" id="happy" value="4" onclick="processFeedback()"  />
                    <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" width="100%" height="100%" viewBox="0 0 24 24"><path d="M20,12A8,8 0 0,0 12,4A8,8 0 0,0 4,12A8,8 0 0,0 12,20A8,8 0 0,0 20,12M22,12A10,10 0 0,1 12,22A10,10 0 0,1 2,12A10,10 0 0,1 12,2A10,10 0 0,1 22,12M10,9.5C10,10.3 9.3,11 8.5,11C7.7,11 7,10.3 7,9.5C7,8.7 7.7,8 8.5,8C9.3,8 10,8.7 10,9.5M17,9.5C17,10.3 16.3,11 15.5,11C14.7,11 14,10.3 14,9.5C14,8.7 14.7,8 15.5,8C16.3,8 17,8.7 17,9.5M12,17.23C10.25,17.23 8.71,16.5 7.81,15.42L9.23,14C9.68,14.72 10.75,15.23 12,15.23C13.25,15.23 14.32,14.72 14.77,14L16.19,15.42C15.29,16.5 13.75,17.23 12,17.23Z" /></svg>
                </label>
      
                <label for="super-happy">
                    <input type="radio" name="rating" class="super-happy" id="super-happy" value="5" onclick="processFeedback()"  />
                    <svg viewBox="0 0 24 24"><path d="M12,17.5C14.33,17.5 16.3,16.04 17.11,14H6.89C7.69,16.04 9.67,17.5 12,17.5M8.5,11A1.5,1.5 0 0,0 10,9.5A1.5,1.5 0 0,0 8.5,8A1.5,1.5 0 0,0 7,9.5A1.5,1.5 0 0,0 8.5,11M15.5,11A1.5,1.5 0 0,0 17,9.5A1.5,1.5 0 0,0 15.5,8A1.5,1.5 0 0,0 14,9.5A1.5,1.5 0 0,0 15.5,11M12,20A8,8 0 0,1 4,12A8,8 0 0,1 12,4A8,8 0 0,1 20,12A8,8 0 0,1 12,20M12,2C6.47,2 2,6.5 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2Z" /></svg>
                </label>
            </div>
        </div>
    `;

    // const aiResponseElement = `
    //     <div class="chatbox-message-item received">
    //         <div class="bot">
    //             <div class="chatbox-message-item-text">
    //                 ${message}
    //             </div>
    //             <span class="chatbox-message-item-time">Replied: ${addZero(currentTime.getHours())}:${addZero(currentTime.getMinutes())}</span>
    //         </div>
    //         <div class="feedback-wrapper">
    //             <input type="radio" name="rating1" />
    //             <input type="radio" name="rating2" />
    //             <input type="radio" name="rating3" />
    //             <input type="radio" name="rating4" />
    //             <input type="radio" name="rating5" />
    //         </div>
    //     </div>
    // `;

    chatboxMessageWrapper.insertAdjacentHTML('beforeend', aiResponseElement);
    scrollBottom();
}

function displayTypingIndicator() {
    if (!isTypingIndicatorDisplayed) {
        isTypingIndicatorDisplayed = true;

        const typingIndicator = `
            <div class="chatbox-message-item received typing-indicator">
                <span class="chatbox-message-item-text typing-indicator-text">AI is typing<span class="dots"></span></span>
            </div>
        `;

        chatboxMessageWrapper.insertAdjacentHTML('beforeend', typingIndicator);

        startTypingAnimation();

        // Disable inactivityTimer when typing indicator is displayed
        clearTimeout(inactivityTimer);
        scrollBottom();
    }
}

// Function to start the typing animation
function startTypingAnimation() {
    const dotsElement = document.querySelector('.typing-indicator-text .dots');

    if (dotsElement) {
        dotsElement.innerHTML = '.';

        setInterval(() => {
            dotsElement.innerHTML += '.';
            if (dotsElement.innerHTML.length > 5) {
                dotsElement.innerHTML = '.';
            }
        }, 1000);
    }
}

function hideTypingIndicator() {
    const typingIndicator = document.querySelector('.chatbox-message-item.received.typing-indicator');
    if (typingIndicator) {
        typingIndicator.remove();
        isTypingIndicatorDisplayed = false;

        // Reset inactivityTimer when typing indicator is removed
        resetInactivityTimerWithFeedback();
    }
}

function addZero(num) {
    return num < 10 ? '0' + num : num;
}

function scrollBottom() {
    chatboxMessageWrapper.scrollTo(0, chatboxMessageWrapper.scrollHeight);
}

function isValid(value) {
    let text = value.replace(/\n/g, '');
    text = text.replace(/\s/g, '');

    return text.length > 0;
}

// Function to handle feedback response
function submitFeedbackResponse() {
    const feedbackResponse = "Is the information meeting your needs? Your feedback is valuable in enhancing our conversation.";
    // displayReceivedMessage(feedbackResponse);
    displayFeedbackMessage(feedbackResponse);
    djangoIdleResponse(feedbackResponse);
    scrollBottom();
}

// Reset Idle time when there's an ongoing conversation
function resetInactivityTimerWithFeedback() {
    clearTimeout(inactivityTimer);
    inactivityTimer = setTimeout(function () {
        submitFeedbackResponse();
    }, 2000); // 10 seconds
}

function goBackToTopics() {
    document.getElementById('optionsContainer').style.display = 'flex';
    document.getElementById('chatContainer').style.display = 'none';
    clearChat();
}



// for online and offline status
function updateChatboxStatus() {
    fetch('/get_chatbox_status/')
        .then(response => response.json())
        .then(data => {
            const chatboxStatusText = document.querySelector('.chatbox-status-text');
            const chatboxMessageLight = document.querySelector('.chatbox-message-light');

            if (data.chatbox_status) {
                chatboxStatusText.innerText = 'online';
                chatboxMessageLight.style.backgroundColor = 'var(--green)';
            } else {
                chatboxStatusText.innerText = 'offline';
                chatboxMessageLight.style.backgroundColor = 'var(--red)';
            }
        })
        .catch(error => {
            console.error('Error fetching chatbox status:', error);
        });
}

// function to update the chatbox status periodically
setInterval(updateChatboxStatus, 5000); 
