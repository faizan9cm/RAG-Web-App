document.getElementById('questionForm').addEventListener('submit', function(event) {
    event.preventDefault();
    
    var question = document.getElementById('question').value;

    fetch('http://127.0.0.1:5000/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ question: question })
    })
    .then(response => response.json())
    .then(data => {
        var answerContainer = document.getElementById('answerContainer');
        answerContainer.innerHTML = ''; // Clear previous content
        
        var answerSentences = data.answer;
        answerSentences.forEach(sentence => {
            var p = document.createElement('p');
            p.textContent = sentence;
            answerContainer.appendChild(p);
        });
        
        answerContainer.style.display = 'block';
    })
    .catch(error => console.error('Error:', error));
});
