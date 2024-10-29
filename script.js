document.addEventListener('DOMContentLoaded', function() {
    // Handle the first "Read More" button
    document.getElementById('readMoreBtn').addEventListener('click', function() {
        var moreText = document.getElementById('moreText');
        if (moreText.classList.contains('hidden')) {
            moreText.classList.remove('hidden');
            this.textContent = 'Read Less';
        } else {
            moreText.classList.add('hidden');
            this.textContent = 'Read More';
        }
    });

    // Handle all "Read More" buttons
    document.querySelectorAll('.readMoreBtn').forEach(function(button) {
        button.addEventListener('click', function() {
            var moreText = this.parentElement.nextElementSibling;
            if (moreText.classList.contains('hidden')) {
                moreText.classList.remove('hidden');
                this.textContent = 'Read Less';
            } else {
                moreText.classList.add('hidden');
                this.textContent = 'Read More';
            }
        });
    });

    // Handle FAQ toggles
    const toggles = document.querySelectorAll('.faq-toggle');
    const answers = document.querySelectorAll('.faq-answer');

    toggles.forEach(toggle => {
        toggle.addEventListener('click', function() {
            // Close all open answers
            answers.forEach(answer => {
                answer.classList.remove('show');
            });

            // Open the clicked one
            const answer = this.parentElement.nextElementSibling;
            if (answer.classList.contains('show')) {
                answer.classList.remove('show');
            } else {
                answer.classList.add('show');
            }
        });
    });
});



