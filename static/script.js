// Wait for the entire webpage to load before running the script
document.addEventListener('DOMContentLoaded', () => {

    // Get references to all our HTML elements
    const imageInput = document.getElementById('imageInput');
    const uploadArea = document.getElementById('upload-area');
    const uploadText = document.getElementById('upload-text');
    const analyzeButton = document.getElementById('analyzeButton');
    
    const resultsSection = document.getElementById('results-section');
    const originalPreview = document.getElementById('original-preview');
    const resultImage = document.getElementById('result-image');
    const spinnerBox = document.getElementById('spinner-box');

    let selectedFile = null;

    // --- Function to handle the selected file ---
    function handleFile(file) {
        if (file) {
            selectedFile = file;

            // Update the upload box text to show the file name
            uploadText.innerText = selectedFile.name;
            
            // Show the "Analyze" button
            analyzeButton.style.display = 'block';

            // Use FileReader to read the file and create a preview
            const reader = new FileReader();
            reader.onload = (e) => {
                originalPreview.src = e.target.result;
                originalPreview.style.display = 'block'; // Show the preview image
            };
            reader.readAsDataURL(selectedFile);

            // Show the results section so the user can see their preview
            resultsSection.style.display = 'block';
            
            // Hide any old results and the spinner
            resultImage.style.display = 'none';
            spinnerBox.style.display = 'none';
        }
    }

    // --- 1. Handle File Selection (Click) ---
    imageInput.addEventListener('change', (event) => {
        handleFile(event.target.files[0]);
    });

    // --- 2. Handle Drag & Drop ---
    // Prevent default browser behavior for drag-and-drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Highlight the drop area when a file is dragged over
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => uploadArea.classList.add('dragover'), false);
    });

    // Remove highlight when the file is no longer dragged over
    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => uploadArea.classList.remove('dragover'), false);
    });

    // Handle the file drop
    uploadArea.addEventListener('drop', (event) => {
        const dt = event.dataTransfer;
        const file = dt.files[0];
        
        // Manually set the input's files property
        imageInput.files = dt.files;
        
        // Trigger the 'change' event to run our preview logic
        const changeEvent = new Event('change');
        imageInput.dispatchEvent(changeEvent);

    }, false);


    // --- 3. Handle the "Analyze" Button Click ---
    analyzeButton.addEventListener('click', async () => {
        if (!selectedFile) {
            alert('Please select an image file first.');
            return;
        }

        // Show the loading spinner and hide the old result
        spinnerBox.style.display = 'flex';
        resultImage.style.display = 'none';

        // Create a FormData object to send the file
        const formData = new FormData();
        formData.append('image', selectedFile);

        try {
            // Send the file to our Flask backend
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Server error: ' + response.statusText);
            }

            // Get the analyzed image back as a 'blob' (a file-like object)
            const imageBlob = await response.blob();
            
            // Create a local URL for the blob
            const imageURL = URL.createObjectURL(imageBlob);

            // Put the new image on the page
            resultImage.src = imageURL;
            resultImage.style.display = 'block';

        // --- THIS IS THE FIX ---
        // Removed the extra "S" after (error)
        } catch (error) { 
            console.error('Error:', error);
            alert('An error occurred. Please try again.');
        } finally {
            // ALWAYS hide the spinner when done
            spinnerBox.style.display = 'none';
        }
    });
});