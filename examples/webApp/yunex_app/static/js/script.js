// Variables for topDown view
let topDownPointsToDrawInCanvas = [];
let topDownPixelPointsJson = [];
let topDownGpsPointsJson = [];
let highlightedPointIndexOnClick = -1;
let highlightedLineIndexOnHover = -1;
let defaultBirdViewFilePath = "";
let img = new Image();

// Variables for perspective camera view
let selectedCameraViews = [];
let activeCameraViewIndex = 0;
// For handling with perspective camera views points
let cameraViewSelectedPoint = [];
let perspectiveCameraViewsPointsJson = [];

// Video stream
let image = new Image();
const targetWidth = 800; // Target width (adjust as needed)
const targetHeight = 450; // Target height (adjust as needed)

window.addEventListener("DOMContentLoaded", () => {

    // Initializing
    console.log("Javascript loaded!");
    for (let i = 0; i < 4; i++) {
        selectedCameraViews.push(new Image());
        cameraViewSelectedPoint.push([]);
        perspectiveCameraViewsPointsJson.push([]);
    }

    // Button for run processing
    document.getElementById("runButton").addEventListener("click", function () {
        console.log("Running background process");
        // Send an AJAX request to the Django view
        //fetch("{% url 'run_external_app' %}", {
        //    method: 'POST',
        //})
        //    .then(response => response.text())
        //    .then(data => {
        //        alert(data);
        //    })
        //    .catch(error => {
        //        alert("Error: " + error);
        //    });
    });

    // Websocket handling, incoming video stream
    const canvasForTopDownView = document.getElementById('topDownViewCanvas');
    const contextForTopDownView = canvasForTopDownView.getContext('2d');
    try {
        udpSocket = new WebSocket('ws://127.0.0.1:8050');

        if (udpSocket.readyState === WebSocket.CONNECTING) {
            console.log('WebSocket connection is in the process of opening');
            const defaultCanvasImage = canvasForTopDownView.getAttribute("src");
            image.src = defaultCanvasImage;

            image.onload = () => {
                contextForTopDownView.drawImage(image, 0, 0, canvasForTopDownView.width, canvasForTopDownView.height);
            };
        }

        udpSocket.onmessage = (event) => {
            const frame_data_base64 = event.data; // Assuming the data received is base64-encoded
            const image = new Image();

            image.onload = () => {
                contextForTopDownView.drawImage(image, 0, 0, canvasForTopDownView.width, canvasForTopDownView.height);
            };

            image.src = 'data:image/jpeg;base64,' + frame_data_base64;
        };
    } catch (error) {
        console.log(error);
    }

    // Horizontal chart handling
    var horizontalBarChartContext = document.getElementById("detectedObjectCounterChart").getContext('2d');
    // Define data for the horizontal bar chart
    var data = {
        labels: ["Person", "Car", "Bus", "Truck", "Train/Tram"],
        datasets: [{
            data: [0, 0, 0, 0, 0], // Update with the actual data
            backgroundColor: [
                'rgba(15, 150, 255, 0.2)',
                'rgba(75, 192, 192, 0.2)',
                'rgba(255, 99, 132, 0.2)',
                'rgba(255, 206, 86, 0.2)',
                'rgba(153, 102, 255, 0.2)'
            ], // Bar colors
            borderColor: [
                'rgba(15, 15, 255, 1)',
                'rgba(75, 192, 192, 1)',
                'rgba(255, 99, 132, 1)',
                'rgba(255, 206, 86, 1)',
                'rgba(153, 102, 255, 1)'
            ], // Border colors
            borderWidth: 1 // Border width
        }]
    };
    // Create a bar chart and set the axis to be horizontal
    var myHorizontalBarChart = new Chart(horizontalBarChartContext, {
        type: 'bar',
        data: data,
        options: {
            indexAxis: 'y', // Set the axis to be horizontal (y-axis)
            scales: {
                x: {
                    beginAtZero: true
                }
            },
            plugins: {
                legend: {
                    display: false, // Set to true to display the legend
                }
            }
        }
    });

    // Websocket handling for horizontal chart, detected object counting
    const socket = new WebSocket('ws://127.0.0.1:8051');
    socket.onmessage = (event) => {
        const newData = JSON.parse(event.data);
        myHorizontalBarChart.data.labels = newData["supportedClassesForCounter"];
        myHorizontalBarChart.data.datasets[0].data = newData["totalObjectsCountForCounter"];
        myHorizontalBarChart.update();
    };

    // Perspective 01 camera view handling
    const canvas01 = document.getElementById("canvas01");
    const context01 = canvas01.getContext("2d");

    // Perspective 02 camera view handling
    const canvas02 = document.getElementById("canvas02");
    const context02 = canvas02.getContext("2d");

    // Perspective 03 camera view handling
    const canvas03 = document.getElementById("canvas03");
    const context03 = canvas03.getContext("2d");

    // Perspective 04 camera view handling
    const canvas04 = document.getElementById("canvas04");
    const context04 = canvas04.getContext("2d");

    // Selected perspective camera view handling
    const canvasSelectedCameraView = document.getElementById("canvasSelectedCameraView");
    const contextSelectedCameraView = canvasSelectedCameraView.getContext("2d");

    // Top down view handling
    const canvas = document.getElementById("canvas");
    const context = canvas.getContext("2d");

    // Initialization of all canvas view
    // Load the image on the canvas
    const defaultCanvasImage = canvas.getAttribute("src");
    img.src = defaultCanvasImage;
    img.onload = () => {
        context.drawImage(img, 0, 0, canvas.width, canvas.height);

        context01.drawImage(img, 0, 0, canvas01.width, canvas01.height);
        context02.drawImage(img, 0, 0, canvas02.width, canvas02.height);
        context03.drawImage(img, 0, 0, canvas03.width, canvas03.height);
        context04.drawImage(img, 0, 0, canvas04.width, canvas04.height);

        contextSelectedCameraView.drawImage(img, 0, 0, canvasSelectedCameraView.width, canvasSelectedCameraView.height);
    };

    function convertCoordinatesToBiggerImage(
        x, // X-coordinate in the smaller image frame
        y, // Y-coordinate in the smaller image frame
        smallerWidth, // Width of the smaller image frame
        smallerHeight, // Height of the smaller image frame
        biggerWidth, // Width of the bigger image frame
        biggerHeight // Height of the bigger image frame
    ) {
        const scaleX = biggerWidth / smallerWidth;
        const scaleY = biggerHeight / smallerHeight;
        const biggerX = x * scaleX;
        const biggerY = y * scaleY;
        return { x: biggerX, y: biggerY };
    }

    // Function to handle mouse clicks and draw points
    canvas.addEventListener("click", (event) => {
        var fullSizeImageWidth = 1920;
        var fullSizeImageHeight = 1080;
        var resizedImageWidth = 800;
        var resizedImageHeight = 450;

        if (defaultBirdViewFilePath != "") {
            const rect = canvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;

            const scaleX = fullSizeImageWidth / resizedImageWidth;
            const scaleY = fullSizeImageHeight / resizedImageHeight;
            const fullSizeImagePointX = x * scaleX;
            const fullSizeImagePointY = y * scaleY;

            highlightedPointIndexOnClick = getHighlightedPointIndex(x, y);
            // State, when is clicked for new point (new point added)
            if (highlightedPointIndexOnClick == -1) {
                topDownPixelPointsJson.push({ x: fullSizeImagePointX, y: fullSizeImagePointY });
                topDownGpsPointsJson.push({ lat: 0, lng: 0 });

                addNewPointToGroupList("0", "0", topDownPointsToDrawInCanvas.length);
                if (topDownPointsToDrawInCanvas.length < 4) {
                    topDownPointsToDrawInCanvas.push({ x, y });
                    if (topDownPointsToDrawInCanvas.length > 1) {
                        //addNewPointToGroupList("0", "0", 0);
                    }
                }
                else {
                    topDownPointsToDrawInCanvas.pop();
                    topDownPointsToDrawInCanvas.push({ x, y });   
                }

                // Draw a total value of selected points
                setTotalValueToGroupList(topDownPointsToDrawInCanvas.length);

                // Connect last added point with first point
                if (topDownPointsToDrawInCanvas.length >= 3) {
                    topDownPointsToDrawInCanvas.push(topDownPointsToDrawInCanvas[0]);
                }
            }
            // State, when point is clicked
            else {
                document.getElementById("latitudeValue").value = topDownGpsPointsJson[highlightedPointIndexOnClick].lat;
                document.getElementById("longitudeValue").value = topDownGpsPointsJson[highlightedPointIndexOnClick].lng;
                showModal();
            }
            drawPointsOnCanvas();
        }
    });

    // Function to handle mousemove and highlight lines
    canvas.addEventListener("mousemove", (event) => {
        if (defaultBirdViewFilePath != "") {
            const rect = canvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;
            highlightedLineIndexOnHover = getHighlightedPointIndex(x, y);
            drawPointsOnCanvas();
        }
    });

    function drawPointsOnCanvas() {
        if (defaultBirdViewFilePath != "") {
            context.clearRect(0, 0, canvas.width, canvas.height);
            context.drawImage(img, 0, 0, canvas.width, canvas.height);

            // Draw points on the canvas
            context.fillStyle = "red";
            context.font = "normal normal normal 14px arial";
            pointIndex = 1;

            for (let i = 0; i < topDownPointsToDrawInCanvas.length; i++) {
                context.beginPath();
                context.arc(topDownPointsToDrawInCanvas[i].x, topDownPointsToDrawInCanvas[i].y, 5, 0, 2 * Math.PI);
                context.fill();

                // Connect last added point with first point
                if (topDownPointsToDrawInCanvas.length >= 3) {
                    if (i == (topDownPointsToDrawInCanvas.length - 1)) {
                        break;
                    }
                }
                // Draw point labels
                context.fillText("P " + pointIndex.toString(), topDownPointsToDrawInCanvas[i].x, topDownPointsToDrawInCanvas[i].y - 10);
                pointIndex = pointIndex + 1;
            }

            // Draw lines between consecutive points
            if (topDownPointsToDrawInCanvas.length > 1) {
                context.strokeStyle = "blue";
                context.lineWidth = 2;
                context.beginPath();
                context.moveTo(topDownPointsToDrawInCanvas[0].x, topDownPointsToDrawInCanvas[0].y);
                for (let i = 1; i < topDownPointsToDrawInCanvas.length; i++) {
                    context.lineTo(topDownPointsToDrawInCanvas[i].x, topDownPointsToDrawInCanvas[i].y);
                }
                context.stroke();
            }

            // Highlight the point on hover
            if (highlightedLineIndexOnHover !== -1 && topDownPointsToDrawInCanvas.length > 1) {              
                context.beginPath();
                context.strokeStyle = "green";
                context.arc(topDownPointsToDrawInCanvas[highlightedLineIndexOnHover].x, topDownPointsToDrawInCanvas[highlightedLineIndexOnHover].y, 8, 0, 2 * Math.PI);
                context.fill();
                context.stroke();
            }

            // Highlight the line on click
            if (highlightedPointIndexOnClick !== -1 && topDownPointsToDrawInCanvas.length > 1) {
                context.beginPath();
                context.strokeStyle = "green";
                context.arc(topDownPointsToDrawInCanvas[highlightedPointIndexOnClick].x, topDownPointsToDrawInCanvas[highlightedPointIndexOnClick].y, 8, 0, 2 * Math.PI);
                context.fill();
                context.stroke();
            }
        }
    }

    function getHighlightedPointIndex(x, y) {
        let closestDistance = 15;
        let index = -1;

        for (let i = 0; i < topDownPointsToDrawInCanvas.length - 1; i++) {
            const { x: x1, y: y1 } = topDownPointsToDrawInCanvas[i];
            const distance = pointToPointDistance(x, y, x1, y1);

            if (distance < closestDistance) {
                closestDistance = distance;
                index = i;
            }
        }
        return index;
    }

    function pointToPointDistance(x, y, x1, y1) {
        const dx = x - x1;
        const dy = y - y1;

        return Math.sqrt(dx * dx + dy * dy);
    }

    const clearLastPointBtnElement = document.getElementById("clearLastPointBtn");
    clearLastPointBtnElement.addEventListener("click", function () {
        drawPointsOnCanvas();
        setTotalValueToGroupList(topDownPointsToDrawInCanvas.length);
    });

    const clearAllPointsBtnElement = document.getElementById("clearAllPointsBtn");
    clearAllPointsBtnElement.addEventListener("click", function () {
        drawPointsOnCanvas();
        setTotalValueToGroupList(0);
    });


    // Get the input element for the number value
    const getLatitudeValue = document.getElementById("latitudeValue");

    const getLongitudeValue = document.getElementById("longitudeValue");

    // Get the "Save" button element
    const saveNumberButton = document.getElementById("saveNumberButton");
    // Variable to store the selected number
    let selectedLatitudeValue;
    let selectedLongitudeValue;

    // Event listener for the "Save" button click
    saveNumberButton.addEventListener("click", () => {
        // Get the value from the input field and convert it to a number
        selectedLatitudeValue = parseInt(getLatitudeValue.value);
        selectedLongitudeValue = parseInt(getLongitudeValue.value);

        // Check if the selected number is valid (not NaN)
        if (!isNaN(selectedLatitudeValue)) {
            // Do something with the selected number
            editPointInGroupList(selectedLatitudeValue.toString(), selectedLongitudeValue.toString(), highlightedPointIndexOnClick);
            closeModal();     
        }
    });

    

    const addItemBtn = document.getElementById("addItemBtn");
    const newItemInput = document.getElementById("newItem");
    const selectedPointsItemList = document.getElementById("selectedPointsItemList");

    // Add event listener for the input change event
    imageInputBirdView.addEventListener("change", function () {
        defaultBirdViewFilePath = imageInputBirdView.files[0];
        if (defaultBirdViewFilePath) {
            const reader = new FileReader();
            reader.onload = function (e) {
                //img = new Image();
                img.onload = function () {
                    context.drawImage(img, 0, 0, canvas.width, canvas.height);
                };
                img.src = e.target.result;
            };
            reader.readAsDataURL(defaultBirdViewFilePath);
        }
    });

    // Add event listener for the input change event
    imageInput01.addEventListener("change", function () {
        let imagePath = imageInput01.files[0];

        // Create an Image element
        const imgInput = new Image();

        if (imagePath) {
            const reader = new FileReader();
            reader.onload = function (e) {
                //img = new Image();
                imgInput.onload = function () {
                    context01.drawImage(imgInput, 0, 0, canvas01.width, canvas01.height);
                };
                imgInput.src = e.target.result;
                selectedCameraViews[0] = imgInput;
            };
            reader.readAsDataURL(imagePath);
        }      
    });

    // Event listener for the "Save" button click
    canvas01.addEventListener("click", () => {
        //let defaultCanvasImage = canvas.getAttribute("src");
        //let imggg = new Image();
        //imggg.src = defaultCanvasImage;
        //const dataURL = canvas01.toDataURL("image/jpeg", 1.0);

        // Create an Image element
        //const image = new Image();

        // Set the image source to the data URL
        //image.src = dataURL;
        //image.src = selectedCameraViews[0]
        activeCameraViewIndex = 0;
        contextSelectedCameraView.drawImage(selectedCameraViews[0], 0, 0, canvasSelectedCameraView.width, canvasSelectedCameraView.height);
        drawPointsOnCanvasForCameraView();
    });

    // Add event listener for the input change event
    imageInput02.addEventListener("change", function () {
        let imagePath = imageInput02.files[0];

        // Create an Image element
        const imgInput = new Image();

        if (imagePath) {
            const reader = new FileReader();
            reader.onload = function (e) {
                //img = new Image();
                imgInput.onload = function () {
                    context02.drawImage(imgInput, 0, 0, canvas02.width, canvas02.height);
                };
                imgInput.src = e.target.result;
                selectedCameraViews[1] = imgInput;
            };
            reader.readAsDataURL(imagePath);
        }
    });

    // Event listener for the "Save" button click
    canvas02.addEventListener("click", () => {
        activeCameraViewIndex = 1;
        contextSelectedCameraView.drawImage(selectedCameraViews[1], 0, 0, canvasSelectedCameraView.width, canvasSelectedCameraView.height);
        drawPointsOnCanvasForCameraView();
    });

    // Add event listener for the input change event
    imageInput03.addEventListener("change", function () {
        let imagePath = imageInput03.files[0];

        // Create an Image element
        const imgInput = new Image();

        if (imagePath) {
            selectedCameraViews[2] = imagePath;
            const reader = new FileReader();
            reader.onload = function (e) {
                //img = new Image();
                imgInput.onload = function () {
                    context03.drawImage(imgInput, 0, 0, canvas03.width, canvas03.height);
                };
                imgInput.src = e.target.result;
                selectedCameraViews[2] = imgInput;
            };
            reader.readAsDataURL(imagePath);
        }
    });

    // Event listener for the "Save" button click
    canvas03.addEventListener("click", () => {
        activeCameraViewIndex = 2;
        contextSelectedCameraView.drawImage(selectedCameraViews[2], 0, 0, canvasSelectedCameraView.width, canvasSelectedCameraView.height);
        drawPointsOnCanvasForCameraView();
    });

    // Add event listener for the input change event
    imageInput04.addEventListener("change", function () {
        let imagePath = imageInput04.files[0];

        // Create an Image element
        const imgInput = new Image();

        if (imagePath) {
            selectedCameraViews[3] = imagePath;
            const reader = new FileReader();
            reader.onload = function (e) {
                //img = new Image();
                imgInput.onload = function () {
                    context04.drawImage(imgInput, 0, 0, canvas04.width, canvas04.height);
                };
                imgInput.src = e.target.result;
                selectedCameraViews[3] = imgInput;
            };
            reader.readAsDataURL(imagePath);
        }
    });

    // Event listener for the "Save" button click
    canvas04.addEventListener("click", () => {
        activeCameraViewIndex = 3;
        contextSelectedCameraView.drawImage(selectedCameraViews[3], 0, 0, canvasSelectedCameraView.width, canvasSelectedCameraView.height);
        drawPointsOnCanvasForCameraView();
    });

    // Function to handle mouse clicks and draw points
    canvasSelectedCameraView.addEventListener("click", (event) => {
        var fullSizeImageWidth = 1920;
        var fullSizeImageHeight = 1080;
        var resizedImageWidth = 800;
        var resizedImageHeight = 450;

        //if (defaultBirdViewFilePath != "") {
            const rect = canvasSelectedCameraView.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;

            const scaleX = fullSizeImageWidth / resizedImageWidth;
            const scaleY = fullSizeImageHeight / resizedImageHeight;
            const fullSizeImagePointX = x * scaleX;
            const fullSizeImagePointY = y * scaleY;

        cameraViewSelectedPoint[activeCameraViewIndex].push({ x, y });
        perspectiveCameraViewsPointsJson[activeCameraViewIndex].push({x: fullSizeImagePointX, y: fullSizeImagePointY });

            drawPointsOnCanvasForCameraView();
        //}
    });

    //const contextSelectedCameraView = canvasSelectedCameraView.getContext("2d");
    function drawPointsOnCanvasForCameraView() {
        // Draw points on the canvas
        for (let i = 0; i < cameraViewSelectedPoint[activeCameraViewIndex].length; i++) {
            // Draw the point
            contextSelectedCameraView.beginPath();
            contextSelectedCameraView.arc(parseInt(cameraViewSelectedPoint[activeCameraViewIndex][i].x), parseInt(cameraViewSelectedPoint[activeCameraViewIndex][i].y), 5, 0, 2 * Math.PI);
            contextSelectedCameraView.fillStyle = 'green';
            contextSelectedCameraView.fill();
            contextSelectedCameraView.stroke();
        }
    }

    const saveJsonBtnElement = document.getElementById("saveJsonBtn");
    saveJsonBtnElement.addEventListener("click", function () {
        // Create a JSON object
        let jsonObject = {
            topDownViewPixelPoints: topDownPixelPointsJson,
            topDownViewGpsCoordinates: topDownGpsPointsJson,
            perspectiveCameraViewPoints: perspectiveCameraViewsPointsJson
        };

        // Convert to JSON string if needed
        let jsonString = JSON.stringify(jsonObject);

        // Logging the JSON string to console
        //console.log(jsonString);

        // Downlo
        downloadJSON(jsonString, 'config.json');
    });

});

// Download a JSON file
function downloadJSON(jsonString, fileName) {
    // Create a Blob from the JSON string
    var blob = new Blob([jsonString], { type: 'application/json' });

    // Create a link element
    var a = document.createElement('a');

    // Create a URL for the blob
    var url = window.URL.createObjectURL(blob);
    a.href = url;
    a.download = fileName;

    // Append the link to the body (required for Firefox)
    document.body.appendChild(a);

    // Programmatically click the link to trigger the download
    a.click();

    // Remove the link after starting the download
    document.body.removeChild(a);

    // Release the created URL
    window.URL.revokeObjectURL(url);
}

function clearLastPoint() {
    topDownPointsToDrawInCanvas.pop();
    topDownGpsPointsJson.pop();
    if (topDownPointsToDrawInCanvas.length >= 3) {
        topDownPointsToDrawInCanvas.pop();
        topDownGpsPointsJson.pop();

        if (topDownPointsToDrawInCanvas.length > 2) {
            topDownPointsToDrawInCanvas.push(topDownPointsToDrawInCanvas[0]);
            topDownGpsPointsJson.push(0);
        }
    }
    highlightedPointIndexOnClick = highlightedPointIndexOnClick - 1;
    highlightedLineIndexOnHover = highlightedLineIndexOnHover - 1;

    const selectedPointsItemList = document.getElementById("selectedPointsItemList");

    if (topDownGpsPointsJson.length > 2) {
        const lastItemForRemove = selectedPointsItemList.children[selectedPointsItemList.childElementCount - 1];
        selectedPointsItemList.removeChild(lastItemForRemove);

        let pointsDistance = topDownGpsPointsJson[topDownGpsPointsJson.length - 1];

        let pointsDistanceText = "Points P" + (selectedPointsItemList.childElementCount - 1).toString() + "-P1, " + "distance: " + pointsDistance.toString() + " cm";

        selectedPointsItemList.children[selectedPointsItemList.childElementCount - 1].innerText = pointsDistanceText;
        selectedPointsItemList.children[selectedPointsItemList.childElementCount - 1].value = pointsDistanceText;
    }
    else {
        let lastItemForRemove = selectedPointsItemList.children[selectedPointsItemList.childElementCount - 1];
        selectedPointsItemList.removeChild(lastItemForRemove);
        lastItemForRemove = selectedPointsItemList.children[selectedPointsItemList.childElementCount - 1];
        selectedPointsItemList.removeChild(lastItemForRemove);
    }
}

function clearAllPoints() {
    topDownPointsToDrawInCanvas.length = 0;
    topDownGpsPointsJson.length = 0;
    highlightedPointIndexOnClick = -1;
    highlightedLineIndexOnHover = -1;

    selectedPointsItemList.innerHTML = "";
}

// Function to show the modal popup
function showModal() {
    document.getElementById("latitudeValue").value = topDownGpsPointsJson[highlightedPointIndexOnClick].lat;
    document.getElementById("longitudeValue").value = topDownGpsPointsJson[highlightedPointIndexOnClick].lng;
    $('#numberModal').modal('show');
}

// Function to close the modal popup
function closeModal() {
    topDownGpsPointsJson[highlightedPointIndexOnClick] = { lat: document.getElementById("latitudeValue").value, lng: document.getElementById("longitudeValue").value };
    $("#numberModal").modal("hide");
}

function setTotalValueToGroupList(newItemInput) {
    document.getElementById("totalSelectedPointsInput").value = newItemInput;
    //addNewPointToGroupList(newItemInput);
}

function addNewPointToGroupList(latitude, longitude, index) {
    const selectedPointsItemList = document.getElementById("selectedPointsItemList");

    let pointsDistanceText = "";
    pointsDistanceText = "Points P" + index.toString() + " lat:" + latitude.toString() + " lng: " + longitude.toString();

    const listItem = document.createElement("a");
    listItem.classList.add("list-group-item");
    listItem.classList.add("list-group-item-action");
    listItem.textContent = pointsDistanceText;

    selectedPointsItemList.appendChild(listItem);
}

function editPointInGroupList(latitude, longitude, index) {
    const selectedPointsItemList = document.getElementById("selectedPointsItemList");
    let pointsDistanceText = "Points P" + (index + 1).toString() + " lat:" + latitude.toString() + " lng: " + longitude.toString();
    selectedPointsItemList.children[index + 1].innerText = pointsDistanceText;
    selectedPointsItemList.children[index + 1].value = pointsDistanceText;
}
